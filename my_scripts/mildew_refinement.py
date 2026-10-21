"""Prepare reviewed Mildew crops and compare fixed-threshold detection with binary refinement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.patches import imwrite, torch_load
from ultralytics.utils.torch_utils import select_device

NAMES = {0: "BrownRust", 1: "Healthy", 2: "Loose-Smut", 3: "Mildew"}
MILDEW = 3


def crop_box(image, box, square=True):
    """Expand by ten percent per side, optionally padding the crop to a square."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    dx, dy = (x2 - x1) * 0.1, (y2 - y1) * 0.1
    x1, y1 = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
    x2, y2 = min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))
    crop = image[y1:y2, x1:x2]
    if not crop.size:
        raise ValueError(f"Empty crop: {box}")
    if not square:
        return crop
    h, w = crop.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(crop, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=(114,) * 3)


def snapshot(args, model=None):
    """Record the actual arguments, environment and detector hash beside the outputs."""
    info = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    info.update(ultralytics=ultralytics.__version__, torch=torch.__version__, matching_iou=0.5)
    if model is not None:
        info["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if getattr(args, "classifier", None):
        info["classifier_sha256"] = hashlib.sha256(args.classifier.read_bytes()).hexdigest()
    (args.output / "config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def inputs(args):
    """Use the existing dataset reader to validate annotations and resolve image paths."""
    data = check_det_dataset(args.data, autodownload=False)
    model = YOLO(str(args.model))
    if model.task != "detect" or model.names != NAMES or data["names"] != NAMES:
        raise ValueError(f"Expected detection classes {NAMES}; got model={model.names}, data={data['names']}")
    dataset = YOLODataset(
        img_path=data[args.split],
        data=data,
        task="detect",
        imgsz=args.imgsz,
        augment=False,
        rect=False,
        batch_size=1,
        cache=False,
    )
    return model, dataset


def predict(model, image, args, confidence):
    """Keep preprocessing and NMS identical for all threshold comparisons."""
    options = {}
    if getattr(args, "classifier_type", None) == "joint":
        from functools import partial

        from my_scripts.mildew_joint import JointPredictor

        options["predictor"] = partial(JointPredictor, binary_threshold=0)
    return model.predict(
        source=image,
        **options,
        conf=confidence,
        imgsz=args.imgsz,
        device=args.device,
        iou=0.7,
        agnostic_nms=False,
        max_det=300,
        rect=False,
        batch=1,
        verbose=False,
        save=False,
    )[0]


def ground_truth(label, image):
    """Convert the dataset owner's normalized labels to original-image xyxy coordinates."""
    height, width = image.shape[:2]
    boxes = ops.xywh2xyxy(torch.from_numpy(label["bboxes"]).float())
    boxes *= torch.tensor([width, height, width, height])
    return torch.cat((boxes, torch.from_numpy(label["cls"]).float()), dim=1)


def prepare(args):
    """Save real detector proposals and suggested labels for human review."""
    model, dataset = inputs(args)
    snapshot(args, model)
    groups = {}
    if args.groups:
        with args.groups.open(newline="", encoding="utf-8-sig") as file:
            groups = {row["image"]: row["group"] for row in csv.DictReader(file)}
        missing = {Path(label["im_file"]).name for label in dataset.labels} - groups.keys()
        if missing:
            raise ValueError(f"Groups CSV is missing {len(missing)} images, e.g. {sorted(missing)[:3]}")
    crops = args.output / "crops"
    crops.mkdir()
    fields = [
        "crop",
        "image",
        "source_group",
        "split",
        "det_score",
        "max_mildew_iou",
        "suggested_label",
        "reviewed_label",
    ]
    counts = Counter()
    with (args.output / "review.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, label in enumerate(dataset.labels):
            result = predict(model, label["im_file"], args, 0.10)
            boxes = result.boxes.data.cpu()
            boxes = boxes[boxes[:, 5] == MILDEW]
            gt = ground_truth(label, result.orig_img)
            gt = gt[gt[:, 4] == MILDEW, :4]
            overlaps = box_iou(boxes[:, :4], gt).max(dim=1).values if len(gt) else torch.zeros(len(boxes))
            image = Path(label["im_file"])
            group = groups.get(image.name, image.stem.split(".rf.")[0])
            bucket = int(hashlib.sha256(f"{args.seed}:{group}".encode()).hexdigest()[:8], 16) / 2**32
            split = "val" if bucket < 0.20 else "train"
            for box_index, (box, overlap) in enumerate(zip(boxes, overlaps)):
                name = f"{index:06d}_{box_index:03d}.png"
                if not imwrite(str(crops / name), crop_box(result.orig_img, box[:4])):
                    raise OSError(f"Failed to write {name}")
                suggested = "mildew" if overlap >= 0.5 else "non_mildew" if overlap <= 0.1 else "uncertain"
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            [
                                f"crops/{name}",
                                str(image),
                                group,
                                split,
                                float(box[4]),
                                float(overlap),
                                suggested,
                                "",
                            ],
                        )
                    )
                )
                counts[f"{split}/{suggested}"] += 1
            file.flush()
            if (index + 1) % 50 == 0:
                print(f"Prepared {index + 1}/{len(dataset)} images: {dict(counts)}", flush=True)
    print(f"Review {args.output / 'review.csv'}; set reviewed_label to mildew, non_mildew or skip. {dict(counts)}")


def build(args):
    """Build a classification dataset exclusively from reviewed, source-disjoint crops."""
    with args.review.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    selected = []
    group_splits = {}
    counts = Counter()
    for row in rows:
        label = row["reviewed_label"].strip()
        if label not in {"", "skip", "mildew", "non_mildew"}:
            raise ValueError(f"Invalid reviewed_label: {label!r}")
        if label in {"", "skip"}:
            continue
        split = row["split"]
        if split not in {"train", "val"}:
            raise ValueError(f"Invalid split: {split!r}")
        group = row["source_group"]
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Source group {group!r} occurs in both classification splits")
        group_splits[group] = split
        crop = args.review.parent / row["crop"]
        if not crop.is_file():
            raise FileNotFoundError(crop)
        counts[f"{split}/{label}"] += 1
        selected.append((crop, split, label))
    for split in ("train", "val"):
        for label in ("mildew", "non_mildew"):
            if counts[f"{split}/{label}"] < 2:
                raise ValueError(f"Need at least two reviewed {split}/{label} crops; got {dict(counts)}")
    for split in ("train", "val"):
        for label in ("mildew", "non_mildew"):
            (args.output / split / label).mkdir(parents=True)
    for index, (crop, split, label) in enumerate(selected):
        shutil.copy2(crop, args.output / split / label / f"{index:07d}.png")
    shutil.copy2(args.review, args.output / "reviewed_manifest.csv")
    snapshot(args)
    print(f"Built {args.output}: {dict(counts)}; excluded {len(rows) - len(selected)} unreviewed/skipped crops")


def evaluate(args):
    """Compare fixed operating points using the existing one-to-one detection matcher."""
    model, dataset = inputs(args)
    classifier = None
    if args.classifier:
        if args.classifier_type == "resnet18":
            from PIL import Image, ImageOps
            from torchvision import models, transforms

            checkpoint = torch_load(args.classifier, map_location="cpu")
            if checkpoint["classes"] != ["non_mildew", "powdery_mildew"]:
                raise ValueError("ResNet18 checkpoint classes must be [non_mildew, powdery_mildew]")
            classifier = models.resnet18(weights=None)
            classifier.fc = torch.nn.Linear(classifier.fc.in_features, 2)
            classifier.load_state_dict(checkpoint["model"], strict=True)
            classifier.to(select_device(args.device, verbose=False)).eval()
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            classifier = YOLO(str(args.classifier))
            if classifier.task != "classify" or set(classifier.names.values()) != {
                "mildew",
                "non_mildew",
            }:
                raise ValueError("Classifier must have exactly two classes named mildew and non_mildew")
            positive_id = next(key for key, value in classifier.names.items() if value == "mildew")
    snapshot(args, model)
    matcher = DetectionValidator(save_dir=args.output / "matching", args={"plots": False})
    matcher.iouv = torch.tensor([0.5])
    variants = [(t, None) for t in dict.fromkeys([0.25, *args.thresholds])]
    if classifier or args.classifier_type == "joint":
        variants += [(t, tau) for t in args.thresholds for tau in args.cls_thresholds]
    totals = {variant: np.zeros((len(NAMES), 3), dtype=np.int64) for variant in variants}
    sent = Counter()
    elapsed = Counter()
    capped = 0
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as cache:
        for index, label in enumerate(dataset.labels):
            start = time.perf_counter()
            result = predict(model, label["im_file"], args, min(min(args.thresholds), 0.25))
            boxes = result.boxes.data.cpu()
            elapsed["detector_seconds"] += time.perf_counter() - start
            capped += len(boxes) == 300
            gt = ground_truth(label, result.orig_img)
            white = boxes[:, 5] == MILDEW
            scores = torch.full((len(boxes),), float("nan"))
            if args.classifier_type == "joint":
                scores = result.mildew_scores.cpu()
            elif classifier and white.any():
                start = time.perf_counter()
                white_indices = torch.where(white)[0]
                for chunk in white_indices.split(args.cls_batch):
                    if args.classifier_type == "resnet18":
                        crops = [
                            Image.fromarray(crop_box(result.orig_img, boxes[i, :4], square=False)[..., ::-1])
                            for i in chunk
                        ]
                        batch = torch.stack(
                            [
                                transform(
                                    ImageOps.pad(
                                        crop,
                                        (224, 224),
                                        method=Image.Resampling.BILINEAR,
                                        color=(124, 116, 104),
                                    )
                                )
                                for crop in crops
                            ]
                        ).to(next(classifier.parameters()).device)
                        with torch.inference_mode():
                            scores[chunk] = classifier(batch).softmax(dim=1)[:, 1].cpu()
                    else:
                        crops = [crop_box(result.orig_img, boxes[i, :4]) for i in chunk]
                        classified = classifier.predict(
                            source=crops,
                            imgsz=224,
                            device=args.device,
                            verbose=False,
                            save=False,
                        )
                        scores[chunk] = torch.tensor([float(item.probs.data[positive_id].cpu()) for item in classified])
                elapsed["crop_and_classifier_seconds"] += time.perf_counter() - start
            correct = {}
            for variant in variants:
                threshold, tau = variant
                keep = torch.where(white, boxes[:, 4] >= threshold, boxes[:, 4] >= 0.25)
                sent[variant] += int((keep & white).sum())
                if tau is not None:
                    keep &= ~white | (scores >= tau)
                pred = boxes[keep]
                tp = matcher.match_predictions(pred[:, 5], gt[:, 4], box_iou(gt[:, :4], pred[:, :4]))[:, 0]
                for class_id in NAMES:
                    predicted = pred[:, 5] == class_id
                    n_tp = int((tp & predicted).sum())
                    totals[variant][class_id] += [
                        n_tp,
                        int(predicted.sum()) - n_tp,
                        int((gt[:, 4] == class_id).sum()) - n_tp,
                    ]
                correct[variant] = pred[pred[:, 5] != MILDEW]
            unchanged = correct[variants[0]]
            if any(not torch.equal(value, unchanged) for value in correct.values()):
                raise AssertionError("Non-Mildew detections changed between experiment groups")
            cache.write(
                json.dumps(
                    {
                        "image": label["im_file"],
                        "gt_xyxy_class": gt.tolist(),
                        "boxes_xyxy_score_class": boxes.tolist(),
                        "classifier_mildew_scores": [None if torch.isnan(value) else float(value) for value in scores],
                    }
                )
                + "\n"
            )
            if (index + 1) % 50 == 0:
                print(f"Evaluated {index + 1}/{len(dataset)} images", flush=True)
    rows = []
    for (threshold, tau), counts in totals.items():
        for class_id, (tp, fp, fn) in enumerate(counts.tolist()):
            precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            rows.append(
                {
                    "split": args.split,
                    "det_threshold": threshold,
                    "binary_threshold": "off" if tau is None else tau,
                    "class": NAMES[class_id],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                    "fp_per_image": fp / len(dataset),
                    "candidate_crops_per_image": sent[(threshold, tau)] / len(dataset),
                }
            )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "timing.json").write_text(
        json.dumps(
            {
                **elapsed,
                "images": len(dataset),
                "max_det_capped_images": capped,
                "note": "Shared cache pass including cold starts; not per-variant deployment latency.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in rows:
        if row["class"] == "Mildew":
            print(
                f"t={row['det_threshold']:.2f}, binary={row['binary_threshold']}: "
                f"P={row['precision']:.4f}, R={row['recall']:.4f}, F1={row['f1']:.4f}, FP={row['fp']}, FN={row['fn']}"
            )
    print(f"Saved {args.output / 'summary.csv'}; max_det capped {capped} images")


def main():
    """Run candidate preparation, reviewed dataset building, or cascade evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "build", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument(
            "--output", type=Path, required=True, help="New directory; existing outputs are never overwritten"
        )
        if command == "build":
            sub.add_argument("--review", type=Path, required=True)
            continue
        sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--device", default="0")
        sub.add_argument("--imgsz", type=int, default=640)
        if command == "prepare":
            sub.set_defaults(split="train")
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--groups", type=Path, help="Optional complete CSV with image,group columns")
        else:
            sub.add_argument("--split", choices=("val", "test"), default="val")
            sub.add_argument("--classifier", type=Path)
            sub.add_argument("--classifier-type", choices=("yolo", "resnet18", "joint"), default="yolo")
            sub.add_argument("--thresholds", nargs="+", type=float, default=[0.15, 0.20, 0.30, 0.35])
            sub.add_argument("--cls-thresholds", nargs="+", type=float, default=[0.50])
            sub.add_argument("--cls-batch", type=int, default=32)
    args = parser.parse_args()
    if args.command == "evaluate":
        if args.classifier_type == "joint" and args.classifier:
            parser.error("Joint classification is embedded in --model; omit --classifier")
        if any(not 0 < value <= 1 for value in args.thresholds + args.cls_thresholds) or args.cls_batch < 1:
            parser.error("Thresholds must be in (0, 1] and cls-batch must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    {"prepare": prepare, "build": build, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
