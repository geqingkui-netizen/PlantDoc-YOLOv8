"""Train binary or four-class region reviewers on a frozen four-class YOLOv8 detector."""

import argparse
import csv
import hashlib
import json
import math
from copy import copy, deepcopy
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML, nms, ops
from ultralytics.utils.files import increment_path
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.torch_utils import unwrap_model

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: "BrownRust", 1: "Healthy", 2: "Loose-Smut", 3: "Mildew"}
MILDEW = 3


def detector_digest(model):
    """Hash every detector parameter and persistent buffer, excluding the reviewer."""
    digest = hashlib.sha256()
    for name, value in model.model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenReviewModel(DetectionModel):
    """Share detached P3 features with a three-convolution binary or four-class reviewer."""

    def __init__(self, cfg, ch=3, nc=4, verbose=True):
        """Build the same classifier body for both tasks, changing only the final output size."""
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        if type(head) is not Detect or nc != 4 or head.end2end:
            raise ValueError("Use the original four-class YOLOv8 detector")
        self.variant = self.yaml["review_variant"]
        channels = head.cv2[0][0].conv.in_channels
        self.region_classifier = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1 if self.variant == "binary" else 4),
        )
        self.roi_stride = float(self.stride[0])
        self.criterion = None
        self.sample_generator = torch.Generator().manual_seed(self.yaml["review_seed"])
        self.sample_counts = {key: [0] * 4 for key in ("gt", "proposal", "selected")}
        self.sample_counts.update(unmatched_skipped=0, ambiguous_skipped=0)

    def regions(self, boxes, feature):
        """Expand regions by ten percent per side and clip to network input boundaries."""
        center, half = (boxes[:, :2] + boxes[:, 2:]) / 2, (boxes[:, 2:] - boxes[:, :2]) * 0.6
        expanded = torch.cat((center - half, center + half), 1).clamp(min=0)
        height, width = feature.shape[-2:]
        return torch.minimum(expanded, expanded.new_tensor([width, height, width, height]) * self.roi_stride)

    def region_logits(self, feature, regions):
        """Read P3 regions without allowing gradients into the detector."""
        from torchvision.ops import roi_align

        pooled = roi_align(
            feature.detach().float(),
            [region.float() for region in regions],
            output_size=7,
            spatial_scale=1 / self.roi_stride,
            sampling_ratio=2,
            aligned=True,
        )
        return self.region_classifier(pooled.to(self.region_classifier[0].weight.dtype))

    def review_regions(self, feature, boxes, classes, threshold):
        """Only remove detections; threshold zero bypasses review completely."""
        keep = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
        selected = classes == MILDEW if self.variant == "binary" else keep.clone()
        if threshold and selected.any():
            logits = self.region_logits(feature, [self.regions(boxes[selected], feature)]).float()
            if self.variant == "binary":
                keep[selected] = logits[:, 0].sigmoid() >= threshold
            else:
                probabilities = logits.softmax(1)
                scores = probabilities.gather(1, classes[selected].long()[:, None]).squeeze(1)
                keep[selected] = (probabilities.argmax(1) == classes[selected]) & (scores >= threshold)
        return keep

    def classification_loss(self, preds, batch):
        """Use the same GT and matched proposal regions for both classification tasks."""
        decoded, raw = preds
        feature = raw["feats"][0]
        candidates = nms.non_max_suppression(
            decoded.detach(),
            conf_thres=0.1,
            iou_thres=0.7,
            multi_label=True,
            max_det=300,
        )
        regions, labels = [], []
        for index, predictions in enumerate(candidates):
            selected = batch["batch_idx"].view(-1) == index
            classes = batch["cls"].view(-1)[selected].long()
            gt = ops.xywh2xyxy(batch["bboxes"][selected]).float()
            height, width = batch["img"].shape[-2:]
            gt *= gt.new_tensor([width, height, width, height])
            gt_regions = self.regions(gt, feature)
            boxes = predictions[:, :4].float()
            expanded = self.regions(boxes, feature)
            proposal_classes = classes.new_full((len(boxes),), -1)
            valid_gt = torch.ones(len(gt), dtype=torch.bool, device=feature.device)
            if len(gt):
                # An expanded region touching another labeled class has ambiguous supervision.
                valid_gt = ~((box_iou(gt_regions, gt) > 0) & (classes[:, None] != classes[None])).any(1)
            if len(gt) and len(boxes):
                maximum, nearest = box_iou(boxes, gt).max(1)
                matched = classes[nearest]
                conflicts = ((box_iou(expanded, gt) > 0) & (matched[:, None] != classes[None])).any(1)
                valid = (maximum >= 0.5) & ~conflicts
                proposal_classes[valid] = matched[valid]
                if self.training:
                    self.sample_counts["ambiguous_skipped"] += int(((maximum >= 0.5) & conflicts).sum())
                    self.sample_counts["unmatched_skipped"] += int((maximum < 0.5).sum())
            elif self.training:
                self.sample_counts["unmatched_skipped"] += len(boxes)
            if self.training:
                self.sample_counts["ambiguous_skipped"] += int((~valid_gt).sum())
            chosen_regions = []
            for category in NAMES:
                gt_pool = gt_regions[valid_gt & (classes == category)]
                proposal_pool = expanded[proposal_classes == category]
                pool = torch.cat((gt_pool, proposal_pool))
                order = (
                    torch.randperm(len(pool), generator=self.sample_generator).to(feature.device)
                    if self.training
                    else torch.arange(len(pool), device=feature.device)
                )
                chosen = pool[order[:32]]
                chosen_regions.append(chosen)
                labels.append(classes.new_full((len(chosen),), category))
                if self.training:
                    for key, count in (
                        ("gt", len(gt_pool)),
                        ("proposal", len(proposal_pool)),
                        ("selected", len(chosen)),
                    ):
                        self.sample_counts[key][category] += count
            regions.append(torch.cat(chosen_regions))
        targets = torch.cat(labels)
        if not len(targets):
            return sum(parameter.sum() * 0 for parameter in self.region_classifier.parameters())
        logits = self.region_logits(feature, regions).float()
        if self.variant == "binary":
            targets = (targets == MILDEW).long()
            losses = F.binary_cross_entropy_with_logits(logits[:, 0], targets.float(), reduction="none")
            categories = (0, 1)
        else:
            losses = F.cross_entropy(logits, targets, reduction="none")
            categories = NAMES
        return torch.stack([losses[targets == t].mean() for t in categories if (targets == t).any()]).mean()

    def loss(self, batch, preds=None):
        """Optimize only the region classifier, retaining evaluation-mode detector outputs."""
        if preds is None:
            with torch.no_grad():
                preds = self.predict(batch["img"])
        loss = self.classification_loss(preds, batch)
        return loss.view(1) * batch["img"].shape[0], {"review_loss": loss.detach()}


class FrozenValidator(DetectionValidator):
    """Use ordinary detection metrics and optionally filter regions before metric matching."""

    def __init__(self, *args, review_threshold=0, **kwargs):
        """Use zero during training so detection metrics remain comparable to the baseline."""
        self.review_threshold = review_threshold
        super().__init__(*args, **kwargs)

    def init_metrics(self, model):
        """Find the reviewer behind the native PyTorch backend."""
        super().init_metrics(model)
        self.frozen_model = unwrap_model(model.model if getattr(model, "format", None) == "pt" else model)
        if not isinstance(self.frozen_model, FrozenReviewModel):
            raise ValueError("Use a checkpoint trained with FrozenTrainer")

    def postprocess(self, preds):
        """Filter boxes in network coordinates using the model's shared review rule."""
        results = super().postprocess(preds)
        if self.review_threshold:
            for index, result in enumerate(results):
                keep = self.frozen_model.review_regions(
                    preds[1]["feats"][0][index : index + 1],
                    result["bboxes"],
                    result["cls"],
                    self.review_threshold,
                )
                results[index] = {key: value[keep] for key, value in result.items()}
        return results

    def __call__(self, trainer=None, model=None, **kwargs):
        """Choose best.pt by validation classification loss instead of unchanged detection AP."""
        stats = super().__call__(trainer=trainer, model=model, **kwargs)
        if self.training:
            stats["fitness"] = 1 / (1 + stats["val/review_loss"])
        return stats


class FrozenPredictor(DetectionPredictor):
    """Apply the same review rule to image and video prediction."""

    def __init__(self, *args, review_threshold=0.25, **kwargs):
        """Accept the independently selected classification threshold."""
        self.review_threshold = review_threshold
        super().__init__(*args, **kwargs)

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Review network-space boxes before the standard result constructor rescales them."""
        model = self.model.model
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
        )
        reviewed = []
        for index, output in enumerate(outputs):
            keep = model.review_regions(
                preds[1]["feats"][0][index : index + 1],
                output[:, :4],
                output[:, 5],
                self.review_threshold,
            )
            reviewed.append(output[keep])
        return self.construct_results(reviewed, img, orig_imgs)


class FrozenTrainer(DetectionTrainer):
    """Reuse the training loop while optimizing only the region classifier."""

    def __init__(self, *args, variant="binary", **kwargs):
        """Select the review task without adding arguments to the package-wide configuration."""
        self.variant = variant
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load every detector tensor from the baseline and freeze all detector layers."""
        if self.data["names"] != NAMES or weights is None:
            raise ValueError("Use the original four-class dataset and a pretrained detector checkpoint")
        cfg = deepcopy(cfg)
        cfg.update(review_variant=self.variant, review_seed=self.args.seed)
        model = FrozenReviewModel(cfg, ch=self.data["channels"], nc=4, verbose=verbose)
        model.load(weights)
        model.names = self.data["names"]
        model.model.requires_grad_(False)
        self.args.freeze = len(model.model)
        self.detector_hash = detector_digest(weights)
        if detector_digest(model) != self.detector_hash:
            raise RuntimeError("The baseline detector did not transfer exactly")
        self.loss_names = ("review_loss",)
        return model

    def _model_train(self):
        """Keep the entire frozen detector, including BatchNorm buffers, in evaluation mode."""
        super()._model_train()
        unwrap_model(self.model).model.eval()

    def build_optimizer(self, model, **kwargs):
        """Pass only classifier parameters to the existing optimizer builder."""
        return super().build_optimizer(unwrap_model(model).region_classifier, **kwargs)

    def get_validator(self):
        """Disable filtering during training validation; select checkpoints using review loss."""
        return FrozenValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def validate(self):
        """Verify frozen parameters and buffers in both the live and EMA detectors."""
        for model in (unwrap_model(self.model), self.ema.ema):
            if detector_digest(model) != self.detector_hash:
                raise RuntimeError("Frozen detector weights or buffers changed")
        (self.save_dir / "region_sampling.json").write_text(
            json.dumps(unwrap_model(self.model).sample_counts, indent=2),
            encoding="utf-8",
        )
        return super().validate()


def check_detector(checkpoint, baseline):
    """Require exact equality with the original detector, including all persistent buffers."""
    candidate = YOLO(str(checkpoint))
    reference = YOLO(str(baseline))
    if not isinstance(candidate.model, FrozenReviewModel) or reference.names != NAMES:
        raise ValueError("Expected a frozen-review checkpoint and the original four-class baseline")
    if detector_digest(candidate.model) != detector_digest(reference.model):
        raise RuntimeError("Checkpoint detector differs from the supplied baseline")
    return candidate


def metric_rows(metrics, group, threshold, split):
    """Report per-class and macro metrics at the framework's confidence-curve working point."""
    rows = []
    for index, category in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.class_result(index)
        rows.append(
            dict(
                group=group,
                review_threshold=threshold,
                split=split,
                category=NAMES[int(category)],
                precision=float(p),
                recall=float(r),
                F1=float(metrics.box.f1[index]),
                AP50=float(ap50),
                AP50_95=float(ap),
            )
        )
    p, r, ap50, ap = metrics.mean_results()
    rows.append(
        dict(
            group=group,
            review_threshold=threshold,
            split=split,
            category="all",
            precision=float(p),
            recall=float(r),
            F1=float(metrics.box.f1.mean()) if len(metrics.box.f1) else 0.0,
            AP50=float(ap50),
            AP50_95=float(ap),
        )
    )
    return rows


def main():
    """Train, check, evaluate, predict, or compare independently saved review tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "val", "predict", "compare", "check"))
    parser.add_argument(
        "--variant", choices=("binary", "fourclass"), help="Required for train; inferred from other checkpoints"
    )
    parser.add_argument("--model", nargs="+", default=[], help="One checkpoint; compare accepts both variants together")
    parser.add_argument("--baseline", default=str(ROOT / "weights/baseline400.pt"))
    parser.add_argument("--data", default="E:/Dataset/dataset_four_classes/data.yaml")
    parser.add_argument("--config", default=str(ROOT / "frozen_review.yaml"))
    for name, kind in (
        ("batch", int),
        ("imgsz", int),
        ("workers", int),
        ("epochs", int),
        ("seed", int),
        ("patience", int),
        ("lr0", float),
    ):
        parser.add_argument(f"--{name}", type=kind)
    parser.add_argument("--device")
    parser.add_argument("--review-threshold", type=float, default=0.25)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0, 0.1, 0.15, 0.2, 0.25, 0.5])
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--source", help="Image, directory, video, or other YOLO prediction source")
    parser.add_argument("--name")
    args = parser.parse_args()
    settings = YAML.load(args.config)
    for key in ("batch", "imgsz", "workers", "epochs", "seed", "patience", "lr0", "device"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    settings["nbs"] = settings["batch"]
    if str(settings["device"]) not in {"cpu", "0"} or settings["batch"] < 1:
        parser.error("This experiment supports CPU or a single GPU (--device 0) and a positive batch size")
    if not 0 <= args.review_threshold <= 1 or any(not 0 <= t <= 1 for t in args.thresholds):
        parser.error("Review thresholds must be between zero and one")
    if args.command == "train" and (args.variant is None or args.model):
        parser.error("Use train --variant binary|fourclass; training always starts from --baseline")
    if args.command != "train" and (not args.model or (args.command != "compare" and len(args.model) != 1)):
        parser.error("Supply --model CHECKPOINT; only compare accepts multiple checkpoints")
    if args.command == "predict" and not args.source:
        parser.error("predict requires --source")
    if args.command == "compare" and args.split != "val":
        parser.error("Select thresholds on val; use val --split test for the final chosen threshold")
    common = dict(
        data=args.data,
        device=settings["device"],
        batch=settings["batch"],
        imgsz=settings["imgsz"],
        workers=settings["workers"],
        project=str(ROOT / "runs/frozen_review"),
    )
    evaluation = dict(conf=0.001, iou=0.7, max_det=300, quantize=None, channels_last=False)
    if args.command == "train":
        detector = YOLO(args.baseline)
        if type(detector.model) is not DetectionModel or detector.names != NAMES:
            parser.error("--baseline must be the original four-class detector, not a joint or review checkpoint")
        settings["freeze"] = len(detector.model.model)
        detector.train(
            trainer=partial(FrozenTrainer, variant=args.variant),
            **settings,
            data=args.data,
            project=common["project"],
            name=args.name or f"{args.variant}_seed{settings['seed']}",
        )
        checkpoint = detector.trainer.best
        check_detector(checkpoint, args.baseline)
        manifest = dict(
            variant=args.variant,
            baseline=str(Path(args.baseline).resolve()),
            baseline_file_sha256=hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
            detector_sha256=detector_digest(YOLO(args.baseline).model),
            data=str(Path(args.data).resolve()),
            data_yaml_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            torch=torch.__version__,
            settings=settings,
        )
        (detector.trainer.save_dir / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"PASS: detector unchanged. Best {args.variant} classifier: {checkpoint}")
        return
    models = [check_detector(path, args.baseline) for path in args.model]
    if args.variant is not None and any(model.model.variant != args.variant for model in models):
        parser.error("--variant does not match the checkpoint's saved review task")
    if args.command == "check":
        print("PASS: all detector parameters and persistent buffers exactly match the baseline")
    elif args.command == "predict":
        model = models[0]
        prediction_settings = {**evaluation, "conf": 0.25}
        model.predict(
            source=args.source,
            predictor=partial(FrozenPredictor, review_threshold=args.review_threshold),
            save=True,
            name=args.name or f"{model.model.variant}_predict",
            **common,
            **prediction_settings,
        )
    elif args.command == "val":
        model = models[0]
        metrics = model.val(
            validator=partial(FrozenValidator, review_threshold=args.review_threshold),
            split=args.split,
            plots=True,
            name=args.name or f"{model.model.variant}_t{args.review_threshold:g}_{args.split}",
            **common,
            **evaluation,
        )
        rows = metric_rows(metrics, model.model.variant, args.review_threshold, args.split)
        with (metrics.save_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        variants = [model.model.variant for model in models]
        if len(set(variants)) != len(variants):
            parser.error("Compare at most one checkpoint per variant; use separate runs to compare seeds")
        folder = increment_path(ROOT / "outputs" / (args.name or "frozen_comparison"), mkdir=True)
        baseline = YOLO(args.baseline)
        metrics = baseline.val(plots=False, name=f"{folder.name}_baseline", **common, **evaluation)
        rows = metric_rows(metrics, "baseline400", None, "val")
        reference = {row["category"]: row for row in rows}
        for path, variant in zip(args.model, variants):
            for threshold in args.thresholds:
                # Each backend may fuse its model; reload so every threshold starts from the same checkpoint.
                model = YOLO(path)
                metrics = model.val(
                    validator=partial(FrozenValidator, review_threshold=threshold),
                    plots=False,
                    name=f"{folder.name}_{variant}_t{threshold:g}",
                    **common,
                    **evaluation,
                )
                current = metric_rows(metrics, variant, threshold, "val")
                if threshold == 0:
                    for row in current:
                        if any(
                            not math.isclose(row[key], reference[row["category"]][key], abs_tol=1e-9)
                            for key in ("precision", "recall", "F1", "AP50", "AP50_95")
                        ):
                            raise RuntimeError(f"Filter-off metrics differ from baseline: {variant}")
                rows.extend(current)
        with (folder / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (folder / "settings.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"Saved comparison: {folder / 'metrics.csv'}")
