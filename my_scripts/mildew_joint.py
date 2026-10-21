"""Three-convolution Mildew region classification jointly trained with YOLOv8."""

from __future__ import annotations

import argparse
import json
from copy import copy, deepcopy
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel, yaml_model_load
from ultralytics.utils import nms, ops
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.torch_utils import unwrap_model

NAMES = {0: "BrownRust", 1: "Healthy", 2: "Loose-Smut", 3: "Mildew"}
MILDEW = 3


class ThreeConvDetectionModel(DetectionModel):
    """Share YOLOv8 P3 features with a three-convolution binary region classifier."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=4, verbose=True):
        """Build the detector and three 3x3, 64-channel convolutions for region classification."""
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        head = self.model[-1]
        if type(head) is not Detect or getattr(head, "one2one_cv2", None) is not None or nc != 4:
            raise ValueError("This experiment requires a four-class YOLOv8 Detect model")
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
            nn.Linear(64, 1),
        )
        self.binary_weight = float(self.yaml.get("mildew_binary_weight", 0.5))
        self.roi_stride = float(self.stride[0])
        self.binary_samples = dict.fromkeys(
            (
                "gt_positive",
                "gt_negative",
                "proposal_positive",
                "proposal_background",
                "proposal_other",
                "selected_positive",
                "selected_negative",
            ),
            0,
        )

    def regions(self, boxes, feature):
        """Expand boxes by ten percent on each side and clip to the network input."""
        center, half = (boxes[:, :2] + boxes[:, 2:]) / 2, (boxes[:, 2:] - boxes[:, :2]) * 0.6
        expanded = torch.cat((center - half, center + half), 1).clamp(min=0)
        height, width = feature.shape[-2:]
        limit = expanded.new_tensor([width, height, width, height]) * self.roi_stride
        return torch.minimum(expanded, limit)

    def region_logits(self, feature, regions):
        """Pool regions from connected P3 features so binary gradients reach the detector."""
        from torchvision.ops import roi_align

        pooled = roi_align(
            feature.float(),
            [region.float() for region in regions],
            output_size=7,
            spatial_scale=1 / self.roi_stride,
            sampling_ratio=2,
            aligned=True,
        )
        return self.region_classifier(pooled.to(self.region_classifier[0].weight.dtype)).flatten()

    def refinement_scores(self, feature, boxes, classes):
        """Score only Mildew detections; leave other detections untouched."""
        scores = boxes.new_ones(len(boxes))
        white = classes == MILDEW
        if self.binary_weight and white.any():
            regions = self.regions(boxes[white], feature)
            scores[white] = self.region_logits(feature, [regions]).sigmoid().to(scores.dtype)
        return scores

    def binary_loss(self, raw, batch):
        """Generate balanced region labels from GT and conservative decoded proposals."""
        feature = raw["feats"][0]
        zero = feature.sum() * 0 + sum(parameter.sum() * 0 for parameter in self.region_classifier.parameters())
        if not self.binary_weight:
            return zero
        with torch.no_grad():
            decoded = ops.xywh2xyxy(self.model[-1]._get_decode_boxes(raw).transpose(1, 2)).detach().float()
            scores = raw["scores"][:, MILDEW].detach()
        regions, labels = [], []
        for image_index in range(feature.shape[0]):
            selected = batch["batch_idx"].view(-1) == image_index
            classes = batch["cls"].view(-1)[selected]
            gt = ops.xywh2xyxy(batch["bboxes"][selected]).float()
            height, width = batch["img"].shape[-2:]
            gt = gt * gt.new_tensor([width, height, width, height])
            gt_regions = self.regions(gt, feature)
            white_gt = gt[classes == MILDEW]
            other_gt = gt[classes != MILDEW]
            gt_negative = classes != MILDEW
            if len(white_gt):
                gt_negative &= box_iou(gt_regions, white_gt).amax(1) == 0
            indices = scores[image_index].topk(min(64, scores.shape[1])).indices
            proposals = self.regions(decoded[image_index, indices], feature)
            proposals = proposals[(proposals[:, 2:] - proposals[:, :2]).amin(1) >= 2]
            overlap = box_iou(proposals, gt)
            white_overlap = overlap[:, classes == MILDEW]
            other_overlap = overlap[:, classes != MILDEW]
            positive = (
                white_overlap.amax(1) >= 0.5
                if len(white_gt)
                else torch.zeros(len(proposals), dtype=torch.bool, device=feature.device)
            )
            no_white = white_overlap.amax(1) == 0 if len(white_gt) else torch.ones_like(positive)
            background = overlap.amax(1) == 0 if len(gt) else torch.ones_like(positive)
            other = (other_overlap.amax(1) >= 0.5) & no_white if len(other_gt) else torch.zeros_like(positive)
            pools = (
                torch.cat((gt_regions[classes == MILDEW], proposals[positive])),
                torch.cat((gt_regions[gt_negative], proposals[background | other])),
            )
            counts = {
                "gt_positive": int((classes == MILDEW).sum()),
                "gt_negative": int(gt_negative.sum()),
                "proposal_positive": int(positive.sum()),
                "proposal_background": int(background.sum()),
                "proposal_other": int(other.sum()),
            }
            image_regions = []
            for target, pool in zip((1, 0), pools):
                indices = (
                    torch.randperm(len(pool), device=pool.device)
                    if self.training
                    else torch.arange(len(pool), device=pool.device)
                )
                chosen = pool[indices[:32]]
                image_regions.append(chosen)
                labels.append(feature.new_full((len(chosen),), target))
                counts["selected_positive" if target else "selected_negative"] = len(chosen)
            regions.append(torch.cat(image_regions))
            if self.training:
                for key, value in counts.items():
                    self.binary_samples[key] += value
        targets = torch.cat(labels)
        if not len(targets):
            return zero
        losses = F.binary_cross_entropy_with_logits(
            self.region_logits(feature, regions), targets.float(), reduction="none"
        )
        groups = [losses[targets == target].mean() for target in (0, 1) if (targets == target).any()]
        return torch.stack(groups).mean()

    def loss(self, batch, preds=None):
        """Add binary region loss to the existing detection loss without replacing its assignment."""
        preds = self.predict(batch["img"]) if preds is None else preds
        loss, items = super().loss(batch, preds)
        raw = preds[1] if isinstance(preds, tuple) else preds
        binary = self.binary_loss(raw, batch) * self.binary_weight
        return torch.cat((loss, binary.view(1) * batch["img"].shape[0])), {**items, "mildew_loss": binary.detach()}


class JointValidator(DetectionValidator):
    """Evaluate the trained region filter after standard class-aware NMS."""

    def __init__(self, *args, binary_threshold=0.5, **kwargs):
        """Keep the binary operating threshold separate from detection confidence."""
        self.binary_threshold = binary_threshold
        super().__init__(*args, **kwargs)

    def init_metrics(self, model):
        """Use the live EMA during training and the native checkpoint model during standalone validation."""
        super().init_metrics(model)
        self.joint_model = unwrap_model(model.model if getattr(model, "format", None) == "pt" else model)
        if not isinstance(self.joint_model, ThreeConvDetectionModel):
            raise ValueError("Expected a checkpoint trained with JointTrainer")

    def postprocess(self, preds):
        """Filter Mildew detections without changing boxes of any other class."""
        results = super().postprocess(preds)
        if self.binary_threshold:
            for index, result in enumerate(results):
                scores = self.joint_model.refinement_scores(
                    preds[1]["feats"][0][index : index + 1], result["bboxes"], result["cls"]
                )
                keep = (result["cls"] != MILDEW) | (scores >= self.binary_threshold)
                results[index] = {key: value[keep] for key, value in result.items()}
        return results


class JointPredictor(DetectionPredictor):
    """Expose binary scores and optionally filter boxes for ordinary image prediction."""

    def __init__(self, *args, binary_threshold=0.5, **kwargs):
        """Set zero to collect unfiltered boxes and binary scores for threshold comparisons."""
        self.binary_threshold = binary_threshold
        super().__init__(*args, **kwargs)

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Run the existing NMS and classify its Mildew regions in network input coordinates."""
        model = self.model.model
        if not isinstance(model, ThreeConvDetectionModel):
            raise ValueError("Expected a checkpoint trained with JointTrainer")
        outputs = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou, self.args.classes, self.args.agnostic_nms, max_det=self.args.max_det
        )
        binary_scores = []
        for index, output in enumerate(outputs):
            scores = model.refinement_scores(preds[1]["feats"][0][index : index + 1], output[:, :4], output[:, 5])
            keep = (output[:, 5] != MILDEW) | (scores >= self.binary_threshold)
            outputs[index], scores = output[keep], scores[keep]
            binary_scores.append(scores)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        results = self.construct_results(outputs, img, orig_imgs)
        for result, scores in zip(results, binary_scores):
            result.mildew_scores = scores
        return results


class JointTrainer(DetectionTrainer):
    """Reuse Ultralytics training, optimization, EMA and checkpointing for the joint model."""

    def __init__(self, *args, binary_weight=0.5, **kwargs):
        """Set the auxiliary loss weight without adding a global YOLO configuration argument."""
        self.binary_weight = binary_weight
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load detector weights and retain the region classifier when resuming a joint checkpoint."""
        if self.data["names"] != NAMES:
            raise ValueError(f"Expected four detection classes {NAMES}")
        cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg or "yolov8n.yaml")
        if not self.resume:
            cfg["mildew_binary_weight"] = self.binary_weight
        model = ThreeConvDetectionModel(cfg, nc=4, ch=self.data["channels"], verbose=verbose)
        model.names = self.data["names"]
        if weights is not None:
            model.load(weights)
        return model

    def get_validator(self):
        """Use the same learned region filter during validation and final checkpoint evaluation."""
        return JointValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def save_model(self):
        """Record automatic training sample counts beside the standard joint checkpoints."""
        saved = super().save_model()
        if saved:
            model = unwrap_model(self.model)
            (self.save_dir / "binary_sampling.json").write_text(
                json.dumps(
                    {
                        "binary_weight": model.binary_weight,
                        "samples": model.binary_samples,
                        "note": "Counts include repeated augmented regions; background negatives assume complete detection labels.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return saved


def main():
    """Train, validate or predict with the standalone joint experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "val", "predict"):
        sub = commands.add_parser(command)
        sub.add_argument(
            "--model", default="weights/yolov8n.pt" if command == "train" else None, required=command != "train"
        )
        sub.add_argument("--device", default="0")
        sub.add_argument("--imgsz", type=int, default=640)
        sub.add_argument("--batch", type=int, default=2)
        sub.add_argument("--workers", type=int, default=0)
        sub.add_argument("--project", default=str(Path(__file__).resolve().parents[1] / "runs/wheat_joint_3conv"))
        sub.add_argument("--name", default=f"joint_3conv_{command}")
        if command != "predict":
            sub.add_argument("--data", required=True)
        if command == "train":
            sub.add_argument("--epochs", type=int, default=100)
            sub.add_argument("--binary-weight", type=float, default=0.5)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--resume", action="store_true")
        else:
            sub.add_argument("--binary-threshold", type=float, default=0.5)
            sub.add_argument("--conf", type=float, default=0.001 if command == "val" else 0.25)
            if command == "val":
                sub.add_argument("--split", choices=("val", "test"), default="val")
            else:
                sub.add_argument("--source", required=True)
    args = parser.parse_args()
    if "," in args.device or args.batch < 1:
        parser.error("Use one GPU or cpu and a positive batch size for this experiment")
    if args.command == "train" and (args.binary_weight < 0 or not torch.isfinite(torch.tensor(args.binary_weight))):
        parser.error("binary-weight must be finite and nonnegative")
    if args.command != "train" and not (0 <= args.binary_threshold <= 1 and 0 < args.conf <= 1):
        parser.error("binary-threshold must be in [0, 1], conf in (0, 1]")
    model = YOLO(args.model)
    common = dict(
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )
    if args.command == "train":
        model.train(
            trainer=partial(JointTrainer, binary_weight=args.binary_weight),
            data=args.data,
            epochs=args.epochs,
            seed=args.seed,
            deterministic=True,
            resume=args.resume,
            **common,
        )
    elif args.command == "val":
        model.val(
            validator=partial(JointValidator, binary_threshold=args.binary_threshold),
            data=args.data,
            split=args.split,
            conf=args.conf,
            **common,
        )
    else:
        model.predict(
            predictor=partial(JointPredictor, binary_threshold=args.binary_threshold),
            source=args.source,
            conf=args.conf,
            save=True,
            **common,
        )
