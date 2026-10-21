"""Joint YOLOv8 and five-class region review using existing detection annotations."""

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
BACKGROUND = 4


class JointDetectionModel(DetectionModel):
    """Share YOLOv8 P3 features with a three-convolution, five-class region reviewer."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=4, verbose=True):
        """Build the detector and a small region classifier without external classification data."""
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
            nn.Linear(64, 5),
        )
        self.region_weight = float(self.yaml.get("region_weight", 0.5))
        self.roi_stride = float(self.stride[0])
        self.region_samples = {key: [0] * 5 for key in ("gt", "proposal", "selected")}

    def regions(self, boxes, feature):
        """Expand boxes by ten percent on each side and clip to the network input."""
        center, half = (boxes[:, :2] + boxes[:, 2:]) / 2, (boxes[:, 2:] - boxes[:, :2]) * 0.6
        expanded = torch.cat((center - half, center + half), 1).clamp(min=0)
        height, width = feature.shape[-2:]
        limit = expanded.new_tensor([width, height, width, height]) * self.roi_stride
        return torch.minimum(expanded, limit)

    def region_logits(self, feature, regions):
        """Pool regions from connected P3 features so region gradients reach the detector."""
        from torchvision.ops import roi_align

        pooled = roi_align(
            feature.float(),
            [region.float() for region in regions],
            output_size=7,
            spatial_scale=1 / self.roi_stride,
            sampling_ratio=2,
            aligned=True,
        )
        return self.region_classifier(pooled.to(self.region_classifier[0].weight.dtype))

    def review_regions(self, feature, boxes, classes, threshold):
        """Approve the original class only; never relabel detections or emit background boxes."""
        scores = boxes.new_ones(len(boxes))
        keep = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)
        if self.region_weight and threshold and len(boxes):
            probabilities = self.region_logits(feature, [self.regions(boxes, feature)]).float().softmax(1)
            scores = probabilities.gather(1, classes.long()[:, None]).squeeze(1)
            keep = (probabilities.argmax(1) == classes.long()) & (scores >= threshold)
        return keep, scores

    def region_loss(self, raw, batch):
        """Assign GT and unambiguous proposals to four target classes or background automatically."""
        feature = raw["feats"][0]
        zero = feature.sum() * 0 + sum(parameter.sum() * 0 for parameter in self.region_classifier.parameters())
        if not self.region_weight:
            return zero
        with torch.no_grad():
            decoded = ops.xywh2xyxy(self.model[-1]._get_decode_boxes(raw).transpose(1, 2)).detach().float()
            scores = raw["scores"].detach().amax(1)
        regions, labels = [], []
        for image_index in range(feature.shape[0]):
            selected = batch["batch_idx"].view(-1) == image_index
            classes = batch["cls"].view(-1)[selected].long()
            gt = ops.xywh2xyxy(batch["bboxes"][selected]).float()
            height, width = batch["img"].shape[-2:]
            gt = gt * gt.new_tensor([width, height, width, height])
            gt_regions = self.regions(gt, feature)
            indices = scores[image_index].topk(min(64, scores.shape[1])).indices
            proposals = self.regions(decoded[image_index, indices], feature)
            proposals = proposals[(proposals[:, 2:] - proposals[:, :2]).amin(1) >= 2]
            proposal_classes = classes.new_full((len(proposals),), -1)
            if len(gt) and len(proposals):
                overlap = box_iou(proposals, gt)
                maximum, nearest = overlap.max(1)
                matched = classes[nearest]
                conflicting = ((overlap >= 0.2) & (classes[None] != matched[:, None])).any(1)
                foreground = (maximum >= 0.5) & ~conflicting
                proposal_classes[foreground] = matched[foreground]
                proposal_classes[maximum == 0] = BACKGROUND
            elif not len(gt):
                proposal_classes.fill_(BACKGROUND)
            image_regions = []
            for target in range(5):
                annotated = gt_regions[classes == target]
                proposed = proposals[proposal_classes == target]
                pool = torch.cat((annotated, proposed))
                indices = (
                    torch.randperm(len(pool), device=pool.device)
                    if self.training
                    else torch.arange(len(pool), device=pool.device)
                )
                chosen = pool[indices[:32]]
                image_regions.append(chosen)
                labels.append(classes.new_full((len(chosen),), target))
                if self.training:
                    for key, value in (("gt", len(annotated)), ("proposal", len(proposed)), ("selected", len(chosen))):
                        self.region_samples[key][target] += value
            regions.append(torch.cat(image_regions))
        targets = torch.cat(labels)
        if not len(targets):
            return zero
        losses = F.cross_entropy(self.region_logits(feature, regions).float(), targets, reduction="none")
        return torch.stack([losses[targets == target].mean() for target in targets.unique()]).mean()

    def loss(self, batch, preds=None):
        """Append class-balanced region loss to the existing detection loss without replacing its assignment."""
        preds = self.predict(batch["img"]) if preds is None else preds
        loss, items = super().loss(batch, preds)
        raw = preds[1] if isinstance(preds, tuple) else preds
        region = self.region_loss(raw, batch) * self.region_weight
        return torch.cat((loss, region.view(1) * batch["img"].shape[0])), {**items, "region_loss": region.detach()}


class JointValidator(DetectionValidator):
    """Evaluate the trained region filter after standard class-aware NMS."""

    def __init__(self, *args, review_threshold=None, **kwargs):
        """Use the checkpoint threshold unless overridden; zero disables review."""
        self.review_threshold = review_threshold
        super().__init__(*args, **kwargs)

    def init_metrics(self, model):
        """Use the live EMA during training and the native checkpoint model during standalone validation."""
        super().init_metrics(model)
        self.joint_model = unwrap_model(model.model if getattr(model, "format", None) == "pt" else model)
        if not isinstance(self.joint_model, JointDetectionModel) or self.names != NAMES:
            raise ValueError("Expected a checkpoint trained with JointTrainer")

    def postprocess(self, preds):
        """Drop rejected boxes without changing retained coordinates, scores or classes."""
        results = super().postprocess(preds)
        threshold = (
            self.joint_model.yaml.get("review_threshold", 0.5)
            if self.review_threshold is None
            else self.review_threshold
        )
        for index, result in enumerate(results):
            keep, _ = self.joint_model.review_regions(
                preds[1]["feats"][0][index : index + 1], result["bboxes"], result["cls"], threshold
            )
            results[index] = {key: value[keep] for key, value in result.items()}
        return results


class JointPredictor(DetectionPredictor):
    """Apply the same review rule in prediction and expose retained region scores."""

    def __init__(self, *args, review_threshold=None, **kwargs):
        """Use the checkpoint threshold unless overridden; zero bypasses review."""
        self.review_threshold = review_threshold
        super().__init__(*args, **kwargs)

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Review regions in network coordinates before standard result construction rescales boxes."""
        model = self.model.model
        if not isinstance(model, JointDetectionModel):
            raise ValueError("Expected a checkpoint trained with JointTrainer")
        threshold = model.yaml.get("review_threshold", 0.5) if self.review_threshold is None else self.review_threshold
        outputs = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou, self.args.classes, self.args.agnostic_nms, max_det=self.args.max_det
        )
        review_scores = []
        for index, output in enumerate(outputs):
            keep, scores = model.review_regions(
                preds[1]["feats"][0][index : index + 1], output[:, :4], output[:, 5], threshold
            )
            outputs[index] = output[keep]
            review_scores.append(scores[keep])
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        results = self.construct_results(outputs, img, orig_imgs)
        for result, scores in zip(results, review_scores):
            result.review_scores = scores
        return results


class JointTrainer(DetectionTrainer):
    """Reuse Ultralytics training, optimization, EMA and checkpointing for the joint model."""

    def __init__(self, *args, region_weight=0.5, review_threshold=0.5, **kwargs):
        """Keep experiment settings in model YAML without extending global YOLO arguments."""
        self.region_weight = region_weight
        self.review_threshold = review_threshold
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Load detector weights and retain the region classifier when resuming a joint checkpoint."""
        if self.data["names"] != NAMES:
            raise ValueError(f"Expected four detection classes {NAMES}")
        cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg or "yolov8n.yaml")
        if not self.resume:
            cfg.update(region_weight=self.region_weight, review_threshold=self.review_threshold)
        model = JointDetectionModel(cfg, nc=4, ch=self.data["channels"], verbose=verbose)
        model.names = self.data["names"]
        if weights is not None:
            model.load(weights)
        return model

    def get_validator(self):
        """Use the same learned region filter during validation and final checkpoint evaluation."""
        return JointValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def save_model(self):
        """Record five-class automatic training sample counts beside the standard joint checkpoints."""
        saved = super().save_model()
        if saved:
            model = unwrap_model(self.model)
            (self.save_dir / "region_sampling.json").write_text(
                json.dumps(
                    {
                        "region_weight": model.region_weight,
                        "review_threshold": model.yaml["review_threshold"],
                        "names": {**NAMES, BACKGROUND: "Background"},
                        "samples": model.region_samples,
                        "note": "Counts cover this run's augmented samples; background assumes complete box labels.",
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
            "--model",
            default=str(Path(__file__).resolve().parents[1] / "weights/yolov8n.pt"),
            required=command != "train",
        )
        sub.add_argument("--device", default="0")
        sub.add_argument("--imgsz", type=int, default=640)
        sub.add_argument("--batch", type=int, default=2)
        sub.add_argument("--workers", type=int, default=0)
        sub.add_argument("--project", default=str(Path(__file__).resolve().parents[1] / "runs/wheat_fourclass_joint"))
        sub.add_argument("--name", default=f"fourclass_{command}")
        sub.add_argument("--review-threshold", type=float, default=0.5 if command == "train" else None)
        if command != "predict":
            sub.add_argument("--data", required=True)
        if command == "train":
            sub.add_argument("--epochs", type=int, default=100)
            sub.add_argument("--region-weight", type=float, default=0.5)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--resume", action="store_true")
        else:
            sub.add_argument("--conf", type=float, default=0.001 if command == "val" else 0.25)
            if command == "val":
                sub.add_argument("--split", choices=("val", "test"), default="val")
            else:
                sub.add_argument("--source", required=True)
    args = parser.parse_args()
    if "," in args.device or args.batch < 1:
        parser.error("Use one GPU or cpu and a positive batch size for this experiment")
    if args.command == "train" and (args.region_weight < 0 or not torch.isfinite(torch.tensor(args.region_weight))):
        parser.error("region-weight must be finite and nonnegative")
    if args.review_threshold is not None and not 0 <= args.review_threshold <= 1:
        parser.error("review-threshold must be in [0, 1]")
    if args.command != "train" and not 0 < args.conf <= 1:
        parser.error("conf must be in (0, 1]")
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
            trainer=partial(JointTrainer, region_weight=args.region_weight, review_threshold=args.review_threshold),
            data=args.data,
            epochs=args.epochs,
            seed=args.seed,
            deterministic=True,
            resume=args.resume,
            **common,
        )
    elif args.command == "val":
        metrics = model.val(
            validator=partial(JointValidator, review_threshold=args.review_threshold),
            data=args.data,
            split=args.split,
            conf=args.conf,
            **common,
        )
        (metrics.save_dir / "per_class_metrics.csv").write_text(metrics.to_csv(), encoding="utf-8-sig")
    else:
        model.predict(
            predictor=partial(JointPredictor, review_threshold=args.review_threshold),
            source=args.source,
            conf=args.conf,
            save=True,
            **common,
        )
