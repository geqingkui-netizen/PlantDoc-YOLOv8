"""Validate wheat checkpoints on the validation or test split."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证四分类模型; 调参用 val, 最终评估用 test.")
    parser.add_argument("--weights", default=str(ROOT / "runs/wheat/innovation_b/pair_seed0/weights/best.pt"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)

    metrics = model.val(
        data="E:/Dataset/dataset_four_classes/data.yaml",
        imgsz=640,
        device=0,
        split=args.split,
        project=str(ROOT / "runs/wheat/innovation_b_eval"),
        name=f"{Path(args.weights).resolve().parents[1].name}_{args.split}",
    )

    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.75:", metrics.box.map75)
    print("mAP@0.5:0.95:", metrics.box.map)
