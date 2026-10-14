"""Predict wheat disease targets with an experiment checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用四分类实验权重预测图片。")
    parser.add_argument("--weights", default=str(ROOT / "runs/wheat/innovation_b/pair_seed0/weights/best.pt"))
    parser.add_argument("--source", default="E:/Dataset/dataset_four_classes/images/test")
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=0.25,
        imgsz=640,
        device=0,
        save=True,
        project=str(ROOT / "runs/wheat/innovation_b_predict"),
        name=Path(args.weights).resolve().parents[1].name,
    )
