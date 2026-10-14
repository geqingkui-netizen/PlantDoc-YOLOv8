"""Train the wheat baseline and geometric-risk ablations.

各组重新加载同一份 yolov8n.pt, 使用相同 seed、640 输入、batch=4、patience=30.
--pair 是额外损失的初始试验权重, 不代表已验证的最优值; --dfl-control 设置 DFL 对照权重.
例: python my_scripts/train_yolov8.py --experiment pair
例: python my_scripts/train_yolov8.py --experiment pair_squared
例: python my_scripts/train_yolov8.py --experiment all --epochs 400 --seed 0
"""

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("baseline", "dfl", "single", "paired_raw", "pair", "pair_squared", "all"),
        default="pair",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair", type=float, default=1.0)
    parser.add_argument("--dfl-control", type=float, default=3.0)
    args = parser.parse_args()
    from ultralytics import YOLO

    experiments = {
        "baseline": dict(pair=0.0, pair_mode="paired", dfl=1.5),
        "dfl": dict(pair=0.0, pair_mode="paired", dfl=args.dfl_control),
        "single": dict(pair=args.pair, pair_mode="single", dfl=1.5),
        "paired_raw": dict(pair=args.pair, pair_mode="paired_raw", dfl=1.5),
        "pair": dict(pair=args.pair, pair_mode="paired", dfl=1.5),
        "pair_squared": dict(pair=args.pair, pair_mode="paired_squared", dfl=1.5),
    }
    selected = experiments if args.experiment == "all" else {args.experiment: experiments[args.experiment]}
    for experiment, settings in selected.items():
        model = YOLO(str(ROOT / "yolov8n.pt"))
        model.train(
            data="E:/Dataset/dataset_four_classes/data.yaml",
            epochs=args.epochs,
            imgsz=640,
            batch=4,
            device=0,
            workers=0,
            patience=30,
            seed=args.seed,
            deterministic=True,
            resume=False,
            project=str(ROOT / "runs" / "wheat" / "innovation_b"),
            name=f"{experiment}_seed{args.seed}",
            **settings,
        )
        del model
        gc.collect()
