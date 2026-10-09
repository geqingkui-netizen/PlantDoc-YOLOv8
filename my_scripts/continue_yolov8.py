from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(r"E:\盐工\code\ultralytics")

    weights = (
        root
        / "my_scripts/runs/detect/runs/plantdoc/yolov8n/weights/last.pt"
    )
    data = Path(r"E:\Dataset\PlantDoc.v1-resize-416x416.yolov8\data.yaml")

    for file in (weights, data):
        if not file.is_file():
            raise FileNotFoundError(f"找不到文件：{file}")

    # 从今天第100轮结束时的模型开始训练
    model = YOLO(str(weights))

    model.train(
        data=str(data),
        epochs=300,
        imgsz=416,
        batch=4,
        device=0,
        workers=0,
        patience=0,  # 禁用提前停止
        resume=False,
        project=str(root / "runs/plantdoc"),
        name="yolov8n_from100_add300",
        exist_ok=False,
    )


if __name__ == "__main__":
    main()