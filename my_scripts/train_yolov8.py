
from ultralytics import YOLO

# 1. 加载 YOLOv8n 预训练权重
model = YOLO("yolov8n.pt")

# 2. 开始训练 PlantDoc 数据集
model.train(
    data=r"E:\Dataset\PlantDoc.v1-resize-416x416.yolov8\data.yaml",
    epochs=400,
    imgsz=416,
    batch=4,
    device=0,
    workers=0,
    patience=20,
    project="runs/plantdoc",
    name="yolov8n_400epochs",
)
