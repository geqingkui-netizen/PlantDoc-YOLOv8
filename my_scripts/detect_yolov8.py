
from ultralytics import YOLO

# 加载自己训练的植物病害检测模型
model = YOLO(r"E:\盐工\code\ultralytics\runs\detect\train\weights\best.pt")

# 检测植物叶片图片
model.predict(
    source=r"E:\Dataset\PlantDoc.v1-resize-416x416.yolov8\test\images",
    conf=0.25,
    imgsz=416,
    device=0,
    save=True
)
