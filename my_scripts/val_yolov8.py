from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(
        r"E:\盐工\code\ultralytics\my_scripts\runs\detect\runs\plantdoc\yolov8n\weights\best.pt"
    )

    metrics = model.val(
        data=r"E:\Dataset\PlantDoc.v1-resize-416x416.yolov8\data.yaml",
        imgsz=416,
        device=0,
    )

    print("mAP@0.5:", metrics.box.map50)
    print("mAP@0.5:0.95:", metrics.box.map)