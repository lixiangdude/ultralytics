import os

from ultralytics import YOLOWorld, YOLO

# Initialize a YOLO-World model
# model = YOLOWorld('yolov8l-worldv2.pt')  # or select yolov8m/l-world.pt for different sizes
model = YOLO('yolov8l.pt')
# model.set_classes(['trash', 'drying along the street', 'trash can'])
# Execute inference with the YOLOv8s-world model on the specified image
model.train(data='/home/lixiang/PycharmProjects/ultralytics/ultralytics/cfg/datasets/tsingyan-pano.yaml', epochs=100, imgsz=640, batch=8)