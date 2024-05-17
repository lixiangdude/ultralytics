import os

from ultralytics import YOLOWorld

# Initialize a YOLO-World model
# model = YOLOWorld('yolov8l-worldv2.pt')  # or select yolov8m/l-world.pt for different sizes
model = YOLOWorld('/home/lixiang/PycharmProjects/ultralytics/runs/detect/train70/weights/best.pt')
model.set_classes(['trash', 'drying along the street', 'manhole cover', 'trash can'])
# Execute inference with the YOLOv8s-world model on the specified image
for img in os.listdir('/home/lixiang/下载/全景图片切分'):

    results = model.predict(source=os.path.join(f'/home/lixiang/下载/全景图片切分/{img}'))

    # Show results
    results[0].show()