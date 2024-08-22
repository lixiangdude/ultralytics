import os
import sys
from datetime import datetime

from ultralytics import YOLO

models = [
    # 'yolov8n-pgds',
    'yolov8s-pgds',
    # 'yolov8m-pgds',
    # 'yolov8l-pgds',
    # 'yolov8x-pgds',
    # 'yolov8n-mpgd-apgd',
    'yolov8s-mpgd-apgd',
    # 'yolov8m-mpgd-apgd',
    # 'yolov8l-mpgd-apgd',
    # 'yolov8x-mpgd-apgd',
    # 'yolov8n-c2fs',
    'yolov8s-c2fs',
    # 'yolov8m-c2fs',
    # 'yolov8l-c2fs',
    # 'yolov8x-c2fs',
    # 'yolov8n',
    'yolov8s',
    # 'yolov8m',
    # 'yolov8l',
    # 'yolov8x'
    ]
# Train the model
date = datetime.now().strftime('%Y-%m-%d')
dataset = 'tsingyan-voc-fisheye'
for batch_size in [8]:
    for model in models:
        # Load a model
        m = YOLO(f'{model}.yaml')  # build a new model from YAML
        train_name = f"{date}_{dataset}_{model}_{batch_size}_train"
        # log_dir = f'logs/{train_name}'
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        # sys.stdout = open(f'logs/{train_name}/train.log', 'w')
        results = m.train(
            data=f"/home/lixiang/PycharmProjects/ultralytics/ultralytics/cfg/datasets/{dataset}.yaml",
            epochs=300, imgsz=640, batch=batch_size, lr0=0.005,
            name=train_name)
