from ultralytics import YOLO
import os
import cv2 as cv2
import numpy as np
import datetime

if __name__ == '__main__':
    # Load a model
    # model = YOLO("custom/cirno-basketball/model1.yaml")
    model = YOLO("yolo11.yaml")

    # Train the model
    train_results = model.train(
        data="custom/cirno-basketball/data1.yaml",  # path to dataset YAML
        epochs=64,  # number of training epochs
        imgsz=640,  # training image size
        device=0  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # amp=False,  # 关闭自动混合精度
    )
