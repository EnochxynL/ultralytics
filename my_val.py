from ultralytics import YOLO
import os
import cv2 as cv2
import numpy as np
import datetime

import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']

if __name__ == '__main__':
    # Load a model
    # model = YOLO("custom/cirno-basketball/model1.yaml")
    # model = YOLO("yolo11n.pt")
    # model = YOLO("best/train32.3/weights/best.pt") # val3
    model = YOLO("best/train28.y/weights/best.pt") # val4
    # model = YOLO("best-e0.pt") # val4

    # Train the model
    metrics = model.val(
        data="custom/Basketball/data2.yaml",  # path to dataset YAML
        imgsz=640,  # training image size
        device=0  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # amp=False,  # 关闭自动混合精度
    )
