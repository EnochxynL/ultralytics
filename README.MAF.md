# YOLO11-MAF Detection Model Codebase

A custom training codebase forked form [ultralytics](https://github.com/ultralytics/ultralytics).

## Quick Start

Train script for YOLO11-Nano: `my_train.py`

Train script for ablation experiment of YOLO11-MAF: `my_batch_train.py`

Validate script for models (for any custom models): `my_val.py`, `my_batch_val.py`

`my_batch_val.py` gives the comparison of metrics among trained models.

## Model

The model is based on YOLO11-Nano, adding with Efficient Channel Attention (ECA), Multi-Scale Convolutional Attention (MSCA), Large Separable Kernel Attention (LSKA) and Coordinate Attention (CoordAtt) modules. The model config file is in `my_model` folder with models for ablation experiment. The full model config is `yolo11-maf0.yaml`.

## Dataset

The dataset is from this [link](https://github.com/tranvietcuong03/Basketball_Detection) with 8521 train images, 812 validate images and 406 test images. The download script of the dataset is `my_dataset/Basketball_Detection.sh`, just open terminal in `my_dataset` folder and run the script. The script also works in Windows.
