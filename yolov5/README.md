# Beam-Analysis

This folder contains a minimal version of the [YOLOv5 repository](https://github.com/ultralytics/yolov5), and more information about the files within can be found there.

## Usage

To train a new YOLO model, run:

```
python3 train.py --weights '' --cfg yolov5s.yaml --data ../data/features/dataset.yaml --hyp ../data/features/hyp.yaml --epochs 35 --img 1280
```

Parameters can be adjusted freely.
