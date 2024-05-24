# Computer Vision for Beam Analysis

This repository contains the code for an AI-based system capable of recognizing and analyzing handwritten sketches of engineering beam diagrams.

## Setup

All files implementing the core functionality of the system (namely `analyze.py`, `beam.py`, `main.py`, `number.py`, `relationships.py`, and `yolo.py`) are in the base directory.

* The `yolo.py` file implements the object detection stage of the workflow (stage 1) via a wrapper to the YOLOv5 programs contained in the `yolov5/` directory.
* The `number.py` file implements the number reading stage of the workflow (stage 2) via a wrapper to the SimpleHTR programs contained in the `numberhtr/` directory.
* The `beam.py`, `relationships.py`, `analyze.py` files implement the feature association stage of the workflow (stage 3).
* The `analyze.py` file also implements the structural analysis stage of the workflow (stage 4).
* The `main.py` file utilizes functions in these programs to complete the overall workflow.

Training and testing datasets are contained in the `data/` directory and the machine-learned models to be used are contained in the `models/` directory. Each of these contains subfolders named `features/`, `number/`, and `relationships/` which contain the relevant files for that model stage. Running `pip install -r requirements.txt` downloads and installs all required packages for the system to work.

## Results

Using the baseline models included, 45% of the images in the testing dataset are analyzed entirely correctly, an impressive figure considering how many model inferences are required for an image to be entirely correct.

## Usage

Proper usage of the end-to-end model is the following:

```
python3 main.py [-h] --image-name IMAGE_NAME [--features-path FEATURES_PATH] [--number-path NUMBER_PATH] [--relationships-path RELATIONSHIPS_PATH]
```

A path to an image to analyze must be provided following the `--image-name` (or `--i`) prefix. This produces structural analysis diagrams in a folder named after the image in the `runs/` directory. The path to the models to be used in the object detection, number reading, and feature association stages can be specified if they differ from the baseline models. For example, `python3 main.py --i data/test/IMG-8287.jpg` analyzes the beam system in the first testing image.

To train a new MLP, ensure the number of parameters is set in the `relationships.py` file, and run:

```
python3 relationships.py --mode create --source data/relationships/preprocessed/<FILE> --preprocess no --epochs 30 --name models/relationships/<NAME>
```

## Citation

```
@article{joffe2024cv,
    AUTHOR = {Joffe, Isaac and Qian, Yuchen and Talebi-Kalaleh, Mohammad and Mei, Qipei},
    TITLE = {A Computer Vision Framework for Structural Analysis of Hand-Drawn Engineering Sketches},
    JOURNAL = {Sensors},
    VOLUME = {24},
    YEAR = {2024},
    NUMBER = {9},
    ARTICLE-NUMBER = {2923},
    URL = {https://www.mdpi.com/1424-8220/24/9/2923},
    PubMedID = {38733029},
    ISSN = {1424-8220},
    ABSTRACT = {Structural engineers are often required to draw two-dimensional engineering sketches for quick structural analysis, either by hand calculation or using analysis software. However, calculation by hand is slow and error-prone, and the manual conversion of a hand-drawn sketch into a virtual model is tedious and time-consuming. This paper presents a complete and autonomous framework for converting a hand-drawn engineering sketch into an analyzed structural model using a camera and computer vision. In this framework, a computer vision object detection stage initially extracts information about the raw features in the image of the beam diagram. Next, a computer vision number-reading model transcribes any handwritten numerals appearing in the image. Then, feature association models are applied to characterize the relationships among the detected features in order to build a comprehensive structural model. Finally, the structural model generated is analyzed using OpenSees. In the system presented, the object detection model achieves a mean average precision of 99.1%, the number-reading model achieves an accuracy of 99.0%, and the models in the feature association stage achieve accuracies ranging from 95.1% to 99.5%. Overall, the tool analyzes 45.0% of images entirely correctly and the remaining 55.0% of images partially correctly. The proposed framework holds promise for other types of structural sketches, such as trusses and frames. Moreover, it can be a valuable tool for structural engineers that is capable of improving the efficiency, safety, and sustainability of future construction projects.},
    DOI = {10.3390/s24092923}
}
```
