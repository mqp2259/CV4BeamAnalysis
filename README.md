# Beam-Analysis

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

Paper is pending, so for now no citation is required when using this work. In the future, the proper citation will be pasted below.

`Citation.`
