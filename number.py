# Made by Isaac Joffe

import cv2
import numpy as np
from math import sqrt
import numberhtr.src.main as number_reader
import os
import time


TEMP_NAME = "temp.png"
IMAGE_HEIGHT = 28


def preprocess_image(image_name):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    image = cv2.bitwise_not(image)
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    kernel_size = int(sqrt(height * width) / 30)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size <= 0:
        kernel_size = 1
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    dimensions = (int(IMAGE_HEIGHT * width / height), IMAGE_HEIGHT)
    image = cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)
    height, width = image.shape

    offset = 2
    blank = np.full((height + 2*offset, width + 2*offset), 0, dtype=np.uint8)
    blank[offset:(offset + height), offset:(offset + width)] = image
    image = blank
    
    kernel_size = 2
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)), iterations=1)

    cv2.imwrite(TEMP_NAME, image)
    time.sleep(3)
    return TEMP_NAME


def segment_image(image_name, dimensions):
    dimensions = [int(value) for value in dimensions]
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    image = image[dimensions[1]:dimensions[3], dimensions[0]:dimensions[2]]

    cv2.imwrite(TEMP_NAME, image)
    time.sleep(3)
    return TEMP_NAME


def read_number(image_name, model):
    image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    value = number_reader.infer_number(image, model)
    print(value)
    cv2.imwrite(TEMP_NAME, image)
    time.sleep(3)
    # os.remove(TEMP_NAME)
    if not value or int(value) == 0:
        return 1
    else:
        return int(value)


def create_number_model(model_name):
    return number_reader.create_model(model_name)


def clear_number_model(number_model):
    number_reader.clear_model(number_model)
    return


def main():
    return


if __name__ == "__main__":
    main()
