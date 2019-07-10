import cv2 as cv, os
import numpy as np

mouth_cascade = cv.CascadeClassifier('haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')
ds_factor = 0.5
PATH = os.path.abspath(
    os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET"))

image_string = os.path.join(PATH, "101_0000.JPG")
image = cv.imread(image_string, cv.IMREAD_COLOR)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
for (x, y, w, h) in mouth_rects:
    y = int(y - 0.15 * h)
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv.imshow('Mouth Detector', image)
