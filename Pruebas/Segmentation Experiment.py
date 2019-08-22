import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread('/DATASET/101_0000.JPG')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Convert from RGB to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

images = []
for i in [0, 1, 2]:
    colour = hsv.copy()
    if i != 0: colour[:,:,0] = 0
    if i != 1: colour[:,:,1] = 255
    if i != 2: colour[:,:,2] = 255
    images.append(colour)

hsv_stack = np.vstack(images)
rgb_stack = cv2.cvtColor(hsv_stack, cv2.COLOR_HSV2RGB)
show(rgb_stack)