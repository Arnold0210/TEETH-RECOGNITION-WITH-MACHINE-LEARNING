#  Copyright (c) 2019. Arnold Julian Herrera Quiñones -  Cristhian Camilo Arce García.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#

import errno
import os

import cv2 as cv


class PreProcessingData:
    path_resize = None
    path_RGB2YCrCb = None
    path_segmentation = None
    path_project = None
    path_dataset = None
    path_RGB2HSV = None
    path_RGB2HSV_Full = None

    def __init__(self, PATH_PROJECT, PATH_DATASET):

        self.path_project = os.path.join(PATH_PROJECT, 'PreProcessing')
        self.path_dataset = PATH_DATASET
        try:
            if not os.path.exists(self.path_project + 'ResizeImages'):
                self.path_resize = os.path.join(self.path_project, 'ResizeImages')
                os.mkdir(self.path_resize)

                print('ResizeImages Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('ResizeImages Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'RGB2YCrCb'):
                self.path_RGB2YCrCb = os.path.join(self.path_project, 'RGB2YCrCb')
                os.mkdir(self.path_RGB2YCrCb)
                print('RGB2YCrCb Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('RGB2YCbCr Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'Segmentation'):
                self.path_segmentation = os.path.join(self.path_project, 'Segmentation')
                os.mkdir(self.path_segmentation)
                print('Segmentation Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Segmentation Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'RGB2HSV'):
                self.path_RGB2HSV = os.path.join(self.path_project, 'RGB2HSV')
                os.mkdir(self.path_segmentation)
                print('RGB2HSV Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('RGB2HSV Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'RGB2HSV_Full'):
                self.path_RGB2HSV_Full = os.path.join(self.path_project, 'RGB2HSV_Full')
                os.mkdir(self.path_segmentation)
                print('RGB2HSV_Full Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('RGB2HSV_Full Directory Already Exists.')
            else:
                raise

    def resize_Image(self, image, name):
        img = cv.resize(image, (600, 400))
        path_name_image = os.path.join(self.path_resize, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_YCrCb(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)
        path_name_image = os.path.join(self.path_RGB2YCrCb, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_hsv(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        path_name_image = os.path.join(self.path_RGB2HSV, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_hsv_Full(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_RGB2HSV_FULL)
        path_name_image = os.path.join(self.path_RGB2HSV_Full, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def segmentation(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        path_name_image = os.path.join(self.path_segmentation, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, thresh1)
        return thresh1
