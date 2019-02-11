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
    path_RGB2YCbCr = None
    path_segmentation = None
    path_project = None
    path_dataset = None

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
            if not os.path.exists(self.path_project + 'RGB2YCbCr'):
                self.path_RGB2YCbCr = os.path.join(self.path_project, 'RGB2YCbCr')
                os.mkdir(self.path_RGB2YCbCr)
                print('RGB2YCbCr Directory Created')
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

    def resize_Image(self, image, name):
        img = cv.resize(image, (600, 400))
        path_name_image = os.path.join(self.path_resize, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_YCbCr(self, image):
        img = cv.COLOR_BGR2YCrCb(image)
        return img

    def segmentation(self, image):
        pass
