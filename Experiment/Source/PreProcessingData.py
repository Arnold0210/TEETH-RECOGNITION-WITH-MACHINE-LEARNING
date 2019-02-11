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
    path_project = ''
    path_dataset = ''

    def __init__(self, PATH_PROJECT, PATH_DATASET):
        self.path_project = os.path.join(PATH_PROJECT, 'PreProcessing')
        self.path_dataset = PATH_DATASET
        try:
            if not os.path.exists(self.path_project + 'ResizeImages'):
                os.mkdir(os.path.join(self.path_project, 'ResizeImages'))
                print('ResizeImages Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('ResizeImages Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'RGB2YCbCr'):
                os.mkdir(os.path.join(self.path_project, 'RGB2YCbCr'))
                print('RGB2YCbCr Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('RGB2YCbCr Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'Segmentation'):
                os.mkdir(os.path.join(self.path_project, 'Segmentation'))
                print('Segmentation Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Segmentation Directory Already Exists.')
            else:
                raise

    def resize_Image(self, image):
        img = cv.resize(image, (600, 400))
        return img

    def rgb_2_YCbCr(self, image):

        img = cv.COLOR_BGR2YCrCb(image)
        return img

    def segmentation(self, image):
        pass
