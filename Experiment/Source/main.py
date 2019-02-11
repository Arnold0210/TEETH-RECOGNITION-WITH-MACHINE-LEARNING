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

import Experiment.Source.PreProcessingData as pD
import Experiment.Source.ReadImages as rI


def show(image):
    cv.imshow('Imagen ', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


class MainClass:
    PROJECT_PATH = os.path.join(os.getcwd(), os.path.pardir)
    PATH_IMAGES = os.path.abspath(
        os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET"))
    readimages = None
    preprocessing = None

    def __init__(self):
        try:
            if not os.path.exists(self.PROJECT_PATH + 'PreProcessing'):
                os.mkdir(os.path.join(self.PROJECT_PATH, 'PreProcessing'))
                print('Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('PreProcessing Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.PROJECT_PATH + 'FeatureExtraction'):
                os.mkdir(os.path.join(self.PROJECT_PATH, 'FeatureExtraction'))
                print('Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('FeatureExtraction Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.PROJECT_PATH + 'Sampling'):
                os.mkdir(os.path.join(self.PROJECT_PATH, 'Sampling'))
                print('Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Sampling Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.PROJECT_PATH + 'Classification'):
                os.mkdir(os.path.join(self.PROJECT_PATH, 'Classification'))
                print('Directory Classification Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Classification Directory Already Exists.')
            else:
                raise
        self.readimages = rI.LoadData(self.PATH_IMAGES)
        self.preprocessing = pD.PreProcessingData(self.PROJECT_PATH, self.PATH_IMAGES)

    def main_run(self):
        # read = self.readimages
        # pp = self.preprocessing
        pass


if __name__ == '__main__':
    tesis = MainClass()
