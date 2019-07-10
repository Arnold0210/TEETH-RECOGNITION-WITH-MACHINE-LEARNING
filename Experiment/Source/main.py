#  Copyright (c) 2019. Arnold Julian Herrera Quinones -  Cristhian Camilo Arce Garcia.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#

import errno
import os

import cv2 as cv
import easygui

import Source.PreProcessingData as pD
import Source.ReadImages as rI


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
        read = self.readimages
        pp = self.preprocessing
        img, name = read.read_One_Image(self.PATH_IMAGES)
        height_ori, width_ori, depth_ori = img.shape
        print("Image original shape: \n Height:", height_ori, ", Width:", width_ori)
        img_resize = pp.resize_Image(img, name)
        height_res, width_res, depth_res = img_resize.shape
        print("name", name)
        print("Image Resize shape: \n Height:", height_res, ", Width:", width_res)
        img_rgb2ycbcr = pp.rgb_2_YCrCb(img_resize, name)
        img_rgb2hsv = pp.rgb_2_HSV(img_resize, name)
        img_rgb2hsv = pp.rgb_2_LAB(img_resize, name)
        img_rgb2hsv = pp.rgb_2_Lab(img_resize, name)
        img_Segmentation = pp.segmentation(img_resize, name)
        easygui.msgbox("Image original shape: \n Height:" + str(height_ori) + "px, Width:" + str(width_ori) + "px" +
                       "\n Image Resize shape: \n Height:" + str(height_res) + "px, Width:" + str(width_res) + "px",
                       image=os.path.join(os.path.join(os.getcwd(), os.path.pardir),
                                          'PreProcessing/Segmentation/' + name),
                       title="Image Shape - PreProcessing ")


if __name__ == '__main__':
    tesis = MainClass()
    tesis.main_run()
    print('Se ha finalizado la ejecución del experimento')
    exit(0)
