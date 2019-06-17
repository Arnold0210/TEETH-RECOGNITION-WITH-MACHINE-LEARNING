#  Copyright (c) 2019. Arnold Julian Herrera Quiñones -  Cristhian Camilo Arce García.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#

import errno
import os
import numpy as np
import cv2 as cv


class PreProcessingData:
    path_resize = None
    path_RGB2YCrCb = None
    path_RGB2HSV = None
    path_RGB2Lab = None
    path_RGB2LAB = None
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
            if not os.path.exists(self.path_project + 'HSV'):
                self.path_RGB2HSV = os.path.join(self.path_project, 'HSV')
                os.mkdir(self.path_RGB2HSV)
                print('HSV Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('HSV Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'LAB'):
                self.path_RGB2LAB = os.path.join(self.path_project, 'LAB')
                os.mkdir(self.path_RGB2LAB)
                print('LAB Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('LAB Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.path_project + 'Lab_'):
                self.path_RGB2Lab = os.path.join(self.path_project, 'Lab_')
                os.mkdir(self.path_RGB2Lab)
                print('Lab_ Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Lab_ Directory Already Exists.')
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

    def rgb_2_YCrCb(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_RGB2YCR_CB)
        path_name_image = os.path.join(self.path_RGB2YCrCb, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_HSV(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_RGB2HSV_FULL)
        path_name_image = os.path.join(self.path_RGB2HSV, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_HSI(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_HS)
        path_name_image = os.path.join(self.path_RGB2HSI, name)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_LAB(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_RGB2LAB)
        path_name_image = os.path.join(self.path_RGB2LAB, name)
        if os.path.exists(path_name_image):
            pass
        else:

            cv.imwrite(path_name_image, img)
        return img

    def rgb_2_Lab(self, image, name):
        img = cv.cvtColor(image, cv.COLOR_RGB2Lab)
        path_name_image = os.path.join(self.path_RGB2Lab, name)
        print('Exitoso RGB2_LAB')
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, img)
        return img

    def segmentation(self, image, name):
        img = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
        ret,img= cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
        #thresh1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,37,2)#threshold(img, 127, 255, cv.THRESH_BINARY)
        path_name_image = os.path.join(self.path_segmentation,"mask_"+name)
        equalize= cv.equalizeHist(img)
        canny_image = cv.Canny(equalize,250,255)
        canny_image = cv.convertScaleAbs(canny_image)
        kernel = np.ones((3,3), np.uint8)
        dilated_image = cv.dilate(canny_image,kernel,iterations=1)
        new,contours = cv.findContours(dilated_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours= sorted(contours, key = cv.contourArea, reverse = True)[:10]
        c=contours[0]
        final = cv.drawContours(img, [c], -1, (255,0, 0), 3)
        img_rgb =  cv.imread(image)
        mask = np.zeros(img_rgb.shape,np.uint8)
        new_image = cv.drawContours(mask,[c],0,255,-1,)
        new_image = cv.bitwise_and(img_rgb,img_rgb,mask=mask)
        if os.path.exists(path_name_image):
            pass
        else:
            cv.imwrite(path_name_image, new_image)
        return new_image
