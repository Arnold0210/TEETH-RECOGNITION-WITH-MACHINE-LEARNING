#  Copyright (c) 2019. Arnold Julian Herrera Quiñones -  Cristhian Camilo Arce García.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#

import os

import cv2 as cv
from tqdm import tqdm


class LoadData:
    PATH = ''

    def __init__(self, PATH_IMAGES):
        self.PATH = PATH_IMAGES

    def read_Images(self, PATH):
        images = []
        amount = len(os.listdir(PATH))
        bar = tqdm(os.listdir(PATH), ncols=amount, unit=' image')
        for i in bar:
            if ('JPG' in i) or ('jpg' in i):
                img = cv.imread(os.path.join(PATH, i))
                images.append(img)
            bar.set_description("Leyendo archivo %s" % i)
        return images

    def read_One_Image(self, PATH):
        test_image = "101_0092.JPG"
		test_image2 = "101_0164.JPG"
        cristhian_mouth = "101_0000.JPG"
        name = test_image2
        image_string = os.path.join(PATH, name)
        image = cv.imread(image_string, cv.IMREAD_COLOR)
        return image, name
