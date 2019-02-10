#  Copyright (c) 2019. Arnold Julian Herrera Quiñones -  Cristhian Camilo Arce García.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#

import Experiment.Tonality.ReadImages as ri
import cv2 as cv


class PreProcessingData:
    def BlackGreyScale(self, images):
        image_bn = cv.cvtColor(images, cv.COLOR_BGR2GRAY)

        return image_bn

    def BorderDetection(self, images):
        pass

    def ImageSegmentation(self, images):
        pass

    def ImageBinarization(self, images):
        pass
