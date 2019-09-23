#  Copyright (c) 2019. Arnold Julian Herrera Quiñones -  Cristhian Camilo Arce García.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#
import errno
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

import Source.PreProcessingData as pD
import Source.ReadImages as rI


class FeatureExtraction:
    readImages = None
    preProcessing = None

    def __init__(self, PATH_PROJECT, PATH_IMAGES):
        self.path_project = os.path.join(PATH_PROJECT, 'FeatureExtraction')
        self.path_dataset = PATH_IMAGES
        try:
            if not os.path.exists(self.path_project + 'GetColors'):
                self.path_getColor = os.path.join(self.path_project, 'GetColors')
                os.mkdir(self.path_getColor)

                print('ResizeImages Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('ResizeImages Directory Already Exists.')
            else:
                raise
        self.preProcessing = pD.PreProcessingData(PATH_PROJECT, PATH_IMAGES)
        self.readImages = rI.LoadData(PATH_PROJECT)

    def RGB2HEX(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def get_colors(self, src, number_of_colors, show_chart, name):
        modified_image = src.reshape(src.shape[0] * src.shape[1], 3)
        clf = KMeans(n_clusters=number_of_colors)
        labels = clf.fit_predict(modified_image)
        counts = Counter(labels)
        center_colors = clf.cluster_centers_
        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]

        if show_chart:
            path_image_name = os.path.join(self.path_getColor, name)
            plt.figure(figsize=(8, 6))
            plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
            plt.savefig(path_image_name)
            plt.show()

        return rgb_colors, hex_colors

    def getFeaturesVector(self, image, mask):
        features = []
        imagecpy = image.copy()
        for i in range(len(mask)):
            for j in range(len(mask[i])):
                if (j == 255):
                    # print(image[i][j])
                    features.append(imagecpy[i][j])
        return features

    def meanVector(vector_caracteristicas):
        return np.mean(vector_caracteristicas)

    def varVector(vector_caracteristicas):
        return np.var(vector_caracteristicas)

    def skewVector(vector_caracteristicas):
        return stats.skew(vector_caracteristicas)
