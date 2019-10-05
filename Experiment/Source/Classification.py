#  Copyright (c) 2019. Arnold Julian Herrera Quiñones -  Cristhian Camilo Arce García.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#
import os

import pandas as pd
from sklearn import svm


class Classification:
    def __init__(self, PATH_PROJECT):
        self.path_project = os.path.join(PATH_PROJECT, 'Classification')
        '''try:
            if not os.path.exists(self.path_project + 'Classification'):
                self.path_getColor = os.path.join(self.path_project, 'Classification')
                os.mkdir(self.path_getColor)
                print('Classification Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Classification Directory Already Exists.')
            else:
                raise'''

    def readfeatures(self, features_Path):
        # featuresFile = open(features_Path, "r")
        # featuresData = csv.reader(featuresFile)
        featuresFile = pd.read_csv(features_Path, sep=',', header=None)
        names = featuresFile.iloc[:, 0]
        features = featuresFile.iloc[:, 1:]
        return names, features

    def classificator(self, features):
        clf = svm.SVC(gamma='scale', decision_function_shape='ovo')


PROJECT_PATH = os.path.join(os.getcwd(), os.path.pardir)
PATH_IMAGES = os.path.abspath(
    os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET"))
PATH_Labels = os.path.abspath(
    os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "Labels"))
PATH_LabelsXML = os.path.abspath(os.path.join(
    os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "Labels"), "LabelsXML"))
PATH_IMAGES_SNIPPING = os.path.abspath(
    os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET - Recortado"))

filefeaturespath = os.path.join(os.path.join(PROJECT_PATH, 'FeatureExtraction'), 'features.csv')
cc = Classification(PROJECT_PATH)
names, feattures = cc.readfeatures(filefeaturespath)
print(feattures)
