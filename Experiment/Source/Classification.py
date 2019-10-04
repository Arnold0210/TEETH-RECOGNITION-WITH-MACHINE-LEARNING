#  Copyright (c) 2019. Arnold Julian Herrera Quiñones -  Cristhian Camilo Arce García.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#
import csv
import os

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
        featuresFile = open(
            os.path.join(os.path.join(os.path.join(self.path_project, os.path.pardir), 'FeatureExtraction'),
                         'features.csv'), "r")
        featuresData = csv.reader(featuresFile)
        featuresFile.close()

    def classificator(self, features):
        clf = svm.SVC(gamma='scale', decision_function_shape='ovo')


cc = Classification(os.path.join(os.getcwd(), os.path.pardir))
cc.readfeatures(os.path.join(os.path.join(os.getcwd(), os.path.pardir), 'Classification'))
