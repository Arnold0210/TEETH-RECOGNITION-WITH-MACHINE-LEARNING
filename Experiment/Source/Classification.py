#  Copyright (c) 2019. Arnold Julian Herrera Quiñones -  Cristhian Camilo Arce García.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#
import errno
import os
from os import listdir
from os.path import isfile, join

import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold


class Classification:
    def __init__(self, PATH_PROJECT):
        self.path_project = os.path.join(PATH_PROJECT, 'Classification')
        try:
            if not os.path.exists(self.path_project + 'SVM'):
                self.path_getColor = os.path.join(self.path_project, 'SVM')
                os.mkdir(self.path_getColor)
                print('SVM Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('SVM Directory Already Exists.')
            else:
                raise

    def readfeatures(self, features_Path):
        # featuresFile = open(features_Path, "r")
        # featuresData = csv.reader(featuresFile)
        featuresFile = pd.read_csv(features_Path, sep=',', header=None)
        names = featuresFile.iloc[:, 0]
        features = featuresFile.iloc[:, 1:]
        shapefile = featuresFile.shape
        col = []
        for x in range(0, shapefile[1]):
            if x == 0:
                col.append("NAME")
            else:
                col.append("VALOR-" + str(x))
        featuresFile.columns = col
        # print(featuresFile)
        return names, features

    def readLabels(self, labels_path):
        labels_path = os.path.join(labels_path, 'Labels.csv')
        labels = pd.read_csv(labels_path, sep=',', header=[0])
        return labels

    def classificatorSVM(self, features, labels):
        X = []
        for f in features:
            X.append(f)
        vals_to_replace = {'a1': '0', 'a2': '1', 'a3': '2', 'a35': '3', 'a4': '4'}
        labels['Color'] = labels['Color'].map(vals_to_replace)
        # labels['Color'] = labels['Color'].map(vals_to_replace)
        label = labels.values
        tags = []
        for tag in label:
            tags.append(tag[1])
        clf = svm.SVC(gamma='scale')
        clf.fit(X, tags)
        print(clf.score(X, tags))

    def validacionCruzada(self, path_dataset):
        onlyfiles = [f for f in listdir(path_dataset) if
                     isfile(join(path_dataset, f))]
        kf = KFold(n_splits=5.)
        kf.get_n_splits(onlyfiles)
        print(kf)
        for train_index, test_index in kf.split(onlyfiles):
            for i in train_index:
                pass
            # aqui se manda cada archivo de train a la SVM y se puede hacer otro for al mismo nivel para el testing
            for i in test_index:
                # la misma vaina xd y se saca la métrica
                print(str(i) + " " + onlyfiles[i])
