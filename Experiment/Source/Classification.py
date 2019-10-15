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

    def classificatorSVM(self, features, labels, vals_to_replace):
        X = []
        for f in features:
            X.append(f)
        # labels['Color'] = labels['Color'].map(vals_to_replace)
        # labels['Color'] = labels['Color'].map(vals_to_replace)
        # label = labels.values
        tags = []
        for tag in labels:
            tags.append(tag)
        clf = svm.SVC(gamma='scale')
        clf.fit(X, tags)
        return clf.score(X, tags)

    def validacionCruzada(self, path_dataset, features, labels, vals_to_replace):

        onlyfiles = [f for f in listdir(path_dataset) if
                     isfile(join(path_dataset, f))]
        kf = KFold(n_splits=5)
        kf.get_n_splits(onlyfiles)

        images_name = labels['Nombre de la imagen'].to_numpy().tolist()
        labels['Color'] = labels['Color'].map(vals_to_replace)
        labels_name = labels['Color'].to_numpy().tolist()
        result_training = []
        for train_index, test_index in kf.split(onlyfiles):
            training_label = []
            test_label = []
            training_features = []
            test_features = []

            for i in train_index:
                training_features.append(features.to_numpy()[images_name.index(str(onlyfiles[i].split('.')[0]))])
                training_label.append(labels_name[images_name.index(str(onlyfiles[i].split('.')[0]))])
                # print(onlyfiles[i])
                # print(labels_name[images_name.index(str(onlyfiles[i].split('.')[0]))])
            result_training.append(self.classificatorSVM(training_features, training_label, vals_to_replace))
            for i in test_index:
                test_features.append(features.to_numpy()[images_name.index(str(onlyfiles[i].split('.')[0]))])
                test_label.append(labels_name[images_name.index(str(onlyfiles[i].split('.')[0]))])
        print(result_training)


PATH_IMAGES_P = os.path.abspath(
    os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET - P"))
PROJECT_PATH = os.path.join(os.getcwd(), os.path.pardir)
PATH_Labels = os.path.abspath(
    os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "Labels"))
cc = Classification(PROJECT_PATH)

filefeaturespath = os.path.join(os.path.join(PROJECT_PATH, 'FeatureExtraction'), 'features3.csv')
names, features = cc.readfeatures(filefeaturespath)
labels = cc.readLabels(PATH_Labels)
vals_to_replace = {'a1': '0', 'a2': '1', 'a3': '2', 'a35': '3', 'a4': '4'}

# cc.classificatorSVM(features, labels, vals_to_replace)
cc.validacionCruzada(PATH_IMAGES_P, features, labels, vals_to_replace)
