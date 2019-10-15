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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.utils.multiclass import unique_labels


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
        return clf

    def validacionCruzada(self, path_dataset, features, labels, vals_to_replace):

        onlyfiles = [f for f in listdir(path_dataset) if
                     isfile(join(path_dataset, f))]
        kf = KFold(n_splits=5)
        kf.get_n_splits(onlyfiles)
        target_names = ["a1", "a2", "a3", "a35", "a4"]
        images_name = labels['Nombre de la imagen'].to_numpy().tolist()
        labels['Color'] = labels['Color'].map(vals_to_replace)
        labels_name = labels['Color'].to_numpy().tolist()
        svm_training_Score = []
        confusion_matrix_svm = []
        labels_test = []
        classification_report_svm = []
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
            SVM = self.classificatorSVM(training_features, training_label, vals_to_replace)
            svm_training_Score.append(SVM.score(training_features, training_label))

            for i in test_index:
                test_features.append(features.to_numpy()[images_name.index(str(onlyfiles[i].split('.')[0]))])
                test_label.append(labels_name[images_name.index(str(onlyfiles[i].split('.')[0]))])
            predict_label = SVM.predict(test_features)
            confusionMatrixSVM = confusion_matrix(test_label, predict_label)
            classification_report_svm.append(classification_report(test_label, predict_label))

            confusion_matrix_svm.append(
                self.plot_confusion_matrix(test_label, predict_label, target_names,
                                           title='Confusión Matrix'))
        return confusion_matrix_svm, classification_report_svm

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax


'''PATH_IMAGES_P = os.path.abspath(
    os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET - P"))
PROJECT_PATH = os.path.join(os.getcwd(), os.path.pardir)
PATH_Labels = os.path.abspath(
    os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "Labels"))
cc = Classification(PROJECT_PATH)

filefeaturespath = os.path.join(os.path.join(PROJECT_PATH, 'FeatureExtraction'), 'features3.csv')
names, features = cc.readfeatures(filefeaturespath)
labels = cc.readLabels(PATH_Labels)
vals_to_replace = {'a1': '0', 'a2': '1', 'a3': '2', 'a35': '3', 'a4': '4'}
target_names = '
cc.validacionCruzada(PATH_IMAGES_P, features, labels, vals_to_replace)'''
