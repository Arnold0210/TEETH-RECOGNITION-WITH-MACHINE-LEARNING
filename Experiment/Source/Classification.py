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
import pandas as pd
from sklearn import svm
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


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
        tags = []
        for tag in labels:
            tags.append(tag)
        clf = svm.SVC(gamma='scale')
        clf.fit(X, tags)
        return clf

    def DecisionTree(self, features, labels):
        dt = DecisionTreeClassifier(random_state=30, max_depth=300)
        X = []
        for f in features:
            X.append(f)
        tags = []
        for tag in labels:
            tags.append(tag)
        dt.fit(X, tags)

        return dt

    def KNN(self, features, labels):
        knn = KNeighborsClassifier(n_neighbors=200, algorithm='auto', weights='distance', n_jobs=-1)
        X = []
        for f in features:
            X.append(f)
        tags = []
        for tag in labels:
            tags.append(tag)
        knn.fit(X, tags)
        return knn

    def classification(self, path_dataset, features, labels, n_splits, tags, target_names, vals_to_replace):
        pd.options.mode.chained_assignment = None
        onlyfiles = [f for f in listdir(path_dataset) if
                     isfile(join(path_dataset, f))]
        k_folds = KFold(n_splits=n_splits)
        k_folds.get_n_splits(onlyfiles)
        SVM = []
        KNN = []
        DT = []
        for stage, feature in zip(labels, features):
            stage['Color'] = stage['Color'].map(vals_to_replace)
            labels_color = stage['Color'].to_numpy().tolist()
            images_name = stage['Nombre de la imagen']#.to_numpy().tolist()
            svm_training_Score = []
            confusion_matrix_svm = []
            confusion_matrix_dt = []
            confusion_matrix_knn = []
            classification_report_svm = []
            classification_report_dt = []
            classification_report_knn = []
            score_accuracy_SVM = []
            score_accuracy_DT = []
            score_accuracy_KNN = []
            for train_index, test_index in k_folds.split(onlyfiles):
                train_label = []
                test_label = []
                train_features = []
                test_features = []
                for i in train_index:

                    print(str(images_name))
                    train_features.append(feature.to_numpy()[str(images_name.str(onlyfiles[i].split('.')[0]))])

                    #print(images_name.str.)
                    train_features.append(feature.to_numpy()[images_name.str(onlyfiles[i].split('.')[0])])

                    train_label.append(labels_color[images_name.index(str(onlyfiles[i].split('.')[0]))])
                SVM_Classifier = self.classificatorSVM(train_features, train_label)
                DT_Classifier = self.DecisionTree(train_features, train_label)
                KNN_Classifier = self.KNN(train_features, train_label)

                for i in test_index:
                    test_features.append(feature.to_numpy()[images_name.index(str(onlyfiles[i].split('.')[0]))])
                    test_label.append(labels_color[images_name.index(str(onlyfiles[i].split('.')[0]))])
                predict_label_SVM = SVM_Classifier.predict(test_features)
                predict_label_DT = DT_Classifier.predict(test_features)
                predict_label_KNN = KNN_Classifier.predict(test_features)

                confusionMatrixSVM = confusion_matrix(test_label, predict_label_SVM)
                confusionMatrixDT = confusion_matrix(test_label, predict_label_DT)
                confusionMatrixKNN = confusion_matrix(test_label, predict_label_KNN)

                classification_report_svm.append(
                    classification_report(test_label, predict_label_SVM, labels=tags,
                                          target_names=target_names, sample_weight=None, digits=5,
                                          output_dict=False))
                classification_report_dt.append(
                    classification_report(test_label, predict_label_DT, labels=tags,
                                          target_names=target_names, sample_weight=None, digits=5,
                                          output_dict=False))
                classification_report_knn.append(
                    classification_report(test_label, predict_label_KNN, labels=tags,
                                          target_names=target_names, sample_weight=None, digits=5,
                                          output_dict=False))

                confusion_matrix_svm.append(confusionMatrixSVM)
                confusion_matrix_dt.append(confusionMatrixDT)
                confusion_matrix_knn.append(confusionMatrixKNN)

                score_accuracy_SVM.append(accuracy_score(test_label, predict_label_SVM))
                score_accuracy_DT.append(accuracy_score(test_label, predict_label_DT))
                score_accuracy_KNN.append(accuracy_score(test_label, predict_label_KNN))
            SVM_RESULTS = [confusion_matrix_svm, classification_report_svm, score_accuracy_SVM]
            DT_RESULTS = [confusion_matrix_dt, classification_report_dt, score_accuracy_DT]
            KNN_RESULTS = [confusion_matrix_knn, classification_report_knn, score_accuracy_KNN]
            SVM.append(SVM_RESULTS)
            KNN.append(KNN_RESULTS)
            DT.append(DT_RESULTS)
        return SVM, KNN, DT

    def CrossValidation(self, image, labels, test_size):
        X = []
        Y = []
        X_train, X_test, y_train, y_test = train_test_split(image, labels, test_size=test_size)
        X.append(X_train)
        X.append(X_test)
        Y.append(y_train)
        Y.append(y_test)
        print()
        return X, Y

    def ROC_CURVE(self, label_test, label_score):
        n_classes = label_test.shape
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(label_test[:, i], label_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(label_test.ravel(), label_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot of a ROC curve for a specific class
        plt.figure()
        plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
