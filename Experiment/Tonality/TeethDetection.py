
#  Copyright (c) 2019. Arnold Julian Herrera Quiñones
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#

import os, cv2 as cv, numpy as np, matplotlib.pyplot as plt, glob, csv, pandas as pd, imageio, random
from time import time
from tqdm import tqdm
from progress.bar import Bar
from scipy import ndimage as ndi
from matplotlib import colors
from sklearn.tree import DecisionTreeClassifier

PATH = os.path.abspath(os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET - copia"))
print(PATH)


def show(image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image, interpolation='nearest')
    plt.show()


def readAllImagesPath(PATH):
    file_List = []
    cantidad = len(glob.glob(PATH + "*"))
    procesing = Bar('Leyendo archivos:', max=cantidad)
    bar = tqdm(os.walk(PATH))
    for dirName, subdirList, fileList in bar:
        for i in fileList:
            if ('JPG' in i) or ('jpg' in i):
                file_List.append(i)
            bar.set_description("Leyendo archivo %s" % i)
        procesing.next()
    procesing.finish()
    return file_List


def resizeAllImages(file_List, PATHSRC, PATHRESIZE):
    start_time = time()
    if not os.path.exists(PATHRESIZE):
        os.mkdir(PATHRESIZE)
    resizing = tqdm(file_List)
    for i in resizing:
        resizing.set_description("Re escalando imagen %s" % i)
        src = PATHSRC + i
        image = cv.imread(src, 1)
        image = cv.resize(image, (600, 200))
        if not os.path.exists(PATHRESIZE + i):
            cv.imwrite(os.path.join(PATHRESIZE, i), image)
    end_time = time()
    elapsed_time = end_time - start_time
    print("Tiempo de re-escalar imagenes: %0.10f segundos." % elapsed_time)


def readImages(PATH, fileList):
    start_time = time()
    images = []
    reading = tqdm(fileList)
    for i in reading:
        reading.set_description("Cargando imagen %s" % i)
        src = PATH + i
        image = cv.imread(src)
        images.append(image)
    end_time = time()
    elapsed_time = end_time - start_time
    print("Tiempo cargando imagenes a un vector: %0.10f segundos." % elapsed_time)
    return images


def imagesRGB2HSV(imagesRGB, directory, files):
    imagesHSV = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + 'HSV/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = directory + 'HSV/' + 'RGB2HSV/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    HSV_ = Bar('Convirtiendo RGB a HSV:', max=len(files) * 4)
    for i, file in zip(imagesRGB, files):
        i = cv.cvtColor(i, cv.COLOR_RGB2HSV)
        imagesHSV.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + 'dataset'
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + '/' + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        HSV_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0: colour[:, :, 0] = 0
            if l != 1: colour[:, :, 1] = 255
            if l != 2: colour[:, :, 2] = 255
            fil = 'HSV_' + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            HSV_.next()
    HSV_.finish()
    return imagesHSV


def imagesBGR2HSV(images, directory, files):
    images_ = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + 'HSV/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    folder = directory + 'HSV/' + 'BGR2HSV/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    HSV_ = Bar('Convirtiendo BGR a HSV:', max=len(files) * 4)
    for i, file in zip(images, files):
        i = cv.cvtColor(i, cv.COLOR_BGR2HSV)
        images_.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + 'dataset'
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + '/' + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        HSV_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0: colour[:, :, 0] = 0
            if l != 1: colour[:, :, 1] = 255
            if l != 2: colour[:, :, 2] = 255
            fil = 'HSV_' + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            HSV_.next()
    HSV_.finish()
    return images_


def imagesBGR2RGB(images, directory, files):
    imagesR = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + 'RGB/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    RGB_ = Bar('Convirtiendo BGR a RGB:', max=len(files) * 4)
    for i, file in zip(images, files):
        i = cv.cvtColor(i, cv.COLOR_BGR2RGB)
        imagesR.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + 'dataset'
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + '/' + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        RGB_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0: colour[:, :, 0] = 0
            if l != 1: colour[:, :, 1] = 0
            if l != 2: colour[:, :, 2] = 0
            fil = 'RGB_' + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            RGB_.next()
    RGB_.finish()
    return imagesR


def imagesBGR2YCR_CB(images, directory, files):
    imagesycr_cb = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + 'YCR_CB/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    YCR_CB_ = Bar('Convirtiendo BGR a YCR_CB:', max=len(files) * 4)
    for i, file in zip(images, files):
        i = cv.cvtColor(i, cv.COLOR_BGR2YCR_CB)
        imagesycr_cb.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + 'dataset'
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + '/' + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        YCR_CB_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0: colour[:, :, 0] = 0
            if l != 1: colour[:, :, 1] = 0
            if l != 2: colour[:, :, 2] = 0
            fil = 'YCR_CB_' + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            YCR_CB_.next()
    YCR_CB_.finish()
    return imagesycr_cb


def imagesRGB2YCR_CB(images, directory, files):
    imagesycr_cb = []
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + 'RGB2YCR_CB/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    YCR_CB_ = Bar('Convirtiendo BGR a YCR_CB:', max=len(files) * 4)
    for i, file in zip(images, files):
        i = cv.cvtColor(i, cv.COLOR_)
        imagesycr_cb.append(i)
        (nomArch, ext) = os.path.splitext(file)
        src = folder + nomArch
        src2 = folder + 'dataset'
        if not os.path.exists(src):
            os.mkdir(src)
        if not os.path.exists(src2):
            os.mkdir(src2)
        fil = nomArch + ".JPG"
        if not os.path.exists(src2 + '/' + fil):
            cv.imwrite(os.path.join(src2, fil), i)
        YCR_CB_.next()
        for l in [0, 1, 2]:
            colour = i.copy()
            if l != 0: colour[:, :, 0] = 0
            if l != 1: colour[:, :, 1] = 0
            if l != 2: colour[:, :, 2] = 0
            fil = 'YCR_CB_' + str(l) + ".JPG"
            cv.imwrite(os.path.join(src, fil), colour)
            YCR_CB_.next()
    YCR_CB_.finish()
    return imagesycr_cb


def imagesGaussianBlur(images_hsv, directory):
    for i in images_hsv:
        i = cv.GaussianBlur(i, (5, 5), 0)
    return images_hsv


def extractFeatures(PATH, fileList):
    matrix_data = []
    bar = tqdm(fileList)
    for file in bar:
        bar.set_description("Procesando Imagen %s" % file)
        src = PATH + file
        image = cv.imread(src, 1)
        image_f = []
        for row in image:
            for col in row:
                for pixel in col:
                    image_f.append(pixel)
        matrix_data.append(image_f)
    return matrix_data


def standar_matrix(matrix_data):
    maxi = []
    for c in matrix_data:
        maxi.append(len(c))
    maximo = max(maxi)
    for i in matrix_data:
        if len(i) < maximo:
            dif = maximo - len(i)
            for j in range(dif):
                i.append(0)
    return maxi


def readlabels(PATH):
    with open(os.path.join(PATH, 'labels.txt')) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def show_hsv_hist(image):
    plt.figure(figsize=(20, 3))
    histr = cv.calcHist([image], [0], None, [180], [0, 180])
    plt.xlim([0, 180])
    colours = [colors.hsv_to_rgb((i / 180, 1, 0.8)) for i in range(0, 180)]
    plt.bar(range(0, 180), np.squeeze(histr), color=colours, edgecolor=colours, width=1)
    plt.title('Hue')
    plt.show()
    plt.figure(figsize=(20, 3))
    histr = cv.calcHist([image], [1], None, [256], [0, 256])
    plt.xlim([0, 256])
    colours = [colors.hsv_to_rgb((0, i / 256, 1)) for i in range(0, 256)]
    plt.bar(range(0, 256), np.squeeze(histr), color=colours, edgecolor=colours, width=1)
    plt.title('Saturation')
    plt.show()

    plt.figure(figsize=(20, 3))
    histr = cv.calcHist([image], [2], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, 1, i / 256)) for i in range(0, 256)]
    plt.bar(range(0, 256), np.squeeze(histr), color=colours, edgecolor=colours, width=1)
    plt.title('Value')
    plt.show()


def ImageSegmentation(images, directory, files):
    start_time = time()
    src = directory + 'SegmentationHSV'
    if not os.path.exists(src):
        os.mkdir(src)
    segmentation = Bar('Segmentando imagenes:', max=len(images))
    for image, file in zip(images, files):
        ycrcbmin = np.array((170, 40, 235))
        ycrcbmax = np.array((180, 140, 245))
        ###imagebar
        show_hsv_hist(image)
        ###endbar
        skin_ycrcb = cv.inRange(image, ycrcbmin, ycrcbmax)
        kernel = np.ones((5, 5), np.uint8)
        img_erode = cv.erode(skin_ycrcb, kernel, iterations=1)
        holesimg = ndi.binary_fill_holes(img_erode).astype(np.int)
        imageio.imwrite(os.path.join(src, file), holesimg)
        segmentation.next()
    # break
    segmentation.finish()
    end_time = time()
    elapsed_time = end_time - start_time
    print("Tiempo de Segmentación: %0.10f segundos." % elapsed_time)


def writefeats(PATH, matrix):
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    src = PATH + 'feats2.csv'
    headers = []
    for i in range(len(matrix[0])):
        headers.append('pixel' + str(i))
    with open(src, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(headers)
        writer.writerows(matrix)


def tooth_recognition(file, labels, images, filename):
    print("Clasificacion")
    data = pd.read_csv(file + "feats2.csv").as_matrix()
    clf = DecisionTreeClassifier()
    xtrain = data[0:20, 0:]
    train_label = labels[:20]
    clf.fit(xtrain, train_label)
    xtest = data[20:, 0:]
    actual_label = labels[20]
    file_Random = random.randint(0, 20)
    d = xtest[file_Random]
    color = str(clf.predict([xtest[file_Random]]))
    print(clf.predict([xtest[file_Random]]))
    plt.imshow(images[file_Random])
    plt.title(filename[file_Random] + " color: " + color)
    plt.axis("off")
    plt.show()
    print("size xtest: ", len(d))

    p = clf.predict(xtest)
    count = 0
    for i in range(0, 20):
        count += 1 if p[i] == actual_label else 0
    print("Accuracy=", (count / 20) * 100)


def main():
    srcdataset = PATH + 'DATASET - copia/'
    datasetresize = PATH + 'ResizeDATASET/'
    directory_segmentation = PATH + 'Segmentation/'
    directory_feats = PATH + 'Features/'
    labels = readlabels(PATH)
    files = readAllImagesPath(PATH)
    # resizeAllImages(files, srcdataset, datasetresize)
    # images = readImages(datasetresize, files)
    # imagesRgb = imagesBGR2RGB(images, directory_segmentation, files)
    # imagesHsv=imagesBGR2HSV(images,directory_segmentation,files)
    # matrix_features = extractFeatures(datasetresize, files)
    # writefeats(directory_feats, matrix_features)
    # tooth_recognition(directory_feats, labels, imagesRgb,files)


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    elapsed_time = end_time - start_time
    print("Tiempo de ejecución del programa: %0.10f segundos." % elapsed_time)
    os.system("PAUSE")
