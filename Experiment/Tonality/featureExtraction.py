import os, cv2 as cv, numpy as np, matplotlib.pyplot as plt, sys, glob, random
import extcolors as ext, sklearn.decomposition.pca as pca
from time import time
import imageio
from tqdm import tqdm
from progress.bar import Bar, ChargingBar
from scipy import ndimage as ndi

PATH = 'C:/Users/TRABAJO/Documents/Semillero/TEETH-RECOGNITION-WITH-MACHINE-LEARNING/'


def show(image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image, interpolation='nearest')
    plt.show()


def readAllImagesPath(PATH):
    start_time = time()
    file_List = []
    cantidad = len(glob.glob(PATH + "*"))
    procesing = Bar('Leyendo archivos:', max=cantidad)
    for dirName, subdirList, fileList in os.walk(PATH):
        for i in fileList:
            if ('JPG' in i) or ('jpg' in i):
                file_List.append(i)
            procesing.next()
    procesing.finish()
    end_time = time()
    elapsed_time = end_time - start_time
    print("Tiempo de cargue de ruta de atchivos: %0.10f segundos." % elapsed_time)
    return file_List


def resizeAllImages(file_List, PATHSRC, PATHRESIZE):
    start_time = time()
    if not os.path.exists(PATHRESIZE):
        os.mkdir(PATHRESIZE)
    for i in tqdm(file_List):
        src = PATHSRC + i
        image = cv.imread(src, 1)
        image = cv.resize(image, (600, 400))
        if not os.path.exists(PATHRESIZE + i):
            cv.imwrite(os.path.join(PATHRESIZE, i), image)
    end_time = time()
    elapsed_time = end_time - start_time
    print("Tiempo de re-escalar imagenes: %0.10f segundos." % elapsed_time)


def readImagesRGB(PATH, fileList):
    start_time = time()
    images = []
    for i in tqdm(fileList):
        src = PATH + i
        image = cv.imread(src, 1)
        images.append(image)
    end_time = time()
    elapsed_time = end_time - start_time
    print("Tiempo cargando imagenes a un vector: %0.10f segundos." % elapsed_time)
    return images


def readImages(PATH, fileList):
    start_time = time()
    images = []
    for i in tqdm(fileList):
        src = PATH + i
        image = cv.imread(src)
        images.append(image)
    end_time = time()
    elapsed_time = end_time - start_time
    print("Tiempo cargando imagenes a un vector: %0.10f segundos." % elapsed_time)
    return images


def imagesRGB2HSV(imagesRGB, directory, files):
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


def imagesBGR2HSV(images, directory, files):
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


def imagesBGR2RGB(images, directory, files):
    if not os.path.exists(directory):
        os.mkdir(directory)
    folder = directory + 'RGB/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    RGB_ = Bar('Convirtiendo BGR a RGB:', max=len(files) * 4)
    for i, file in zip(images, files):
        i = cv.cvtColor(i, cv.COLOR_BGR2RGB)
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


def imagesGaussianBlurHSV(imagesHSV, directory):
    for i in imagesHSV:
        i = cv.GaussianBlur(i, (5, 5), 0)
    return imagesHSV


def imagesGaussianBlurRGB(imagesHSV, directory):
    for i in imagesHSV:
        i = cv.GaussianBlur(i, (5, 5), 0)
    return imagesHSV


def extractFeatures(PATH, fileList):
    matrix_data = []
    a = len(fileList)
    cont = 0
    for i in fileList:
        src = PATH + i
        print('path:+ ', PATH + i)
        for i in range(int((cont * 100) / a)):
            print('\u2588', end="")
        print(" ", int((cont * 100) / a), "%")
        extract, countPixel = ext.extract(src)
        colors_image = []
        for colors in extract:
            for l in colors[0]:
                colors_image.append((l))
        matrix_data.append(colors_image)
        cont = cont + 1
        os.system('cls')
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


def ImageSegmentation(images, directory,files):
    start_time = time()
    src = directory + 'Segmentation'
    if not os.path.exists(src):
        os.mkdir(src)
    segmentation = Bar('Segmentando imagenes:', max=len(images))
    for image,file in zip(images,files):
        ycrcbmin = np.array((100, 133, 77))
        ycrcbmax = np.array((255, 173, 127))
        skin_ycrcb = cv.inRange(image, ycrcbmin, ycrcbmax)
        kernel = np.ones((5, 5), np.uint8)
        img_erode = cv.erode(skin_ycrcb, kernel, iterations=1)
        holesimg = ndi.binary_fill_holes(img_erode).astype(np.int)
        imageio.imwrite(os.path.join(src, file), holesimg)
        segmentation.next()
    segmentation.finish()
    end_time = time()
    elapsed_time = end_time - start_time
    print("Tiempo de Segmentación: %0.10f segundos." % elapsed_time)


def main():
    srcdataset = PATH + 'DATASET/'
    datasetresize = PATH + 'ResizeDATASET/'
    directory_segmentation = PATH + 'Segmentation/'
    #labels = readlabels(srcdataset)
    files = readAllImagesPath(srcdataset)
    #resizeAllImages(files, srcdataset, datasetresize)
    images = readImages(datasetresize, files)
    #imagesRGB = readImagesRGB(datasetresize, files)
    #imagesRGB2HSV(imagesRGB, directory_segmentation, files)
    #imagesBGR2HSV(images, directory_segmentation, files)
    #imagesBGR2RGB(images, directory_segmentation, files)
    imageycrcb = imagesBGR2YCR_CB(images, directory_segmentation, files)
    ImageSegmentation(imageycrcb, directory_segmentation,files)


start_time = time()
main()
end_time = time()
elapsed_time = end_time - start_time
print("Tiempo de ejecución del programa: %0.10f segundos." % elapsed_time)
os.system("PAUSE")
