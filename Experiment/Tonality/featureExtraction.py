import os, cv2 as cv, numpy as np, matplotlib.pyplot as plt, sys, glob
import extcolors as ext

PATH = 'C:/Users/TRABAJO/Documents/Semillero/TEETH-RECOGNITION-WITH-MACHINE-LEARNING/DATASET/'
PATH_SAVE = 'C:/Users/TRABAJO/Documents/Semillero/TEETH-RECOGNITION-WITH-MACHINE-LEARNING/DATASETRESIZE/'


def show(image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image, interpolation='nearest')
    plt.show()


def readAllImagesPath(PATH):
    file_List = []
    for dirName, subdirList, fileList in os.walk(PATH):
        for i in fileList:
            if ('JPG' in i) or ('jpg' in i):
                file_List.append(i)
    print("Termino de añadirpath")
    return file_List


def readAllImages(file_List, PATH):
    images = []
    for i in file_List:
        src = PATH + i
        image = cv.imread(src, 1)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        height, width, channels = image.shape
        reheight, rewidth = int(height / 9), int(width / 9)  # Resize inage to 1/9
        image = cv.resize(image, (rewidth, reheight))
        cv.imwrite(os.path.join(PATH_SAVE, i), image)
        images.append(image)
    print("Termino Carga")
    return images


def imagesBGR2RGB(images):
    for i in images:
        i = cv.cvtColor(i, cv.COLOR_BGR2RGB)
    print("Termino convertir")
    return images


def imagesRGB2HSV(imagesRGB):
    for i in imagesRGB:
        i = cv.cvtColor(i, cv.COLOR_RGB2HSV)
    return imagesRGB


def imagesGaussianBlur(imagesHSV):
    for i in imagesHSV:
        i = cv.GaussianBlur(i, (5, 5), 0)
    return imagesHSV


def extractFeatures(PATH,fileList):
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
        print('postextract')
        colors_image = []
        for colors in extract:
            print('f')
            for l in colors[0]:
                print('e')
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


def main():
    matrixData = []
    filelist = readAllImagesPath(PATH_SAVE)
    # images = readAllImages(filelist, PATH)
    # imagesRGB = imagesBGR2RGB(images)
    # imagesHSV = imagesRGB2HSV(imagesRGB)
    # imagesGaussBlur = imagesGaussianBlur(imagesHSV)
    matrixData = extractFeatures(PATH_SAVE,filelist)
    standarMatriz = standar_matrix(matrixData)


main()
