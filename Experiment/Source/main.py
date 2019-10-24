#  Copyright (c) 2019. Arnold Julian Herrera Quiñones -  Cristhian Camilo Arce Garcia.
#  All Rights Reserved
#
#  This product is protected by copyright and distributed under
#  licenses restricting copying, distribution, and decompilation.
#  It is forbidden the use partial or global of this algorithm  unless authors written permission.
#

import errno
import os
import sys

import cv2 as cv
from tqdm import tqdm

import Source.Classification as Cl
import Source.FeatureExtraction as fE
import Source.PreProcessingData as pD
import Source.ReadImages as rI


def show(image):
    cv.imshow('Imagen ', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


class MainClass:
    PROJECT_PATH = os.path.join(os.getcwd(), os.path.pardir)
    PATH_IMAGES = os.path.abspath(
        os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET"))
    PATH_Labels = os.path.abspath(
        os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "Labels"))
    PATH_LabelsXML = os.path.abspath(os.path.join(
        os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "Labels"), "LabelsXML"))
    PATH_IMAGES_SNIPPING = os.path.abspath(
        os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET - Recortado"))
    PATH_IMAGES_P = os.path.abspath(
        os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET - P"))
    readimages = None
    preprocessing = None
    featureExtraction = None
    clasification = None

    def __init__(self):
        try:
            if not os.path.exists(self.PROJECT_PATH + 'PreProcessing'):
                os.mkdir(os.path.join(self.PROJECT_PATH, 'PreProcessing'))
                print('Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('PreProcessing Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.PROJECT_PATH + 'FeatureExtraction'):
                os.mkdir(os.path.join(self.PROJECT_PATH, 'FeatureExtraction'))
                print('Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('FeatureExtraction Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.PROJECT_PATH + 'Sampling'):
                os.mkdir(os.path.join(self.PROJECT_PATH, 'Sampling'))
                print('Directory Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Sampling Directory Already Exists.')
            else:
                raise
        try:
            if not os.path.exists(self.PROJECT_PATH + 'Classification'):
                os.mkdir(os.path.join(self.PROJECT_PATH, 'Classification'))
                print('Directory Classification Created')
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Classification Directory Already Exists.')
            else:
                raise
        self.readimages = rI.LoadData(self.PATH_IMAGES)
        self.preProcessing = pD.PreProcessingData(self.PROJECT_PATH, self.PATH_IMAGES)
        self.featureExtraction = fE.FeatureExtraction(self.PROJECT_PATH, self.PATH_IMAGES)
        self.clasification = Cl.Classification(self.PROJECT_PATH)

    def main_run(self):
        # Se declaran las clases para poder utilizar los elementos
        read = self.readimages
        pp = self.preProcessing
        fe = self.featureExtraction

        # Se lee el nombre y la imagen que se encuentre en el PATH del dataset ORIGINAL
        # img, name = read.read_One_Image(self.PATH_IMAGES)
        # Se lee el nombre y la imagen que se encuentre en el PATH del dataset RECORTADO
        img, name = read.read_One_Image(self.PATH_IMAGES_SNIPPING)

        # Se obtiene las dimensiones de la imagen original
        height_ori, width_ori, depth_ori = img.shape
        # print("Image original shape: \n Height:", height_ori, ", Width:", width_ori)

        # Se realiza un ajuste de tamaño para reducir la imagen a unas dimensiones de 600x400
        img_resize = pp.resize_Image(img, name)

        # La imagen reajustada se convierte de BGR a RGB
        img_resize = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)
        # Se convierte la imagen de RGB a HSV
        hsv_image = pp.rgb_2_HSV(img_resize, name)
        # Se saca la imagen en pila de tono de rojos
        stack, name = pp.stackColors(hsv_image, name)
        # Se saca los histogramas por imagen para determinar el rango de color de los dientes
        pp.hsv_hist(hsv_image, name)
        #
        # plt.imshow(img_resize)
        # Blur image slightly
        name, blurimage = pp.blurImage(img_resize, name)
        pp.show_mask(blurimage, name)
        pp.overlay_mask(blurimage, img_resize, name)

        # Se obtiene la rueda cromatica de la imágen
        # pp.getChromatiColor(img_resize,name,fe)

        # img_rgb2ycbcr = pp.rgb_2_YCrCb(img_resize, name)
        img_rgb2hsv = pp.rgb_2_HSV(img_resize, name)

    '''easygui.msgbox("Image original shape: \n Height:" + str(height_ori) + "px, Width:" + str(width_ori) + "px" +
                   "\n Image Resize shape: \n Height:" + str(height_res) + "px, Width:" + str(width_res) + "px",
                   image=os.path.join(os.path.join(os.getcwd(), os.path.pardir),
                                      'PreProcessing/ResizeImages/' + name),
                   title="Image Shape - PreProcessing ")'''

    def savebin(self):
        read = self.readimages
        pp = self.preProcessing
        fe = self.featureExtraction
        images, names = read.read_Images(self.PATH_IMAGES_P)
        for image_point, name_point in zip(images, names):
            img_resize = pp.resize_Image(image_point, name_point)
            img_resize = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)
            img_resize = cv.cvtColor(img_resize, cv.COLOR_RGB2GRAY)
            pp.bin(img_resize, name_point)

    def main_alldataset(self):
        # Se declaran las clases para poder utilizar los elementos
        read = self.readimages
        pp = self.preProcessing
        fe = self.featureExtraction
        cc = self.clasification
        doption = input(
            'Que desea hacer? \n1. Leer imagenes y obtener caracteristicas'
            '\n2. Leer archivo de caracteristicas y entrenar algoritmo\n')
        option = int(doption)
        print(option)
        images, names = read.read_Images(self.PATH_IMAGES_P)
        if option == 1:
            bar = tqdm(os.listdir(PATH), ncols=amount, unit=' image')
            for image_point, name_point in zip(images, names):
                img_resize = pp.resize_Image(image_point, name_point)

                # La imagen reajustada se convierte de BGR a RGB
                img_resize = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)

                # Se convierte la imagen de RGB a HSV
                hsv_image = pp.rgb_2_HSV(img_resize, name_point)

                # Se saca la imagen en pila de tono de rojos
                stack, name_point = pp.stackColors(hsv_image, name_point)

                # Se saca los histogramas por imagen para determinar el rango de color de los dientes
                pp.hsv_hist(hsv_image, name_point)

                # Blur image slightly
                name_point, blurimage = pp.blurImage(img_resize, name_point)
                '''file_ = open(os.path.join(self.PROJECT_PATH, 'Pruebas') + name_point + '.txt', "w")
                for i in blurimage:
                    file_.write(str(i))
                file_.close()'''

                # A partir del rango de color, se saca una máscara donde se ubican los dientes y se procede a
                mask = pp.findBiggestContour(blurimage, name_point)

                channelR, channelG, channelB = cv.split(img_resize)
                red = fe.getFeaturesVector(channelR, mask)
                green = fe.getFeaturesVector(channelG, mask)
                blue = fe.getFeaturesVector(channelB, mask)
                # colores = ["RED", "GREEN", "BLUE"]

                # Se obtiene una mascara de la imágen compuesta por los valores de los canales donde se encontraron los datos de interes.
                imagen = [red, green, blue]
                # Se coloca la ruta del archivo de caracteristicas
                filefeaturespath = os.path.join(os.path.join(self.PROJECT_PATH, 'FeatureExtraction'), 'features.csv')
                # Se escribe en un archivo las caracteristicas del dataset
                fe.getFeatures(imagen, filefeaturespath, name_point)
                fileLabels = open(os.path.join(self.PATH_Labels, 'Labels.csv'))
                pp.show_mask(blurimage, name_point)
                pp.overlay_mask(blurimage, img_resize, name_point)
        elif option == 2:
            filefeaturespath = os.path.join(os.path.join(self.PROJECT_PATH, 'FeatureExtraction'), 'features3.csv')
            names, features = cc.readfeatures(filefeaturespath)
            labels = cc.readLabels(self.PATH_Labels)
            # print(feattures)
            features_images = features.values
            vals_to_replace = {'a1': '0', 'a2': '1', 'a3': '2', 'a35': '3', 'a4': '4'}
            matrix_confusion, report_clasification = cc.validacionCruzada(self.PATH_IMAGES_P, features, labels,
                                                                          vals_to_replace)
            import numpy, matplotlib
            numpy.set_printoptions(precision=2)
            matplotlib.pyplot.show()


if __name__ == '__main__':
    tesis: MainClass = MainClass()
    # tesis.main_run()
    tesis.main_alldataset()
    # tesis.savebin()
    print('Se ha finalizado la ejecución del experimento')
    sys.exit(0)
