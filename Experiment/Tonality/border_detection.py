import os, cv2 as cv, numpy as np, matplotlib.pyplot as plt

PATH = 'C:/Users/TRABAJO/Documents/Semillero/TEETH-RECOGNITION-WITH-MACHINE-LEARNING/DATASET/'


def test():
    source_image = '101_0040.JPG'
    src = PATH + source_image
    print(src)
    image = cv.imread(src)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    imageRGB = cv.resize(image, (500, 400))
    image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image_gray = cv.resize(image_gray, (500, 400))
    img_gauss_blur = cv.GaussianBlur(image_gray, (5, 5), 1)
    ret, imgBinA = cv.threshold(img_gauss_blur, 118, 255, cv.THRESH_BINARY)
    ret, imgBinB = cv.threshold(img_gauss_blur, 127, 255, cv.THRESH_BINARY_INV)
    edges_imgA = cv.Canny(imgBinA, 100, 700)
    edges_imgB = cv.Canny(imgBinB, 100, 200)
    con, mas = find_biggest_contour(edges_imgA)
    big_contour, mask = find_biggest_contour(edges_imgA)
    #imgmask = image+mask
    cv.imshow('Original', imageRGB)
    # cv.imshow('BinA', imgBinA)
    cv.imshow('EdgesA', edges_imgA)
    cv.imshow('mask', mask)
    #cv.imshow('img+mask',imgmask)
    # cv.imshow('EdgesB', edges_imgB)
    cv.waitKey(0)
    cv.destroyAllWindows()

def find_biggest_contour(image):
    a, contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv.drawContours(mask, [biggest_contour], -1, 255, -1)
    mask = cv.cvtColor(mask,cv.COLOR_GRAY2BGR)
    return biggest_contour, mask
def readAllImagesPath(PATH):
    file_List = []
    for dirName, subdirList, fileList in os.walk(PATH):
        for i in fileList:
            if ('JPG' in i) or ('jpg' in i):
                file_List.append(i)
    return file_List

fileList = readAllImagesPath(PATH)

def readAllImages(file_List, PATH):
    images = []
    for i in file_List:
        src = PATH + file_List
        image = cv.imread(src)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (800, 1000))
        images.append(image)
    return images

test()