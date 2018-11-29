import os, cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def show(image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image, interpolation='nearest')
    plt.show()


path = 'C:\\Users\\TRABAJO\\Documents\\Semillero\\TEETH-RECOGNITION-WITH-MACHINE-LEARNING\\DATASET\\'
path1 = 'C:/Users/TRABAJO/Documents/Semillero/TEETH-RECOGNITION-WITH-MACHINE-LEARNING/DATASET'
names = os.listdir(path1)
nameDir = []
subdir_List = []
file_List = []
for dirName, subdirList, fileList in os.walk(path):
    for i in fileList:
        if ('JPG' in i) or ('jpg' in i):
            file_List.append(i)
print("file_List tamaño:", len(file_List))
test = '101_0091.JPG'#file_List[34]
print(test)
testpath = path + test
image = cv2.imread(testpath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
show(image)
for i in [0, 1, 2]:
    colour = image.copy()
    if i != 0: colour[:, :, 0] = 0
    if i != 1: colour[:, :, 1] = 0
    if i != 2: colour[:, :, 2] = 0
    show(colour)


def show_rgb_hist(image):
    colours = ('r', 'g', 'b')
    for i, c in enumerate(colours):
        plt.figure(figsize=(20, 4))
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        if c == 'r': colours = [((i / 256, 0, 0)) for i in range(0, 256)]
        if c == 'g': colours = [((0, i / 256, 0)) for i in range(0, 256)]
        if c == 'b': colours = [((0, 0, i / 256)) for i in range(0, 256)]
        plt.bar(range(0, 256), np.squeeze(histr), color=c, edgecolor=colours, width=1)
        plt.show()


show_rgb_hist(image)
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
show(hsv)
images = []
for i in [0, 1, 2]:
    colour = hsv.copy()
    if i != 0: colour[:, :, 0] = 0
    if i != 1: colour[:, :, 1] = 255
    if i != 2: colour[:, :, 2] = 255
    show(colour)


def show_hsv_hist(image):
    plt.figure(figsize=(20, 3))
    histr = cv2.calcHist([image], [0], None, [180], [0, 180])
    plt.xlim([0, 180])
    colours = [colors.hsv_to_rgb((i / 180, 1, 0.9)) for i in range(0, 180)]
    plt.bar(range(0, 180), np.squeeze(histr), color=colours, edgecolor=colours, width=1)
    plt.title('Hue')
    plt.show()

    plt.figure(figsize=(20, 3))
    histr = cv2.calcHist([image], [1], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, i / 256, 1)) for i in range(0, 256)]
    plt.bar(range(0, 256), np.squeeze(histr), color=colours, edgecolor=colours, width=1)
    plt.title('Saturation')
    plt.show()

    plt.figure(figsize=(20, 3))
    histr = cv2.calcHist([image], [2], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, 1, i / 256)) for i in range(0, 256)]
    plt.bar(range(0, 256), np.squeeze(histr), color=colours, edgecolor=colours, width=1)
    plt.title('Value')
    plt.show()


show_hsv_hist(hsv)
image_blur = cv2.GaussianBlur(image, (7, 7), 0)
show(image_blur)
image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

# 0-10 hue
min_red = np.array([0, 100, 80])
max_red = np.array([10, 256, 256])
image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)


def show_mask(mask):
    plt.figure(figsize=(10, 10))
    plt.title('mask')
    plt.imshow(mask, cmap='gray')


show_mask(image_red1)
min_red2 = np.array([170, 100, 80])
max_red2 = np.array([180, 256, 256])
image_red2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

show_mask(image_red2)
image_red = image_red1 + image_red2
show_mask(image_red)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

# Fill small gaps
image_red_closed = cv2.morphologyEx(image_red, cv2.MORPH_CLOSE, kernel)
show_mask(image_red_closed)

# Remove specks
image_red_closed_then_opened = cv2.morphologyEx(image_red_closed, cv2.MORPH_OPEN, kernel)
plt.figure(figsize=(10,10))
plt.imshow(image_red_closed_then_opened)
plt.title('image_red_closed_then_opened')
show_mask(image_red_closed_then_opened)


def find_biggest_contour(image):
    a, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    if not contour_sizes:
        return None, image

    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


big_contour, mask = find_biggest_contour(image_red_closed_then_opened)
show_mask(mask)


def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    show(img)


overlay_mask(mask, image)
moments = cv2.moments(mask)
centre_of_mass = (
    int(moments['m10'] / moments['m00']),
    int(moments['m01'] / moments['m00'])
)
image_with_com = image.copy()
cv2.circle(image_with_com, centre_of_mass, 10, (0, 255, 0), -1, cv2.CV_AA)
show(image_with_com)
image_with_ellipse = image.copy()

