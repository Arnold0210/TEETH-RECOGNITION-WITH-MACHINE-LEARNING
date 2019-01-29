import os, cv2 as cv
from tqdm import tqdm


class loadData:
    def readImages(self, PATH):
        images = []
        amount = len(os.listdir(PATH))
        bar = tqdm(os.listdir(PATH), ncols=amount, unit=' image')
        for i in bar:
            if ('JPG' in i) or ('jpg' in i):
                img = cv.imread(os.path.join(PATH, i))
                images.append(img)
            bar.set_description("Leyendo archivo %s" % i)
        return images

    def readOneImage(self, PATH):
        imageString = os.path.join(PATH, "100_0055.JPG")
        image = cv.imread(imageString)
        image = cv.resize(image, (600, 400))
        return image
