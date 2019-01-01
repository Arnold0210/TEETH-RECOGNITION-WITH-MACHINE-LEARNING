import os,cv2 as cv
from tqdm import tqdm
class loadData:
    def readImages(self,PATH):
        images_List = []
        cantidad = len(os.listdir(PATH))
        bar = tqdm(os.listdir(PATH),ncols=cantidad,unit=' image')
        for i in bar:
            if ('JPG' in i) or ('jpg' in i):
                img = cv.imread(os.path.join(PATH,i))
                images_List.append(img)
            bar.set_description("Leyendo archivo %s" % i)
        return images_List