import TEETHRECOGNITIONWITHMACHINELEARNING.Experiment.Tonality.ReadImages as rI
import os

if __name__== '__main__':
    PATH_IMAGES =os.path.abspath(os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir),os.pardir),"DATASET"))
    readImages = rI.loadData()
    imagesArray = readImages.readImages(PATH_IMAGES)
    print(imagesArray)