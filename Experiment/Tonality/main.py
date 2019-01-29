import TEETHRECOGNITIONWITHMACHINELEARNING.Experiment.Tonality.ReadImages as rI
import TEETHRECOGNITIONWITHMACHINELEARNING.Experiment.Tonality.PreProcessingData as pd
import os, cv2 as cv


class MainClass:
    preprocessing = pd.PreProcessingData()
    read_images = rI.loadData()

    def main_run(self, path_images):
        # imagesArray = read_images.read_images(path_images)
        image = self.read_images.readOneImage(path_images)
        pre = self.preprocessing.BlackGreyScale(image)
        show(pre)


def show(image):
    cv.imshow('Imagen ', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    PATH_IMAGES = os.path.abspath(
        os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), "DATASET"))
    tesis = MainClass()
    tesis.main_run(PATH_IMAGES)
