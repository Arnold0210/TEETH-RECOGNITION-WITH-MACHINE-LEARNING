import TEETHRECOGNITIONWITHMACHINELEARNING.Experiment.Tonality.ReadImages as ri
import cv2 as cv


class PreProcessingData:
    def BlackGreyScale(self, images):
        image_bn = cv.cvtColor(images, cv.COLOR_BGR2GRAY)

        return image_bn

    def BorderDetection(self, images):
        pass

    def ImageSegmentation(self, images):
        pass

    def ImageBinarization(self, images):
        pass
