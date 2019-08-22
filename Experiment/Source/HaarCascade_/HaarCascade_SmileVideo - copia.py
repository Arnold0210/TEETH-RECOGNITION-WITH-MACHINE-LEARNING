import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)
    for (x, y, w, h) in smiles:
        cv.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 255, 0), 2)
      #  roi_gray = gray[y:y + h, x:x + w]
      #  roi_color = frame[y:y + h, x:x + w]
       # cv.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame


#video_capture = cv.VideoCapture(0)
frame = cv.imread('E.JPG')
frame=cv.resize(frame,(400,400))
#while True:
    # Captures video_capture frame by frame
    #_, frame = video_capture.read()

    # To capture image in monochrome
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calls the detect() function
canvas = detect(gray, frame)

    # Displays the result on camera feed
cv.imshow('Video', canvas)
cv.waitKey(0)
    # The control breaks once q key is pressed
#    if cv.waitKey(1) & 0xff == ord('q'):
 #       break

# Release the capture once all the processing is done.
#video_capture.release()
#cv.destroyAllWindows()
