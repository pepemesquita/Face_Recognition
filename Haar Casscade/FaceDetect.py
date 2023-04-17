import numpy as np
import cv2
faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640) # set Width
cap.set(4, 480) # set Height

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = faceCascade.detectMultiScale(
        RGB,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_RGB = RGB[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eyeCascade.detectMultiScale(roi_color)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('Video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
