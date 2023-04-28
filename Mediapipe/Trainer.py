import cv2
import numpy as np
import os
import mediapipe as mp

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detect = face_detection.FaceDetection()

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        list_face = face_detect.process(img)
        if not list_face.detections:
            continue

        for face in list_face.detections:
            face_box = face.location_data.relative_bounding_box
            xmin = int(img.shape[1] * face_box.xmin)
            ymin = int(img.shape[0] * face_box.ymin)
            width = int(img.shape[1] * face_box.width)
            height = int(img.shape[0] * face_box.height)
            face_img = img[ymin:ymin+height, xmin:xmin+width]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            faceSamples.append(face_img)
            ids.append(int(os.path.split(imagePath)[-1].split(".")[1]))

    return faceSamples, ids


print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

cv2.destroyAllWindows()
