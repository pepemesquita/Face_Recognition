

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# 0 type model: Will be able to detect the faces within the range of 2 meters from the camera.
# 1 type model: Will be able to detect the faces within the range of 5 meters. Though the default value is 0.
mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
sample_img = cv2.imread('1.jpg')
plt.figure(figsize=[8, 8])

plt.title("Teste");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()

face_detection_results = face_detection.process(sample_img[:,:,::-1])

if face_detection_results.detections:
    
    for face_no, face in enumerate(face_detection_results.detections):
        
        print(f'FACE NUMBER: {face_no+1}')
        print('==============================')
        
        print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')
        
        face_data = face.location_data

        print(f'\nFACE BOUNDING BOX:\n{face_data.relative_bounding_box}')
        
        for i in range(2):

            print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
            print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}')

img_copy = sample_img[:,:,::-1].copy()

if face_detection_results.detections:
    
    for face_no, face in enumerate(face_detection_results.detections):

        mp_drawing.draw_detection(image=img_copy, detection=face, 
                                 keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                              thickness=5,
                                                                              circle_radius=5))
fig = plt.figure(figsize = [8, 8])

plt.title("Resultado");plt.axis('off');plt.imshow(img_copy);plt.show()