import cv2  # importar o opencv -> para instalar rode pip install opencv-python
import mediapipe as mp  # para instalar rode pip install mediapipe

webcam = cv2.VideoCapture(0)  # para conectar o python com a nossa webcam.

face_detection = mp.solutions.face_detection  # ativando a solução de reconhecimento de rosto
mp_drawing = mp.solutions.drawing_utils  # ativando a solução de desenho
face_detect = face_detection.FaceDetection()  # criando o item que consegue ler uma imagem e reconhecer os rostos ali dentro

while webcam.isOpened():
    validation, frame = webcam.read()  # lê a imagem da webcam
    if not validation:
        break
    img = frame
    list_face = face_detect.process(
        img)  # usa o reconhecedor para criar uma lista com os rostos reconhecidos

    if list_face.detections:  # caso algum rosto tenha sido reconhecido
        for face in list_face.detections:  # para cada rosto que foi reconhecido
            mp_drawing.draw_detection(img, face)  # desenha o rosto na imagem

    cv2.imshow("Webcam", img)  # mostra a imagem da webcam para a gente
    if cv2.waitKey(
            5) == 27:  # ESC # garante que o código vai ser pausado ao apertar ESC (código 27) e que o código vai esperar 5 milisegundos a cada leitura da webcam
        break
webcam.release()  # encerra a conexão com a webcam
cv2.destroyAllWindows()  # fecha a janela que mostra o que a webcam está fechada