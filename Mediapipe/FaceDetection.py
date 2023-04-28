import cv2  # importar o opencv -> para instalar rode pip install opencv-python
import mediapipe as mp  # para instalar rode pip install mediapipe

capture = cv2.VideoCapture(0)  # para conectar o python com a nossa webcam.

face_detection = mp.solutions.face_detection  # ativando a solução de reconhecimento de rosto
mp_drawing = mp.solutions.drawing_utils  # ativando a solução de desenho
face_detect = face_detection.FaceDetection()  # criando o item que consegue ler uma imagem e reconhecer os rostos ali dentro
face_id = input('\n Insira o id numérico do usuário e pressione enter.')
print('\n Inicializando... Aguarde e olhe para a camera.')
contador = 0

while True:
    ret, img = capture.read()  # lê a imagem da webcam

    if not ret:
        break

    img = cv2.flip(img, 1)
    list_face = face_detect.process(img)  # usa o reconhecedor para criar uma lista com os rostos reconhecidos
    print("face:", list_face.detections)

    if list_face.detections:  # caso algum rosto tenha sido reconhecido
        for face in list_face.detections:  # para cada rosto que foi reconhecido
            mp_drawing.draw_detection(img, face)  # desenha o rosto na imagem
            face_box = face.location_data.relative_bounding_box
            xmin = face_box.xmin
            ymin = face_box.ymin
            width = face_box.width
            height = face_box.height
            contador += 1
            face_img = img[int(img.shape[0] * ymin):int(img.shape[0] * (ymin + height)),
                       int(img.shape[1] * xmin):int(img.shape[1] * (xmin + width))]
            cv2.imwrite("dataset/User." + str(face_id) + '_' + str(contador) + ".jpg", face_img)

    cv2.imshow('Webcam', img)  # mostra a imagem da webcam para a gente
    if cv2.waitKey(5) == 27:  # ESC garante que o código vai ser pausado ao apertar ESC (código 27) e que o código vai esperar 5 milisegundos a cada leitura da webcam
        break

    elif contador >= 30:    # Tira 30 fotos do exemplo e para o video
        break

capture.release()  # encerra a conexão com a webcam
cv2.destroyAllWindows()  # fecha a janela que mostra o que a webcam está fechada