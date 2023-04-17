import cv2

capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)

while 1:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)  # Flip the camera (if you want)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('RGB', frame)  # Normal
    cv2.imshow('GRAY', gray)  # Or gray scale

    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # Press ESC to close the panel
        break

capture.release()
cv2.destroyAllWindows()
