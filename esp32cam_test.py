import cv2

url = 'rtsp://admin:admin@192.168.43.229:8554/live'
cap = cv2.VideoCapture(url)

while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
