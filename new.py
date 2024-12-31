import cv2
from ultralytics import YOLO
import time
from playsound import playsound

cap = cv2.VideoCapture(1)
model = YOLO("yolo11n.pt")

lastplayed = 0
delay = 4

poorImg = cv2.imread(r"D:\VISUAL STUDIO\Python\belajar\opencv_objtracking\assets\poor.jpg", -1)
poorImg = cv2.resize(poorImg, (800, 449))

while True:
    ret, frame = cap.read()
    result = model.predict(source=frame)
    cellphoneDetect = False

    for x in result:
        for i in x.boxes:
            cls = int(i.cls[0])
            label = model.names[cls]
            if label == "cell phone":
                cellphoneDetect = True
        detected_objects = x.plot()
        cv2.imshow("Simple Object Detection", detected_objects)


    currentTime = time.time()
    if cellphoneDetect and (currentTime - lastplayed >= delay):
        playsound(r"D:\VISUAL STUDIO\Python\belajar\opencv_objtracking\assets\future.mp3")
        lastplayed = currentTime
        cv2.imshow("YOUR FUTURE IF YOU DONT LOCK IN RN", poorImg)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()