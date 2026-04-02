import cv2
import numpy as np
import csv
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load Haarcascade from OpenCV built-in folder
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

names = ['None', 'Kishore']

cam = cv2.VideoCapture(0)

attendance_list = []

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if confidence < 100:
            name = names[id]

            if name not in attendance_list:
                attendance_list.append(name)

                with open('attendance.csv','a',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, datetime.now()])

        else:
            name = "Unknown"

        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)

    cv2.imshow('Face Recognition', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()