import cv2
import numpy as np

facepath="C:\\Users\\Sayan\\Desktop\\ML Internship\\python_codes\\face_detection\\haarcascade_frontalface_default.xml"
eyepath="C:\\Users\\Sayan\\Desktop\\ML Internship\\python_codes\\face_detection\\haarcascade_eye.xml"

face_casecade=cv2.CascadeClassifier(facepath)
eye_casecade=cv2.CascadeClassifier(eyepath)
#video_path="\\,\\,\\xyz.mp4"
#video=cv2.imread(video_path,0)

cap=cv2.VideoCapture(0)
while True:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_casecade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#rect on face
        roi_gray=gray[y:y+h,x:x+w]#region of face
        roi_color=img[y:y+h,x:x+w]
        eyes=eye_casecade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
    
