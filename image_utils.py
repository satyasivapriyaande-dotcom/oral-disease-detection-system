import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def is_selfie(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(w//5, h//5)
    )
    return len(faces) > 0

def preprocess(img_rgb):
    img = cv2.resize(img_rgb, (160,160))
    img = img / 255.0
    return img.reshape(1,160,160,3)

