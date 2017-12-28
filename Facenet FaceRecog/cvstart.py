# Bhatnagar, Nishank
# 1001-397-7098
# 2017-11-26
# Assignment_06-01
# Description: Used OpenCV to use Camera Feed as the source and fed the Frame or Image over to FaceNet
# FaceNet is the implementation of the CNN to recognize faces, 
# The model is pre trained on given Class photo data 

# import the necessary packages
from imutils import face_utils
import time
import numpy as np
from PIL import Image
import imutils
import dlib
import cv2
import src.face 

frame_interval = 1  # Number of frames after which to run face detection
fps_display_interval = 5  # seconds
frame_rate = 0
frame_count = 1

# Starting video camera feed
# cameraCapture = cv2.VideoCapture("class_video_data.mp4")
cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyCameraWindow')


def add_overlay(frame, faces):
    detected_images = []
    actual_images = []

    if faces is not None:
        for face in faces:
            (x, y, w, h) = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (x, y), (w, h),
                          (255, 0, 0), 2)
            if face.name is not None:
                cv2.putText(frame, "Det", (370, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 
                    thickness=1, lineType=2)

                cv2.putText(frame, "Act", (450, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 
                    thickness=1, lineType=2)

                cv2.putText(frame, face.name, (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                            thickness=2, lineType=2)

                file_name = "_".join(face.name.split(" "))
                file_path = "Face_data_compare/" + file_name + "/" + file_name + ".png"
                im = cv2.imread(file_path)
                im = imutils.resize(im, width=50)

                actual_images.append(im)

                cropped_image = frame[y: h, x : w]
                cropped_image = imutils.resize(cropped_image, width=50)

                detected_images.append(cropped_image)

    return (detected_images , actual_images)

start_time = time.time()


face_recogintion = src.face.Recognition()


success, img = cameraCapture.read()
while success and cv2.waitKey(1) != 27:

    image = imutils.resize(img, width=500)
    faces = None

    if (frame_count % frame_interval) == 0:
            faces = face_recogintion.identify(image)


    detected_imgs , actual_imgs = add_overlay(image,faces)

    if len(detected_imgs) != 0:
        y = 20
        for i in range(len(detected_imgs)):
            image[y: y + detected_imgs[i].shape[0], 370: 370 + detected_imgs[i].shape[1]] = detected_imgs[i]
            image[y: y + actual_imgs[i].shape[0], 450: 450 + actual_imgs[i].shape[1]] = actual_imgs[i]
            y += 50


    frame_count += 1
    cv2.imshow('MyCameraWindow', image)
    success, img = cameraCapture.read()



cv2.destroyWindow('MyCameraWindow')
