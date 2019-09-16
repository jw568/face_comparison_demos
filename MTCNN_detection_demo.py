#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:10:09 2019

@author: jasonwang
"""

import sys
sys.path.append('./mtcnn-master')
from mtcnn.mtcnn import MTCNN
from Configuration import *
import cv2
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

cam = cv2.VideoCapture(0)
cv2.namedWindow("Sentry Face Detection")
detector = MTCNN()
while True:
    ret, frame = cam.read()
    if not ret:
        break
    k = cv2.waitKey(1)
    result = detector.detect_faces(frame)
    frame_height, frame_width = frame.shape[:2]
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
            cv2.rectangle(frame,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (255, 0, 0),
                      2)
            cv2.circle(frame,(keypoints['left_eye']), 2, (255, 0, 0), 2)
            cv2.circle(frame,(keypoints['right_eye']), 2, (255, 0, 0), 2)
            cv2.circle(frame,(keypoints['nose']), 2, (255, 0, 0), 2)
            cv2.circle(frame,(keypoints['mouth_left']), 2, (255, 0, 0), 2)
            cv2.circle(frame,(keypoints['mouth_right']), 2, (255, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Sentry AI"
    cv2.putText(frame, text, (15,40), font, 1, (255,0,0), 2)
    text = "Number of Faces Detected: {}".format(len(result))
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    cv2.putText(frame, text, (int((frame_width - textsize[0]) / 2), frame_height - 30),
                font, 1, (255, 0, 0), 2)
    cv2.imshow("Sentry Face Detection", frame)
    
    if k%256 == 32:
        cam.release()
        cv2.destroyWindow("Sentry Face Detection")
        break

    
            