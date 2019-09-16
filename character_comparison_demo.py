#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:36:03 2019

@author: jasonwang
"""

import sys

sys.path.append('./face_comparison')
sys.path.append('./face_morphing')
sys.path.append('./FaceSwap')
sys.path.append('./yoloface')
sys.path.append('./comparefaces')

from comparefaces import compare
from face_morphing import face_morph
from FaceSwap import new_main
from Configuration import *
from yoloface import utils
from yoloface import GoT_yoloface
import numpy as np
import os
import cv2
import argparse

net = cv2.dnn.readNetFromDarknet('./yoloface/cfg/yolov3-face.cfg',
                                 './yoloface/model-weights/yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_WIDTH = 416
IMG_HEIGHT = 416
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


if __name__ == "__main__":

    font = cv2.FONT_HERSHEY_SIMPLEX

    # User selects theme
    parser = argparse.ArgumentParser()
    parser.add_argument("theme", help='Which theme?')
    parser.add_argument('-o', "--email", help='Email?')
    args = parser.parse_args()
    folder = './' + args.theme + '/'
    if not args.email == None:
        email = str(args.email)
    else:
        email = " "

    # Captures user's face and draw's bounding box and adds text
    img1 = GoT_yoloface._main()
    my_dict = {}

    cap = cv2.VideoCapture(img1)
    count = 0
    boxed_frame = cv2.imread(img1)
    boxed_height, boxed_width = boxed_frame.shape[:2]

    blob = cv2.dnn.blobFromImage(boxed_frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(utils.get_outputs_names(net))
    faces = utils.post_process(boxed_frame, outs, CONF_THRESHOLD, NMS_THRESHOLD, cap, count)
    count += len(faces)
    if len(faces) == 1:
        text = "Let's compare your face!"
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        cv2.putText(boxed_frame, 'Sentry sees a face.', (10, 30),
                    font, 1, (255, 0, 0), 2)
        cv2.putText(boxed_frame, text, (int((boxed_width - textsize[0]) / 2), boxed_height - 40),
                    font, 1, (255, 0, 0), 2)
    boxed_frame_1 = cv2.resize(boxed_frame, (640, 480))
    boxed_frame_1 = image_resize(boxed_frame, width=384)
    boxed_height, boxed_width = boxed_frame_1.shape[:2]

    files = []
    for f in os.listdir(folder):
        if f.endswith(".jpg"):
            files.append(folder + f)

    files.insert(0, img1)
    print(files)
    similarities = compare.main(files)

    # adds all characters of theme to a dictionary
    for f in os.listdir(folder):
        if f.endswith(".jpg"):
            my_dict[folder + f] = None

    count = 0
    for f in os.listdir(folder):
        if f.endswith(".jpg"):
            my_dict[folder + f] = similarities[count]
            count = count + 1

        #    for file in os.listdir(folder):
        #        if (file.endswith(".jpg")):
        filename = folder + str(f)
        #            print(filename)
        #            similarity = 'Face not detected.'
        #            similarity = face_comparison.compare(img1, filename)
        #            my_dict[file] = similarity
        character = cv2.imread(filename)

        if len(faces) == 1:
            count_char = 1
            print(my_dict)
            for key in my_dict:
                print(key)
                char = cv2.imread(key)
                # char = cv2.resize(char, (0,0), None, 0.3, 0.3)
                char = image_resize(char, width=384)
                char_height, char_width = char.shape[:2]
                if boxed_height > char_height:
                    char = cv2.copyMakeBorder(char, top=int((boxed_height - char_height) / 2),
                                              bottom=int((boxed_height - char_height) / 2), left=0, right=0,
                                              borderType=cv2.BORDER_CONSTANT)
                # cv2.imwrite('testing.jpg', char)
                if my_dict[key] == None:
                    text = 'Processing'
                else:
                    # adds bounding box
                    char_blob = cv2.dnn.blobFromImage(char, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                                      [0, 0, 0], 1, crop=False)
                    net.setInput(char_blob)
                    outs = net.forward(utils.get_outputs_names(net))
                    face = utils.post_process(char, outs, CONF_THRESHOLD, NMS_THRESHOLD, cap, 1)
                    # sets text to similarity
                    text = str(round((1.65 - my_dict[key]) * 60.6, 3)) + '%'

                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                cv2.putText(char, text, (int((char_width - textsize[0]) / 2), char_height - 15), font, 1, (0, 0, 255),
                            2)

                if count_char == 1:
                    sbs1 = char
                if count_char == 2 or count_char == 3:
                    sbs1 = np.concatenate((sbs1, char), axis=1)
                if count_char == 4:
                    sbs2 = char
                if count_char == 5:
                    sbs2 = np.concatenate((sbs2, boxed_frame_1), axis=1)
                    sbs2 = np.concatenate((sbs2, char), axis=1)
                if count_char == 6:
                    sbs3 = char
                if count_char == 7 or count_char == 8:
                    sbs3 = np.concatenate((sbs3, char), axis=1)
                count_char = count_char + 1

            three_by_three = np.concatenate((sbs1, sbs2), axis=0)
            three_by_three = np.concatenate((three_by_three, sbs3), axis=0)

            wind_name = 'Sentry Face Detection'
            cv2.namedWindow(wind_name)
            cv2.imshow(wind_name, three_by_three)
            cv2.waitKey(500)

    cv2.imwrite('./entertainment_results/' + email + '.jpg', three_by_three)
    # output the character that they are most similar to
    minimum = min(my_dict, key=my_dict.get)
    # original_out = cv2.resize(boxed_frame, (0,0), None, .5, .5)
    original_out = image_resize(boxed_frame, height=360)
    celeb_char_out = cv2.imread(str(minimum))
    # celeb_char_out = cv2.resize(celeb_char_out, (0,0), None, .5, .5)
    celeb_char_out = image_resize(celeb_char_out, height=360)
    sbs = np.concatenate((original_out, celeb_char_out), axis=1)
    sbs_h, sbs_w = sbs.shape[:2]
    text = 'Your face looks the most like {} with {}% similarity.'.format(minimum, str(
        round((1.65 - my_dict[minimum]) * 60.6, 3)))
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    cv2.putText(sbs, text, (int((sbs_w - textsize[0]) / 2), sbs_h - 40), font, 1, (0, 0, 255), 2)
    cv2.imwrite('./entertainment_results/' + email + '_sbs.jpg', sbs)
    wind_name = 'Sentry Face Detection'
    cv2.namedWindow(wind_name)
    cv2.imshow(wind_name, sbs)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # face morph
    face_morph.morph('face', img1, str(minimum), email)

    # face swap
    new_main.start(img1, str(minimum), out='./entertainment_results/' + email + '_faceswap_results.jpg', warp_2d=True, correct_color=True)
    cv2.namedWindow('face_swap_results')
    out = cv2.imread('./entertainment_results/' + email + '_faceswap_results.jpg')
    cv2.imshow('face_swap_results', out)
    cv2.waitKey(5000)
    cv2.destroyWindow('face_swap_results')
    cv2.waitKey(1)
