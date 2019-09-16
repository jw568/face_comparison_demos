#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:49:42 2019

@author: jasonwang
"""

import sys
sys.path.append('./yoloface')
sys.path.append('./comparefaces')

import time
from comparefaces import compare
from yoloface import utils
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import os
import cv2
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from pathlib import Path
import dlib

net = cv2.dnn.readNetFromDarknet('./yoloface/cfg/yolov3-face.cfg', './yoloface/model-weights/yolov3-wider_16000.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
IMG_WIDTH = 416
IMG_HEIGHT = 416
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
pool = ThreadPool(4)

font = cv2.FONT_HERSHEY_SIMPLEX
wind_name = 'Sentry Face Detection'
count = 0
themes = ['./GoT/','./Avengers/']
theme_num = np.random.randint(0, 2)
theme = themes[theme_num]
celeb_num = np.random.randint(0, 7)
celeb = theme + os.listdir(theme)[celeb_num]
celeb_count = 1

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5"
modhash = 'fbe63257a054c1c5466cfd7bf14646d6'

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def Age_Gender(img):
    weight_file = get_file("weights.28-3.73.hdf5", pretrained_model, cache_subdir="pretrained_models",
                           file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))
    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=16, k=8)()
    model.load_weights(weight_file)

    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_h, img_w, _ = np.shape(input_img)

    # detect faces using dlib detector
    detected = detector(input_img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(predicted_ages[i]),
                                    " Male" if predicted_genders[i][0] < 0.5 else " Female")
            print(d.left())
            draw_label(img, (d.left()+50, d.top()+250), label)

    return img

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

while True:
    print("hello")
    cv2.namedWindow(wind_name)
    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)
    has_frame, frame = cap.read()
    if not has_frame:
        break
    
    key = cv2.waitKey(5000)
    if key % 256 == 32:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break

    count = 0
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(utils.get_outputs_names(net))
    # Remove the bounding boxes with low confidence
    faces = utils.post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD, cap, count)
    count += len(faces)
    
    if len(faces) == 1:
        cv2.imwrite("./split_four/split_4_original.jpg", frame)
        print('written!')
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        break

original = cv2.imread("./split_four/split_4_original.jpg")
original = image_resize(original, width = 640)
cv2.imwrite('original.jpg',original)
orig_height, orig_width = original.shape[:2]
if (480 > orig_height):
    original = cv2.copyMakeBorder(original, top=int((480 - orig_height) / 2),
                                   bottom=int((480 - orig_height) / 2), left=0, right=0,
                                   borderType=cv2.BORDER_CONSTANT)
elif (640 > orig_width):
    original = cv2.copyMakeBorder(original, top=0, bottom=0, left=int((640 - orig_width) / 2),
                                   right=int((640 - orig_width) / 2), borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('original_after_border.jpg', original)
orig_height, orig_width = original.shape[:2]
print(orig_height, orig_width)


def two_screen(my_dict):
    cv2.namedWindow('Side By Side')
    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)
    if my_dict:
        minimum = min(my_dict, key=my_dict.get)
        while True:
            has_frame, frame = cap.read()
            if not has_frame:
                break

            key= cv2.waitKey(1)
            if key % 256 == 87 or key % 256 == 119:
                cap.release()
                cv2.destroyWindow('Side By Side')
                cv2.waitKey(1)
                four_screen()

            count = 0
            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                         [0, 0, 0], 1, crop=False)
            # Sets the input to the network
            net.setInput(blob)
            # Runs the forward pass to get output of the output layers
            outs = net.forward(utils.get_outputs_names(net))
            # Remove the bounding boxes with low confidence
            faces = utils.post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD, cap, count)
            count += len(faces)

            cv2.imwrite('./two_screen_current_img.jpg', frame)

            if len(faces) == 1:
                similarities = compare.main(['./two_screen_current_img.jpg', minimum])

                out_frame = image_resize(frame, height = 360)
                out_height, out_width = out_frame.shape[:2]

                shown = cv2.imread(minimum)

                shown = image_resize(shown, height = 360)
                shown_height, shown_width = shown.shape[:2]

                if (out_width > shown_width):
                    shown = cv2.copyMakeBorder(shown, top=int((out_height - shown_height) / 2),
                                                  bottom=int((out_height - shown_height) / 2), left=0, right=0,
                                                  borderType=cv2.BORDER_CONSTANT)
                shown_height, shown_width = shown.shape[:2]

                if not len(similarities) == 0:
                    text = str(round((1.6 - similarities[0]) / 1.6, 3) * 100) + "%"
                else:
                    text = "Face not detected."
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                cv2.putText(shown, text, (int((shown_width - textsize[0]) / 2), shown_height - 20),
                            font, 1, (255, 0, 0), 2)

                sbs = np.concatenate((out_frame, shown), axis=1)
                cv2.imshow('Side By Side', sbs)
                cv2.waitKey(1)


def four_screen():
    cv2.namedWindow('Sentry Face Detection')
    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)
    cycle_counter = 0
    celeb_count = 1
    themes = ['./GoT/', './Bigbang/', './Avengers/']
    theme_num = np.random.randint(0, 3)
    theme = themes[theme_num]
    celeb_num = np.random.randint(0, 8)
    celeb = theme + os.listdir(theme)[celeb_num]
    similarities = [0,0,0]
    my_dict = {}
    orig_age_gender = False
    sbs2_age_gender = False
    while True:
        print('cycle_counter: {}'.format(cycle_counter))
        start_time = time.time()
        has_frame, frame = cap.read()
        if not has_frame:
            break

        frame_height, frame_width = frame.shape[:2]

        print('made it here')
        key = cv2.waitKey(5000)
        if key % 256 == 32:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
        if key % 256 == 113 or key % 256 == 81:
            print('entered two_screen')
            cap.release()
            cv2.destroyWindow('Sentry Face Detection')
            cv2.waitKey(1)
            two_screen(my_dict)

        count = 0
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = net.forward(utils.get_outputs_names(net))
        # Remove the bounding boxes with low confidence
        faces = utils.post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD, cap, count)
        count += len(faces)

        if len(faces) == 1:

            cv2.imwrite('./split_four/current_frame.jpg', frame)

            #outputs live frame
            text = "Let's compare your face!"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            cv2.putText(frame, 'Sentry sees a face.', (10, 30),
                font, 1, (255,0,0) , 2)
            cv2.putText(frame, text, (int((frame_width-textsize[0])/2), frame_height - 40),
                font, 1, (255,0,0), 2)
            #out_frame = cv2.resize(frame, (0,0), None, 0.35, 0.35)
            out_frame = image_resize(frame, width = orig_width)
            out_height, out_width = out_frame.shape[:2]
            print('original width, height: {}{}'.format(orig_width, orig_height))
            print('out width, height {}{}'.format(out_width, out_height))
            if (orig_height > out_height):
                out_frame = cv2.copyMakeBorder(out_frame, top = int((orig_height - out_height) / 2), bottom = int((orig_height - out_height) / 2), left = 0, right = 0, borderType=cv2.BORDER_CONSTANT)
            elif(orig_width > out_width):
                out_frame = cv2.copyMakeBorder(out_frame, top = 0, bottom = 0, left = int((orig_width - out_width) / 2), right = int((orig_width - out_width) / 2), borderType=cv2.BORDER_CONSTANT)
            cv2.imwrite('out_frame_after_border.jpg', out_frame)

            #change celeb every 7 cycles
            if (celeb_count % 500 == 0):
                theme_num = np.random.randint(0, 3)
                theme = themes[theme_num]
                celeb_num = np.random.randint(0, 8)
                celeb = theme + os.listdir(theme)[celeb_num]
                sbs2_age_gender = False
            celeb_count = celeb_count + 1
            print(celeb)

            #outputs black frame if linkedin is not detected, otherwise output linkedin photo
            if (os.path.isfile('./split_four/linkedin.jpeg')):
                third_frame = './split_four/linkedin.jpeg'
                third_img = cv2.imread(third_frame)
                third_height, third_width = third_img.shape[:2]
                print('detected linkedin')
                similarities = compare.main(['./split_four/current_frame.jpg', "./split_four/split_4_original.jpg", './split_four/linkedin.jpeg', celeb])
                my_dict["./split_four/split_4_original.jpg"] = similarities[0]
                my_dict['./split_four/linkedin.jpeg'] = similarities[1]
                my_dict[celeb] = similarities[2]
                text = str(round((1.6 - similarities[1])/1.6,3)*100) + "%"
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                third_img = cv2.putText(third_img, text, (30, third_height - 40), font, 1, (255,0,0), 2) #linkedin similarity writing
                third_img = image_resize(third_img, height = orig_height)
                cv2.imwrite('third_img.jpg', third_img)
                third_height, third_width = third_img.shape[:2]
                if (orig_height > third_height):
                    third_img = cv2.copyMakeBorder(third_img, top = int((orig_height - third_height) / 2), bottom = int((orig_height - third_height) / 2), left = 0, right = 0, borderType=cv2.BORDER_CONSTANT)
                elif(orig_width > third_width):
                    third_img = cv2.copyMakeBorder(third_img, top = 0, bottom = 0, left = int((orig_width - third_width) / 2), right = int((orig_width - third_width) / 2), borderType=cv2.BORDER_CONSTANT)
                cv2.imwrite('third_img_after_border.jpg', third_img) #bottom left
            else:
                third_frame = './split_four/black_screen.jpg'
                third_img = cv2.imread(third_frame)
                text = 'Please add LinkedIn Photo'
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                third_img = cv2.putText(third_img, text, (30, 680), font, 1, (255,0,0), 2)
                #third_img = cv2.resize(third_img, (0,0), None,  0.35, 0.35)
                third_img = image_resize(third_img, width = orig_width)
                third_height, third_width = third_img.shape[:2]
                if (orig_height > third_height):
                    third_img = cv2.copyMakeBorder(third_img, top = int((orig_height - third_height) / 2), bottom = int((orig_height - third_height) / 2), left = 0, right = 0, borderType=cv2.BORDER_CONSTANT)
                elif(orig_width > third_width):
                    third_img = cv2.copyMakeBorder(third_img, top = 0, bottom = 0, left = int((orig_width - third_width) / 2), right = int((orig_width - third_width) / 2), borderType=cv2.BORDER_CONSTANT)
                similarities = compare.main(['./split_four/current_frame.jpg', "./split_four/split_4_original.jpg", celeb])
                my_dict["./split_four/split_4_original.jpg"] = similarities[0]
                my_dict[celeb] = similarities[1]

            #outputs photo taken at beginning
            text = str(round((1.6 - similarities[0])/1.6,3)*100) + "%"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            orig_out = cv2.imread('original_after_border.jpg') #top right
            orig_out = Age_Gender(orig_out)
            cv2.putText(orig_out, text, (30, orig_height - 40), font, 1, (255, 0, 0), 2)
            #puts first and second frames together
            sbs1 = np.concatenate((out_frame, orig_out), axis=1)

            #celebrity frame
            fourth_img = cv2.imread(celeb)
            if len(similarities) == 3:
                text = str(round((1.6 - similarities[2])/1.6,3)*100) + "%"
            else:
                text = str(round((1.6 - similarities[1])/1.6,3)*100) + "%"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            fourth_img = cv2.putText(fourth_img, text, (30, 680), font, 1, (255,0,0), 2) #writing bottom right similarity
            #fourth_img = cv2.resize(fourth_img, (0,0), None, 0.35, 0.35)
            fourth_img = image_resize(fourth_img, width = orig_width)
            cv2.imwrite('fourth_img.jpg', fourth_img)
            fourth_height, fourth_width = fourth_img.shape[:2]
            if (orig_height > fourth_height):
                fourth_img = cv2.copyMakeBorder(fourth_img, top = int((orig_height - fourth_height) / 2), bottom = int((orig_height - fourth_height) / 2), left = 0, right = 0, borderType=cv2.BORDER_CONSTANT)
            elif(orig_width > fourth_width):
                fourth_img = cv2.copyMakeBorder(fourth_img, top = 0, bottom = 0, left = int((orig_width - fourth_width) / 2), right = int((orig_width - fourth_width) / 2), borderType=cv2.BORDER_CONSTANT)
            cv2.imwrite('fourth_img_after_border.jpg', fourth_img)
            #puts 3rd and fourth frames together
            sbs2 = np.concatenate((third_img, fourth_img), axis = 1)
            sbs2 = Age_Gender(sbs2)
            #puts all frames together
            two_by_two = np.concatenate((sbs1, sbs2), axis = 0)



            cv2.imshow(wind_name, two_by_two)
            cv2.waitKey(2)

        key = cv2.waitKey(5000)
        if key % 256 == 113 or key % 256 == 81:
            print('entered two_screen')
            cap.release()
            cv2.destroyWindow('Sentry Face Detection')
            cv2.waitKey(1)
            two_screen(my_dict)
        cycle_counter = cycle_counter + 1
        print("--- %s seconds ---" % (time.time() - start_time))

four_screen()
    
    
    
