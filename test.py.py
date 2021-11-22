# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:39:55 2020

@author: shoun
"""

from keras.models import load_model
import numpy as  np
import cv2

REV_CLASS_MAP = {
                0 : "rock",
                1 : "paper",
                2 : "scissors",
                3 : "none"
                }

def mapper(val):
    
    return REV_CLASS_MAP[val]

model = load_model('rock-paper-scissor-model.h5')

#image to be identified if the model correctly predicts it or not
img = cv2.imread('test_rock.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img , (255,255))

#predict the image

pred = model.predict(np.array([img]))


move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))