# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:16:51 2020

@author: shoun
"""

############# IMPORTING THE LIBRARIES  #####################

import os
import cv2
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout


###############  IMPORTING THE DATASET  #######################    
   
IMG_SAVE_PATH = 'image_data' 

CLASS_MAP = {
        "rock" : 0,
        "paper" : 1,
        "scissors" : 2,
        "none" : 3}

def mapper(val):
    
    return CLASS_MAP[val]

dataset = []

for directory in os.listdir(IMG_SAVE_PATH):
    
    path = os.path.join(IMG_SAVE_PATH,directory)
    
    for item in os.listdir(path):
        
        img = cv2.imread(os.path.join(path,item))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(255,255))
        dataset.append([img,directory])
        

data , labels = zip(*dataset)

labels = list(map(mapper, labels)) 

#one hot encode the labels -> labels consists of rock,paper,scissor,none labels
labels =  to_categorical(labels)

###############     CREATE THE NEURAL NETWORK     ###################3
    
model = Sequential()
        
model.add(Convolution2D(32,3,3,input_shape = (255,255,3),activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
        
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
        
model.add(Convolution2D(128,3,3,activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
        
model.add(Convolution2D(256,3,3,activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
        
model.add(Convolution2D(256,3,3,activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
        
model.add(Convolution2D(512,3,3,activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
        
model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
        
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
        
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
        
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
        
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
        
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
        
model.add(Dense(4,activation = 'softmax'))

opt = keras.optimizers.Adam(learning_rate = 0.0001)

model.compile(optimizer = opt ,loss='categorical_crossentropy',metrics=['accuracy'])


####################    train the model     #########################
model.fit(np.array(data),np.array(labels),epochs = 100)



##############                                   ##########################
############## GOT ALMOST 98-99% TRAINING ACCURACY  ##########################
##############                                   ##########################



####################    save the model      ##########################
model.save("rock-paper-scissor-model.h5")    





      