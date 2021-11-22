# -*- coding: utf-8 -*-
"""
Created on Mon May 11 22:45:32 2020

@author: shoun
"""

import cv2
import numpy as np
from keras.models import load_model
from random import choice

REV_CLASS_MAP = {0 : 'rock',
                 1 : 'paper',
                 2 : 'scissors',
                 3 : 'none'}

def mapper(val):
    
    return REV_CLASS_MAP[val]

def calculate_winner(move1,move2):
    
    if move1 == move2:
        
        return "TIE !!!"
    
    if move1 == 'rock':
        
        if move2 == 'paper':
            
            return "WINNER IS COMPUTER !!!"
        
        if move2 == 'scissors':
             
             return "WINNER IS PLAYER !!!"
         
    if move1 == 'paper':
        
        if move2 == 'rock':
            
            return "WINNER IS PLAYER !!!"
        
        if move2 == 'scissors':
             
             return "WINNER IS COMPUTER !!!"
         
    if move1 == 'scissors':
        
        if move2 == 'paper':
            
            return "WINNER IS PLAYER !!!"
        
        if move2 == 'rock':
             
             return "WINNER IS COMPUTER !!!"


model = load_model('rock-paper-scissor-model.h5')

cap = cv2.VideoCapture(0)

w = cap.set(3,1080)
h = cap.set(4,1920)

prev_move = None

while True:
        
    ret , frame = cap.read()
    
    #player rectangle 
    cv2.rectangle(frame , (100,100), (500,500), (0,255,0), 2)                        
    
    #computer rectangle
    cv2.rectangle(frame , (800,100), (1200,500), (0,255,0), 2)         

    roi = frame[100:500 , 100:500]
    img = cv2.cvtColor(roi , cv2.COLOR_BGR2RGB)
    img = cv2.resize(img , (255,255))
    
    #test.py
    #predict the move captured in "img"
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)
    
    #input the computer move
    #predict the winner     
    if prev_move != user_move_name:
        
        if user_move_name != 'none':
            
            computer_move_name = choice(['rock','paper','scissors'])
            
            winner = calculate_winner(user_move_name , computer_move_name)
            
        else :
            
            computer_move_name = 'none'
            
            winner = "WAITING !!!"
            
    prev_move = user_move_name

    #displaying the information
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (250, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
    
    if computer_move_name != 'none':
        icon = cv2.imread("computer_images/{}.png".format(computer_move_name))
        #icon = cv2.cvtColor(icon , cv2.COLOR_BGR2RGB)
        icon = cv2.resize(icon , (400,400))
        frame[100:500 , 800:1200] = icon
        
        
    cv2.imshow('PLAYING THE ROCK-PAPER-SCISSOR GAME',frame)
     
    if cv2.waitKey(10) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()    
    
    
    