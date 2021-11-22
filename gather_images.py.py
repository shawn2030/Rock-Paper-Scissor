import cv2
import os

WORK_DIR = os.getwd()

#message is a nominal variable that when called upon should be a string that should have the value "rock","paper" or "scissor"
def gather_images(message):
    x = message
    os.chdir(WORK_DIR + "/image_data/" + x)
    cap = cv2.VideoCapture(0)
    w = cap.set(3,1080)
    h= cap.set(4,1920)

    count = 0
    start = False

    while True:
        
        ret ,frame = cap.read()
        
        if count == 200:
            break
        
        cv2.rectangle(frame, (100,100), (500,500), (0,255,0), 2)
        roi = frame[100:500 , 100:500]
        
        if start:
            
            cv2.imwrite('{}.jpg'.format(count),roi)
            count = count + 1
            
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame , "Collecting {}".format(count),(5,50),font,0.7,(0,255,255),2,cv2.LINE_AA)    
        
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(10) & 0xFF == ord('a'):
            
            start = not start
        
        if cv2.waitKey(10) & 0xFF == 27:
            break



    print("\n{} images saved to {} label. ".format(count,x))        
    cap.release()
    cv2.destroyAllWindows()

    os.chdir(WORK_DIR)



gather_images("rock")
gather_images("paper")
gather_images("scissor")

#to have images with random noise and real photos we will add another directory of images that is called as none
gather_images("none")