#!/usr/bin/env python
# coding: utf-8

# <font color='red' size='6' ><center>**IPMV Project 2021 ( Hand Sign Recognition)**</font></center>

# In[1]:


#Import libraries and modules#
import warnings
import numpy as np
import cv2
from tkinter import *
from tkinter import font
from PIL import Image, ImageTk
from threading import Thread
from tkinter.font import Font

#Video Capture Window#
width, height = 1000, 1000
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#Defining the frames#
root = Tk()
my_font= Font(size=50, family ='Times New Roman' , weight='bold')
root.title("sign detection")
root.bind('<Escape>', lambda e: root.quit())

frame1=LabelFrame(root, text="Thresholding Output",padx=10,pady=10)
frame2=LabelFrame(root, text="Sign Detected Output",padx=130,pady=85)
frame3=LabelFrame(root, text="Camera Frame")

frame1.grid(row=0, column= 1,padx=0 , pady= 0)
frame2.grid(row=1, column= 1,padx=5 , pady= 5)
frame3.grid(row=0, column= 0,rowspan=2 ,padx=10 , pady= 10)

lmain1= Label(frame1)
lmain1.pack()
lmain2= Label(frame2 , font=my_font)
lmain2.pack()
lmain3= Label(frame3)
lmain3.pack()
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


#Function for processing the image obtained#
def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


# In[3]:


#Function to find the difference and the contours#
def segment_hand(frame, threshold=25):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    _ , thresholded = cv2.threshold(diff, threshold,255,cv2.THRESH_BINARY)

    #Grab the external contours for the image#
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        return (thresholded, hand_segment_max_cont)


# In[4]:


#Converting the numpy array output to understandable output#
def find_alpha(predict):
    alpha = ''
    if predict[0][0]==1:
        alpha = 'A'
    elif predict[0][1]==1:
        alpha = 'B'
    elif predict[0][2]==1:
        alpha = 'C'
    elif predict[0][3]==1:
        alpha = 'D'
    elif predict[0][4]==1:
        alpha = 'E'
    elif predict[0][5]==1:
        alpha = 'F'
    elif predict[0][6]==1:
        alpha = 'G'
    elif predict[0][7]==1:
        alpha = 'H'
    elif predict[0][8]==1:
        alpha = 'I'
    elif predict[0][9]==1:
        alpha = 'J'
    elif predict[0][10]==1:
        alpha = 'K'
    elif predict[0][11]==1:
        alpha = 'L'
    elif predict[0][12]==1:
        alpha = 'M'
    elif predict[0][13]==1:
        alpha = 'N'
    elif predict[0][14]==1:
        alpha = 'O'
    elif predict[0][15]==1:
        alpha = 'P'
    elif predict[0][16]==1:
        alpha = 'Q'
    elif predict[0][17]==1:
        alpha = 'R'
    elif predict[0][18]==1:
        alpha = 'S'
    elif predict[0][19]==1:
        alpha = 'T'
    elif predict[0][20]==1:
        alpha = 'U'
    elif predict[0][21]==1:
        alpha = 'V'
    elif predict[0][22]==1:
        alpha = 'W'
    elif predict[0][23]==1:
        alpha = 'X'
    elif predict[0][24]==1:
        alpha = 'Y'
    elif predict[0][25]==1:
        alpha = 'Z'
   
    else:
        alpha = 'Unable to recognise'
    return alpha


# In[5]:


def find_word(predict):
    alpha = ''
    if predict[0][22]==1:
        alpha = 'Welcome'
    elif predict[0][17]==1:
        alpha = 'Thanks'
    elif predict[0][2]==1:
        alpha = 'Stop'
    elif predict[0][5]==1:
        alpha = 'Great'
    elif predict[0][19]==1:
        alpha = 'Disagree'
    elif predict[0][24]==1:
        alpha = 'Agree'
    else:
        alpha = 'Unable to recognise'
    return alpha


# In[6]:


def find_number(predict):
    alpha = ''
    if predict[0][0]==1:
        alpha = '0'
    elif predict[0][8]==1:
        alpha = '1'
    elif predict[0][10]==1:
        alpha = '2'
    elif predict[0][22]==1:
        alpha = '3'
    else:
        alpha = 'Unable to recognise'
    return alpha


# In[7]:


#Importing libraries# 
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from keras.preprocessing import image
background = None
accumulated_weight = 0.5


# In[8]:


#Loading the model# 
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights('model1.h5')


# In[9]:


#Button click function#
def button_click():
    string1 = e.get()
    string2 = " <--Rubbish "
    e.delete(0, END)
    e.insert(0, str(string1)+string2)


# In[10]:


#Text box#
e= Entry(root, width= 105 , borderwidth = 5)
e.grid(row = 3 , column = 0  , padx= 10, pady =10)
#button
button_1 = Button(root , text ="Click to Clear Text", padx= 80, pady= 3, command = button_click )
button_1.grid(row=3 , column= 1)

background = None
accumulated_weight = 0.5

#Creating the dimensions for the ROI#
ROI_top = 100
ROI_bottom = 300
ROI_right = 350
ROI_left = 600


# In[11]:


#Main Code#
import numpy as np

cam = cv2.VideoCapture(0) 
num_frames = 0
element = 10
num_imgs_taken = 0
result = 0
predicted = 0
predicted_word = 0
predicted_num = 0
thresholded = np.zeros((480, 640, 3), dtype=np.uint8)
#Function to obtain the input from the user and show the prediction#
def show_frame():
    global num_frames
    global element
    global num_imgs_taken
    global thresholded
    global result
    global result1
    global predicted
    global predicted_word
    global predicted_num
    ret, frame = cam.read()

    #Flipping the frame to prevent inverted image of captured frame#
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 60:
        cal_accum_avg(gray_frame, accumulated_weight)
        if num_frames <= 59:
            
            cv2.putText(frame_copy, "Fetching Background...PLEASE WAIT!",(80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
    #To configure the hand specifically into the ROI#
    elif 60 <= num_frames <= 300: 

        hand = segment_hand(gray_frame)
        
        cv2.putText(frame_copy, "Adjust your hand...Gesture for" + str(element), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
  (0,0,255),2)
        
        #Checking if the hand is actually detected by counting the numberof contours detected#
        if hand is not None:
            
            thresholded, hand_segment = hand

            #Draw contours around hand segment#
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
            ROI_top)], -1, (255, 0, 0),1)
            
            cv2.putText(frame_copy, str(num_frames)+"For" + str(element),
            (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            #Also display the thresholded image
            #cv2.imshow("Thresholded Hand Image", thresholded)

            
        
    else: 
        
        #Segmenting the hand region#
        hand = segment_hand(gray_frame)
        
        #Checking if we are able to detect the hand#
        if hand is not None:
            
            #Unpack the thresholded img and the max_contour#
            thresholded, hand_segment = hand

            #Drawing contours around hand segment#
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right,
            ROI_top)], -1, (255, 0, 0),1)
            
            #cv2.putText(frame_copy, str(num_frames), (70, 45),
            #cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            
            
            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            #thresholded = np.reshape(thresholded,(1,thresholded.shape[0],thresholded.shape[1],3))
            result = classifier.predict(thresholded.reshape(-1,64,64,3))
            #inp_text = e.get()
            predicted = find_alpha(result)
            predicted_word = find_word(result)
            predicted_num = find_number(result)
            print(result.shape)
            # Displaying the thresholded image
            #cv2.imshow("Thresholded Hand Image", thresholded)
           # global thresholded




    #Drawing ROI on frame copy#
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,ROI_bottom), (255,128,0), 3)
    
    cv2.putText(frame_copy, " IPMV Proj Hand Sign Recognition", (20, 30), cv2.FONT_ITALIC, 1, (0,0,255), 2)
    
    #Increment the number of frames for tracking#
    
    num_frames += 1
    #lmain2= Label(frame2,text= result)
    #lmain2.pack()
    #lmain2.config(text="")

    #Display the frame with segmented hand#
    #cv2.imshow("Sign Detection", frame_copy)
    cv2image = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain3.imgtk = imgtk
    lmain3.configure(image=imgtk)

    #print(color.BOLD + 'Hello World !' + color.END)
    
    #predicted = "\033[1m" + str(predicted) + "\033[0m"
    #predicted= '\033[1m' + str(predicted)
    inp_text = e.get()
    for i in inp_text:
        if(i == 'a' or 'alphabets'):
            lmain2.config(text = predicted)
        elif(i == 'n' or 'number'):
            lmain2.config(text = predicted_num)
        elif(i == 'w' or 'word'):
            lmain2.config(text = predicted_word)
        else:
            lmain2.config(text = 'Enter the choice')
    # lmain2.config(text = result)
    
    img = Image.fromarray(thresholded)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain1.imgtk = imgtk
    lmain1.configure(image=imgtk)
    
    root.after(10, show_frame)
   
    
    #Closing windows with Esc key (any other key with ord can be used too.)#
    #k = cv2.waitKey(1) & 0xFF

show_frame()
root.mainloop()
cv2.destroyAllWindows()
cam.release()


# In[ ]:




