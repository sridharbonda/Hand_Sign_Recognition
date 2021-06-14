# Hand_Sign_Recognition

Team members:
1. Bonda Sridhar
2. Balaram Mohanty
3. Kalicharan Jalui
4. Tejas Uday
5. Jacob Panicker

Sign language is one of the oldest and most natural forms of language for communication, but since most people do not know sign language and interpreters are very difficult to come by, we have come up with a real time method using neural networks for American Hand Sign Language. The objective of the project is to create a sign detector, which detects English language alphabets A-Z when the user makes the sign with hand in ROI. The project has been developed using Opencv, keras, tkinter and TensorFlow modules in python.

Project has 3 main parts: Image Processing, Machine Learning and GUI

## Image Processing
 This involves few IP techniques that is to detect the hand in the ROI, for this we calculated the accumulated weighted average from the background and then subtracting this the frame that contains objects like the hand in front of the background  is  differentiated.  This  values  obtained  then  used  to  segment  out  the  hand  from background by thresholding (threshold value decided through hit and try method). 
 
## Machine learning
The dataset (source: https://github.com/shadabsk/Sign-Language-Recognition-Using-Hand-Gestures-Keras-PyQT5-OpenCV) consisting of 45,526 training images and 6,526 testing images, has been used to train and test the CNN model for prediction of signs. The model is trained for 15 epochs resulting in accuracy of 94.67%. 

## GUI
The tkinter module of python has been use to make the GUI. The GUI display the video, ROI thresholded images, prediction and a text box.

## Results:

For Hand Sign Alphabet:  B

![B](https://user-images.githubusercontent.com/58266816/121853639-c7c8e480-cd0e-11eb-8c4b-d86dd7766fc5.JPG)

For Hand Sign Alphabet:  C

![C](https://user-images.githubusercontent.com/58266816/121853724-e7600d00-cd0e-11eb-8069-cd91f001b5d9.JPG)

## Summary

Considering the future prospects of this project, we could attempt to achieve higher accuracy even in case of complex backgrounds by trying out various background subtraction algorithms. Also we could look into the possibility of improving the pre-processing of the already available data in order to predict gestures in low light conditions with a higher accuracy. This project can be further extended to predict many words as well as forming sentences.
