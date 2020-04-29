### The code below aims to create hand-gesture image datasets using 
### your webcam feed. It implements background subtraction and binary thresholding
### and saves both the original colour image with 3 channels and the single channel processed image

import os
import numpy as np
import cv2
import time

class ImageDatasetCreator():
    
    path_processed = os.getcwd()
    classname = ""
    path_original = os.getcwd()
    batch_size = 10
    start_pos = 0
    image_dimensions_h = 300
    image_dimensions_w = 300
    isBgCaptured = False
    threshold = 60
    learningRate = 0
    bgModel = None
    
    def folderCreator(self):
        root_path = os.getcwd()
        self.path_processed = os.path.join(root_path + "/DataSets/" + self.classname)
        self.path_original = os.path.join(root_path + "/DataSets/" + self.classname + "_original")
                            
        if os.path.exists(self.path_processed) == False :
            os.mkdir(self.path_processed)
            
        if os.path.exists(self.path_original) == False :
            os.mkdir(self.path_original)
            
            
    def remove_background(self, frame):        
        foremask = self.bgModel.apply(frame, learningRate=self.learningRate)
        kernel = np.ones((3, 3), np.uint8)
        foremask = cv2.erode(foremask, kernel, iterations=1)
        finalres = cv2.bitwise_and(frame, frame, mask=foremask)
        return finalres
        
    def clickProcessSave(self):
        #Clicks images of size image_dimensions and saves them after 
        #appropriate background subtraction and thresholding
        
        ###Initialization
        count = self.start_pos
        startrect = ((100 + self.image_dimensions_w), (100 + self.image_dimensions_h))     #Start point of capturing window
        endrect = (100,100)       #End point of capturing window
        number_frames = 0         #Frame Count        
        ###
        
        cap = cv2.VideoCapture(0) #Turns on web cam and starts capturing
        
        while(True):
            ret, img = cap.read()
            img = cv2.bilateralFilter(img, 5, 50, 100)
            img = cv2.flip(img,1)
            img = img[endrect[0]:startrect[0], endrect[1]:startrect[1]]
            cv2.imshow('Original Window', img)
            
            
            
            if self.isBgCaptured == True:
                frame = self.remove_background(img)
                #frame = frame[endrect[0]:startrect[0], endrect[1]:startrect[1]]
                cv2.imshow('mask', frame)
                
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                value = (35, 35)
                blurred = cv2.GaussianBlur(grey, value, 0)
                _, thresh = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imshow('Thresholded', thresh)
                
            
            
            
            #KeyBoard stuff
            k = cv2.waitKey(1)
            if k == 27:
                break
            elif k == ord('b'):  # press 'b' to capture the background
                self.bgModel = cv2.createBackgroundSubtractorMOG2(0, 50, detectShadows = False)
                self.isBgCaptured = True
                time.sleep(2)
                print('Background captured')
            elif k == ord('r'):  # press 'r' to reset the background
                time.sleep(1)
                self.bgModel = None
                self.isBgCaptured = False
                print('Reset background')
                
            elif k == ord('c'): # press 'c' to capture  the images
                print(count)
                cv2.imwrite(os.path.join(self.path_processed + "/" + self.classname + "_" + str(count) + ".jpg"), thresh)
                cv2.imwrite(os.path.join(self.path_original + "/" + self.classname + "_col_" + str(count) + ".jpg"), img)
                count += 1
                if(count == (self.start_pos + self.batch_size)):
                    print("Batch Done")
                    break                    
                
            
        cap.release()
        cv2.destroyAllWindows()              
        
            
    def main(self):
        self.classname = input("Enter the Class Name : ")
        self.batch_size = int(float(input("No of images to click now : ")))
        self.start_pos = int(float(input("Starting image index(Be Careful It Overwrites) : ")))
        
        self.image_dimensions_h = int(float(input("Height of image to be captured : ")))
        self.image_dimensions_w = int(float(input("Width of image to be captured : ")))
        
        self.folderCreator()
        self.clickProcessSave()    