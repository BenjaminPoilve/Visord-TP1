from __future__ import division
import numpy as np
import csv
from matplotlib import pyplot as plt 
import cv2
import os
from os import listdir
from os.path import isfile, join
from mpl_toolkits.mplot3d import Axes3D
import sys
img=0
mask=0
hist_fullLABF =0
hist_fullRGBF =0
hist_fullHSVF =0
hist_fullLABR =0
hist_fullRGBR =0
hist_fullHSVR =0
mypath='./BaseImageUCD/Pratheepan_Dataset/FacePhoto/' 
toTest='m_unsexy_gr'
ppeau=0
totalp=0
percentPeau=0
TP=0
FP=0

def loadImage(src): 
    global img
    print 'BaseImageUCD/Pratheepan_Dataset/FacePhoto/'+src
    img=cv2.imread('BaseImageUCD/Pratheepan_Dataset/FacePhoto/'+src ,1) 
    img.astype(np.uint8)
    cv2.imshow('image',img) 


def loadMask(src): 
    global mask     
    global ppeau
    global totalp
    mask=cv2.imread('BaseImageUCD/Ground_Truth/GroundT_FacePhoto/'+src ,1)
    mask.astype(np.uint8)
    mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ppeau+=cv2.countNonZero(mask)
    totalp+=mask.shape[0]*mask.shape[1]
    cv2.imshow('image',mask)

def classifyPixel(x,y):
    R=img[x,y][2]
    B=img[x,y][0]
    G=img[x,y][1]
    if(R>95 and G>40 and B>20 and (max(R,G,B)-min(R,G,B))>15 and abs(R-G)>15 and R>G and R>B):
        return [255,255,255]
    else:
        return [0,0,0] 

def analyzeImage(imgName):
    global hist_fullLABF
    global hist_fullHSVF
    global hist_fullRGBF
    global hist_fullLABR 
    global hist_fullRGBR 
    global hist_fullHSVR 
    print imgName    
    loadImage (imgName+'.jpg')
    loadMask(imgName+'.png')
    imgHeight=img.shape[0]
    imgWidth=img.shape[1]
    print "The image width is %d." % imgWidth
    print "The image height is %d." % imgHeight
    print "it has %d channels" % img.shape[2]
    if(img.shape[2]<3):
                    print "not enough channel to work on, try RGB images"
                    sys.exit()

    hist_fullLABF += cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2LAB)],[1,2],mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    hist_fullRGBF += cv2.calcHist([img],[1,2],mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    hist_fullHSVF += cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)],[0,1],mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)

    hist_fullLABR += cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2LAB)],[1,2],255-mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    hist_fullRGBR += cv2.calcHist([img],[1,2],255-mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    hist_fullHSVR += cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)],[0,1],255-mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)

def translate(value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def classify(imgName):
    global img
    global FP
    global TP
    loadImage(imgName+'.jpg')
    loadMask(imgName+'.png')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgHeight=img.shape[0]
    imgWidth=img.shape[1]
    blank_image =np.zeros((imgHeight, imgWidth, 1), np.uint8)
    for x in range(0, imgHeight):
        for y in range(0,imgWidth):
            A=img[x,y][0]
            A=translate(A,0,255,0,31)
            B=img[x,y][1]
            B=translate(B,0,255,0,31)
            histopeau=hist_fullHSVF[A,B]
            histononpeau=hist_fullHSVR[A,B]
            print (histopeau*percentPeau)/(histopeau*percentPeau+histononpeau*(100-percentPeau))
            if((histopeau*percentPeau)/(histopeau*percentPeau+histononpeau*(100-percentPeau))>0.001):
                blank_image[x,y]=255
                if(mask[x,y]!=0):
                    TP+=1
                else:
                    FP+=1
            else:
                blank_image[x,y]=0
    cv2.imwrite('output2.jpg',blank_image)
    print TP
    print FP
    sys.exit()
    

if __name__ =='__main__':
    global percentPeau
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    count=0
    for imagefile in onlyfiles :
        print os.path.splitext(imagefile)[1]
        if(os.path.splitext(imagefile)[1]== '.jpg' or os.path.splitext(imagefile)[0]== '.png'):
            if(os.path.splitext(imagefile)[0]!=toTest):
                analyzeImage(os.path.splitext(imagefile)[0])
                count+=1
    print ppeau
    print totalp
    percentPeau=ppeau/totalp
    print percentPeau
    print TP
    print FP


    classify(toTest);
    
    

   

