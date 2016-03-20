import numpy as np
from matplotlib import pyplot as plt 
import cv2
import sys
img=0

def loadImage(src): 
	global img
	img=cv2.imread(src ,1) 
	cv2.imshow('image',img) 

def classifyPixel(x,y):
    R=img[x,y][2]
        B=img[x,y][0]
        G=img[x,y][1]
        if(R>95 and G>40 and B>20 and (max(R,G,B)-min(R,G,B))>15 and abs(R-G)>15 and R>G and R>B):
            return [255,255,255]
        else:
            return [0,0,0] 



if __name__ =='__main__':
	loadImage ('lena.jpg')
	imgHeight=img.shape[0]
	imgWidth=img.shape[1]
	print "The image width is %d." % imgWidth
	print "The image height is %d." % imgHeight
	print "it has %d channels" % img.shape[2]
	if(img.shape[2]<3):
					print "not enough channel to work on, try RGB images"
					sys.exit()
	blank_image =np.zeros((imgHeight, imgWidth, 3), np.uint8) 
	for x in range(0, imgHeight):
			for y in range(0,imgWidth):
					blank_image[x,y]=classifyPixel(x,y)
	cv2.imwrite('output.jpg',blank_image)
	sys.exit()


	

