import numpy as np
from matplotlib import pyplot as plt 
import cv2


def loadImage(src): 
	img=cv2.imread(src ,1) 
	cv2.imshow('image',img) 
	cv2.waitKey(0)
	cv2.destroyAllWindows()
if __name__ =='__main__': 
	loadImage ('lena.jpg')
