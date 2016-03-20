import numpy as np
import csv
from matplotlib import pyplot as plt 
import cv2
from mpl_toolkits.mplot3d import Axes3D
import sys
img=0
mask=0

def loadImage(src): 
    global img
    img=cv2.imread('BaseImageUCD/Pratheepan_Dataset/FacePhoto/'+src ,1) 
    img.astype(np.uint8)
    cv2.imshow('image',img) 


def loadMask(src): 
    global mask                                                                                                                                                                 
    mask=cv2.imread('BaseImageUCD/Ground_Truth/GroundT_FacePhoto/'+src ,1)
    mask.astype(np.uint8)
    mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',mask)

def classifyPixel(x,y):
    R=img[x,y][2]
    B=img[x,y][0]
    G=img[x,y][1]
    if(R>95 and G>40 and B>20 and (max(R,G,B)-min(R,G,B))>15 and abs(R-G)>15 and R>G and R>B):
        return [255,255,255]
    else:
        return [0,0,0] 


if __name__ =='__main__':
    loadImage ('06Apr03Face.jpg')
    loadMask('06Apr03Face.png')
    imgHeight=img.shape[0]
    imgWidth=img.shape[1]
    print "The image width is %d." % imgWidth
    print "The image height is %d." % imgHeight
    print "it has %d channels" % img.shape[2]
    if(img.shape[2]<3):
                    print "not enough channel to work on, try RGB images"
                    sys.exit()
    """blank_image =np.zeros((imgHeight, imgWidth, 3), np.uint8) 
    for x in range(0, imgHeight):
            for y in range(0,imgWidth):
                    blank_image[x,y]=classifyPixel(x,y)

    cv2.imwrite('output.jpg',blank_image)
    """
    """
    here the faces
    """

    hist_fullLABF = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2LAB)],[1,2],mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    hist_fullRGBF = cv2.calcHist([img],[1,2],mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    hist_fullHSVF = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)],[0,1],mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    """
    here the rest
    """

    hist_fullLABR = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2LAB)],[1,2],255-mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    hist_fullRGBR = cv2.calcHist([img],[1,2],255-mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    hist_fullHSVR = cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)],[0,1],255-mask,[32,32],[0,256,0,256])*100000/(imgWidth*imgHeight)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x in range (0,32):
        for y in range (0,32):
            xs = x
            ys = y
            zs = hist_fullHSVR[x][y]
            ax.scatter(xs, ys, zs)
    plt.show()
    
    
    
    
    with open('Hist/RGBFaces.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        data = hist_fullRGBF
        a.writerows(data)
    with open('Hist/HSVFaces.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')                                                                                                                                        
        data = hist_fullHSVF
        a.writerows(data)
    with open('Hist/LABFaces.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')                                                                                                                                        
        data = hist_fullLABF
        a.writerows(data)
    with open('Hist/RGBRest.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        data = hist_fullRGBR
        a.writerows(data)
    with open('Hist/HSVRest.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')                                                                                                                                        
        data = hist_fullHSVR
        a.writerows(data)
    with open('Hist/LABRest.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')                                                                                                                                        
        data = hist_fullLABR
        a.writerows(data)
        plt.show()


    

