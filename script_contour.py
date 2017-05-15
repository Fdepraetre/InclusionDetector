#coding: utf-8
#!/usr/bin/env python

import sys 
import os 

import numpy as np
import cv2

import csv

from Tkinter import Tk
from tkFileDialog import askopenfilename

DEBUG = True # True/False


def DefineThreshold(thresholdVal=200):
    COMPUTE_THRESHOLD = thresholdVal #0/255
    return COMPUTE_THRESHOLD

def DefineContourSize(contourSize = 1):
    CONTOUR_SIZE = contourSize
    return CONTOUR_SIZE

def DefineCSVOutputFile(csvFile = 0):
    if type(csvFile) == type(str) :
        MYCSVFILE = csvFile
    else :
        Tk().withdraw() 
        MYCSVFILE = askopenfilename() 
    if not os.path.isfile(MYCSVFILE) :
	sys.exit("{} is not a file").format(MYCSVFILE)
    else :
        return MYCSVFILE

def DefineImageFile(imageFile = 0):
    if type(imageFile) == type(str) :
        image_path = imageFile
    else :
        Tk().withdraw() 
        image_path = askopenfilename() 
    if not os.path.isfile(image_path) :
	sys.exit("{} is not a file").format(image_path)
    else :
        return image_path



def InclusionDetector(image_path,csv_path,thresholdVal=200 ,contour_size= 1):
    im = cv2.imread(image_path)

    #Traitement de l'image
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, tresh = cv2.threshold(imgray, thresholdVal,255,cv2.THRESH_BINARY_INV)
    img2, contours, otherstuff= cv2.findContours(tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if DEBUG :
    	cv2.namedWindow("Original",cv2.WINDOW_NORMAL)
	cv2.imshow("Original",img2)
	tmp2 = cv2.waitKey(0)

    # On récupére le plus grand contour.
    areaArray = []
    for i, c in enumerate(contours) : 
    	area = cv2.contourArea(c)
    	areaArray.append(area)
    sorted_data = sorted(zip(areaArray, contours), key = lambda x:x[0], reverse=True)
    # Peut être qu'il faut prendre le second car le plus grand peut être le cadre de l'image.
    # largest_contour = sorted_data[1][1]
    largest_contour = sorted_data[1][1]


    # On calcule le mask
    mask = np.zeros(img2.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, (255),-1)
    cv2.drawContours(mask, [largest_contour], -1, (0), contour_size)
    if DEBUG : 
    	cv2.namedWindow("Mask",cv2.WINDOW_NORMAL)
    	cv2.imshow("Mask",mask)
    	tmp2 = cv2.waitKey(0)

    # On applique le mask
    final_img = cv2.bitwise_and(tresh, tresh, mask=mask)

    if DEBUG :
    	cv2.namedWindow("Final",cv2.WINDOW_NORMAL)
	cv2.imshow("Final",final_img)
    	tmp2 = cv2.waitKey(0)

#**********************************************************
#**********************************************************
#**********************************************************

    white_pixels_final = float(cv2.countNonZero(final_img))
    white_pixels_mask = cv2.countNonZero(mask)
    print ""
    print "  *** USING : ***"
    print "  -> TRESHOLD = {}".format(thresholdVal)
    print "  -> CONTOUR_SIZE = {}\n".format(contour_size)
    print "   *********************************************************"
    print "\t{} pixels over {} pixels in the contour".format(int(white_pixels_final), white_pixels_mask)
    print "\t > Percentage : {:.2%}".format(white_pixels_final/white_pixels_mask)
    print "   *********************************************************"

    with open(csv_path,'rb') as myfile:
        cr = csv.reader(myfile) #on récupére le fichier
        datalist = list(cr) #on le met sous forme de liste
    
    with open(csv_path,'wb') as myfile:
        writer = csv.writer(myfile,delimiter=',',quoting=csv.QUOTE_ALL)
        writer.writerows(datalist+[[image_path,white_pixels_final,white_pixels_mask,(white_pixels_final/white_pixels_mask)]])


#if __name__ == '__main__':
##    ImportNecessary()
#    threshold = DefineThreshold()
#    contourSize = DefineContourSize()
#    csvFile = DefineCSVOutputFile()
#    imageFile = DefineImageFile()
#    print threshold
#    print contourSize
#    print csvFile
#    print imageFile
#    InclusionDetector(threshold,contourSize,imageFile,csvFile)
#
