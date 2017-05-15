#coding: utf-8
#!/usr/bin/env python

import sys 
import os 
import yaml

import numpy as np
import cv2
import csv

from Tkinter import Tk
from tkFileDialog import askopenfilename

COMPUTE_THRESHOLD = 200 #0/255
CONTOUR_SIZE = 1
DEBUG = True # True/False

RED = [0,0,255]
BLUE = [255,0,0]
GREEN = [0,255,0]



if len(sys.argv) == 2 :
	image_path = sys.argv[1]
else :
	Tk().withdraw() 
	image_path = askopenfilename() 
if not os.path.isfile(image_path) :
	sys.exit("{} is not a file").format(image_path)

with open('./Config_File.yaml','r') as Conf:
    settings = yaml.load(Conf)

COMPUTE_THRESHOLD = settings["Threshold"]
CONTOUR_SIZE = settings["Contour_Size"]
DEBUG = settings["DEBUG"]

im = cv2.imread(image_path)

#Traitement de l'image
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, tresh = cv2.threshold(imgray, COMPUTE_THRESHOLD,255,cv2.THRESH_BINARY_INV)
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

drawings = cv2.imread(image_path)

#for i in range(0,np.size(contours)):
#        cv2.drawContours(drawings,,i,RED,2,8,otherstuff,0)


cv2.drawContours(drawings, [largest_contour], -1, RED, CONTOUR_SIZE)

if DEBUG :

    draw_path = image_path[0:len(image_path)-4] +"_contour.jpg"
    cv2.namedWindow("Drawings",cv2.WINDOW_NORMAL)
    cv2.imshow("Drawings", drawings)
    cv2.imwrite(draw_path,drawings)
    tmp2 = cv2.waitKey(0)


# On calcule le mask
mask = np.zeros(img2.shape, np.uint8)
cv2.drawContours(mask, [largest_contour], -1, (255),-1)
cv2.drawContours(mask, [largest_contour], -1, (0), CONTOUR_SIZE)
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
print "  -> TRESHOLD = {}".format(COMPUTE_THRESHOLD)
print "  -> CONTOUR_SIZE = {}\n".format(CONTOUR_SIZE)
print "   *********************************************************"
print "\t{} pixels over {} pixels in the contour".format(int(white_pixels_final), white_pixels_mask)
print "\t > Percentage : {:.2%}".format(white_pixels_final/white_pixels_mask)
print "   *********************************************************"
with open(settings["CSV_File"],'rb') as myfile:
        cr = csv.reader(myfile) #on récupére le fichier
        datalist = list(cr) #on le met sous forme de liste
    
with open(settings["CSV_File"],'wb') as myfile:
        writer = csv.writer(myfile,delimiter=',',quoting=csv.QUOTE_ALL)
        writer.writerows(datalist+[[image_path,white_pixels_final,white_pixels_mask,(white_pixels_final/white_pixels_mask)]])



if DEBUG :
    Meb_path = "./STS6_tri_incl.csv"

#else:
#    Tk().withdraw() 
#    Meb_path = askopenfilename() 
#    if not os.path.isfile(Meb_path) :
#	sys.exit("{} is not a file").format(Meb_path)
#
if DEBUG == True :
    with open(Meb_path,'rb') as myfile:
            cr = csv.reader(myfile,delimiter=';') #on récupére le fichier
            datalist = list(cr) #on le met sous forme de array


    print datalist[0][0]

    Nb_line_datalist = np.size(datalist)/np.size(datalist[0])
    Nb_Classe = 0

    for i in range(2,(Nb_line_datalist)):
        if( datalist[i][0] > Nb_Classe):
            Nb_Classe = int(datalist[i][0])

    
    print Nb_Classe
    
    Color = list() 
    
    
    for i in range(1,Nb_Classe+1):
        if (i % 3) == 1 :
            Color.append([255*(Nb_Classe - i)/Nb_Classe,255 * i/ Nb_Classe,255 * i/
                    Nb_Classe])
        elif (i % 3) == 2 :
            Color.append([255 * i/ Nb_Classe,255*(Nb_Classe - i)/Nb_Classe,255 * i/
                    Nb_Classe])
        elif (i % 3) == 0:
            Color.append([255 * i/ Nb_Classe,255 * i/ Nb_Classe,255*(Nb_Classe -
                i)/Nb_Classe])
    
    Taille_Pixel_X = 0.01
    Taille_Pixel_Y = 0.01
    
    position_Pixel_X_1 = float(datalist[1][2])/Taille_Pixel_X
    print "Position Pixel 1: " + str(position_Pixel_X_1)
    position_Pixel_Y_1 = float(datalist[1][3])/Taille_Pixel_Y
    print "Position Pixel 1: " + str(position_Pixel_Y_1)
    
    position_Pixel_X_2 = float(datalist[2][2])/Taille_Pixel_X
    print "Position Pixel 2: " + str(position_Pixel_X_2)
    position_Pixel_Y_2 = float(datalist[2][3])/Taille_Pixel_Y
    print "Position Pixel 2: " + str(position_Pixel_Y_2)
    
    position_Pixel_X_3 = float(datalist[1][2])/Taille_Pixel_X
    print "Position Pixel 3: " + str(position_Pixel_X_3)
    position_Pixel_Y_3 = float(datalist[2][3])/Taille_Pixel_Y
    print "Position Pixel 3: " + str(position_Pixel_Y_3)
    
    px= tresh[position_Pixel_X_1,position_Pixel_Y_1]
    px2= tresh[position_Pixel_X_2,position_Pixel_Y_2]
    px3= tresh[position_Pixel_X_3,position_Pixel_Y_3]
    
    print px
    print px2
    print px3
