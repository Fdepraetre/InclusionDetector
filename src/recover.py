import sys 
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt

from Tkinter import Tk
from tkFileDialog import askopenfilename

if len(sys.argv) == 3 :
	img1_path = sys.argv[1]
	img2_path = sys.argv[1]
else :
	Tk().withdraw() 
	img1_path = askopenfilename() 
	img2_path = askopenfilename() 
if not os.path.isfile(img1_path) :
	sys.exit("{} is not a file").format(img1_path)
if not os.path.isfile(img2_path) :
	sys.exit("{} is not a file").format(img2_path)


img1 = cv2.imread(img1_path,0)
img4 = cv2.imread(img2_path,0)
img2 = cv2.imread(img2_path,0)


#Initiate ORB detector
orb = cv2.ORB_create()

#find keypoints and descriptors with ORB
kp1, des1= orb.detectAndCompute(img1,None)
kp2, des2= orb.detectAndCompute(img2,None)

#create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= True)

#Match descriptor
matches = bf.match(des1,des2)

# Sort them in order of their distance
matches = sorted(matches, key = lambda x:x.distance)


img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],img4, flags=2)

cv2.namedWindow("Final",cv2.WINDOW_NORMAL)
cv2.imshow("Final",img3)

tmp2 = 0
tmp2 = cv2.waitKey(0)
if tmp2 == ord('q'):
    cv2.destroyAllWindows()

#plt.imshow(img4),plt.show()


