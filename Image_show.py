import numpy as np
import cv2


img = cv2.imread("F:\Script Python\STS6_mosa_x5_1.jpg")

cv2.imshow('img',img)

img_clone = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

tmp = cv2.waitKey(0)
if tmp == ord('q'):
	cv2.destroyAllWindows()

canny_output = cv2.Canny(img_clone,50,100)
ret, thresh = cv2.threshold(canny_output, 200, 255,cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(img_clone, 150, 255,cv2.THRESH_BINARY_INV)
ret2, thresh3 = cv2.threshold(img_clone, 150, 255,cv2.THRESH_BINARY)

im2, contours, hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE );


cv2.namedWindow("source",cv2.WINDOW_NORMAL)
cv2.imshow("source",canny_output)
cv2.namedWindow("img grey",cv2.WINDOW_NORMAL)
cv2.imshow("img grey",img_clone)
cv2.namedWindow("img grey 2",cv2.WINDOW_NORMAL)
cv2.imshow("img grey 2",thresh2)
cv2.namedWindow("img grey 3",cv2.WINDOW_NORMAL)
cv2.imshow("img grey 3",thresh3)

cv2.namedWindow("img contour",cv2.WINDOW_NORMAL)
cv2.imshow("img grey",im2)

tmp2 = cv2.waitKey(0)
if tmp2 == ord('q'):
	cv2.destroyAllWindows()

heigth = np.size(contours)

drawings = cv2.cvtColor(img_clone,cv2.COLOR_GRAY2BGR)


if heigth > 0:
	for i in range(0,heigth):
		RED = [0,0,255]
		cv2.drawContours(drawings,contours, i, RED, 2, 8, hierarchy, 0);
		print i


cv2.namedWindow("img DRAWINGS",cv2.WINDOW_NORMAL)
cv2.imshow("img DRAWINGS", drawings)
tmp2 = cv2.waitKey(0)
if tmp2 == ord('q'):
	cv2.destroyAllWindows()

# lower_blue = np.array([110,50,50])
# upper_blue = np.array([130,255,255])



# tmp2 = cv2.waitKey(0)
# if tmp2 == ord('q'):
# 	cv2.destroyAllWindows()


# mask = cv2.inRange(img_clone, lower_blue, upper_blue)

# res = cv2.bitwise_and(img,img, mask= mask)

# cv2.imshow('image',img)
# cv2.imshow('mask',mask)
# cv2.imshow('res',res)
# tmp2 = cv2.waitKey(0)
# if tmp2 == ord('q'):
# 	cv2.destroyAllWindows()
