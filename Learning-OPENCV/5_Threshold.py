'''
convert image with many colors into Binary image
'''


#    ............Global Threshold...............

# import cv2
#
# image_path=cv2.imread(r"E:\images2_detection\Bear.jpg")
#
# img = image_path
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh=cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY)
#
#
#
# cv2.imshow("Orignal Image",img)
# cv2.imshow("Gray Image",img_gray)
# cv2.imshow("Thresh Image",thresh)
#
# cv2.waitKey(0)

#.............................. AND .........................
import cv2

image_path=cv2.imread(r"B:\images2_detection\Bear.jpg")

img = image_path
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY)


thresh=cv2.blur(thresh,(10,10))
ret,thresh=cv2.threshold(thresh,80,255,cv2.THRESH_BINARY)

cv2.imshow("Orignal Image",img)
cv2.imshow("Gray Image",img_gray)
cv2.imshow("Thresh Image1",thresh)

cv2.waitKey(0)



#    ............Adaptive Threshold...............

# import cv2
#
# image_path=cv2.imread(r"E:\images2_detection\handwritten2.jpg")
#
# img = image_path
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# thresh=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,30)
#
#
# cv2.imshow("Orignal Image",img)
# cv2.imshow("Threshold Image",thresh)   #orignal ma jo black black show hoti jaga usy khatam krta
#
# cv2.waitKey(0)



#.............................. AND .........................

# import cv2
#
# image_path=cv2.imread(r"B:\images2_detection\Bear.jpg")
#
# img = image_path
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# ret,simple_thresh=cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY)
# Adaptive_thresh=cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,30)
#
# cv2.imshow("Orignal Image",img)
# cv2.imshow("Simple Thresh Image1",simple_thresh)
# cv2.imshow("Adaptive Thresh Image1",Adaptive_thresh)
#
# cv2.waitKey(0)
