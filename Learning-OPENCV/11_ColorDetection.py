import cv2
import numpy as np
def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 420, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)



while True:
    image = cv2.imread(r"B:\images2_detection\lambo.jpg")
    img = cv2.resize(image, (300, 250))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hu_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    hu_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    sa_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    sa_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    va_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    va_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    print(hu_min,hu_max,sa_min,sa_max,va_min,va_max)

    lower = np.array([hu_min,sa_min,va_min])
    upper = np.array([hu_max, sa_max, va_max])
    mask =  cv2.inRange(imgHSV,lower,upper)
    imgResult=cv2.bitwise_and(img,img,mask=mask)


    cv2.imshow('Original image', img)
    cv2.imshow('HSV image', imgHSV)
    cv2.imshow('Mask image', mask)
    cv2.imshow('Result image', imgResult)
    if cv2.waitKey(40) & 0xFF == ord('q'):
              break



