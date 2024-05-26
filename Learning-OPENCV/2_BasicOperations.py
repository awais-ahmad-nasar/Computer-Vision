#..................Resizing..............

# import cv2
#
# img=cv2.imread(r"E:\images2_detection\doggo.jpg")
# resize_img=cv2.resize(img,(250,250))
#
# print(img.shape)
# print(resize_img.shape)
#
# cv2.imshow('Normal Image',img)
# cv2.imshow('Resize Image',resize_img)
# cv2.waitKey(0)


#.......................or........................

import cv2

img=cv2.imread(r"B:\images2_detection\doggo.jpg",0)
#if i put flag=0 then it will show me balck&white img

resize_img=cv2.resize(img,(250,250))

print(img.shape)
print(resize_img.shape)

cv2.imshow('Normal Image',img)
cv2.imshow('Resize Image',resize_img)
cv2.waitKey(0)


#..................Croping..............
# import cv2
#
# img=cv2.imread(r"E:\images2_detection\two-dogs.jpg")
# print(img.shape)
# crop_img=img[220:740,320:1040]
#
# cv2.imshow("image",img)
# cv2.imshow("Croped image",crop_img)
#
# cv2.waitKey(0)






# ...............Image Multiplication and Division.........................

import cv2
import numpy as np

# Load an image
image = cv2.imread(r"B:\images2_detection\cat_dog.jpg")

# Perform image multiplication by a scalar (increase brightness)
brightened_image = cv2.multiply(image, np.array([1.5]))  # Increase brightness by multiplying by 1.5

# Perform image division by a scalar (decrease brightness)
darkened_image = cv2.divide(image, np.array([2.0]))  # Decrease brightness by dividing by 2.0

# Display the original image and the results
cv2.imshow('Original Image', image)
cv2.imshow('Brightened Image', brightened_image.astype(np.uint8))  # Convert back to uint8 for display
cv2.imshow('Darkened Image', darkened_image.astype(np.uint8))  # Convert back to uint8 for display
cv2.waitKey(0)
cv2.destroyAllWindows()

