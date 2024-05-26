import cv2

image_path1=cv2.imread(r"B:\images2_detection\cat_dog.jpg")
k_size=7   #Neighborhood  (if ksize=70, it ill show most blured img)

img = image_path1
img_blur = cv2.blur(img,(k_size,k_size))
img_guassion = cv2.GaussianBlur(img,(k_size,k_size),5)
img_median = cv2.medianBlur(img,k_size)


cv2.imshow("Orignal Image",img)
cv2.imshow("Blur Image",img_blur)
cv2.imshow("Guassion Blur Image",img_guassion)
cv2.imshow("Median Blur Image",img_median)

cv2.waitKey(0)
#



##########################################################################################

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# .......................... 1st way : By choosing cv2 models...................

# img=cv2.imread(r"B:\images2_detection\cat_dog.jpg")
# blr_img=cv2.GaussianBlur(img,(59,59),5)
#
# fig,axis=plt.subplots(1,2)
# axis[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# axis[1].imshow(cv2.cvtColor(blr_img,cv2.COLOR_BGR2RGB))
# plt.show()


# ........................... 2nd way : By choosing the Mask............................
# img=cv2.imread(r"B:\images2_detection\cat_dog.jpg")
# blr_img=cv2.GaussianBlur(img,(59,59),5)
# ksize=9
# Mask=np.ones((ksize,ksize),np.float32)/(ksize*ksize)
# blred_img=cv2.filter2D(img,-1,Mask)
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# axes[1].imshow(cv2.cvtColor(blred_img, cv2.COLOR_BGR2RGB))
# plt.show()
