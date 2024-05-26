# import cv2
#
# img=cv2.imread(r"E:\images2_detection\two-dogs.jpg")
# resize_img=cv2.resize(img,(720,900))
#
# edge_img=cv2.Canny(resize_img,50,150)
#
#
# cv2.imshow("Orignal Image",resize_img)
# cv2.imshow("Edge Image",edge_img)
#
#
# cv2.waitKey(0)

#
# import cv2
# import numpy as np
# # img=cv2.imread(r"E:\images2_detection\two-dogs.jpg")
# img = cv2.imread("B:\\images2_detection\\two-dogs.jpg")
#
#
#
# resize_img=cv2.resize(img,(500,500))
#
# img_edge=cv2.Canny(resize_img,50,150)
# img_edge_D=cv2.dilate(img_edge,np.ones((3,3),dtype=np.int8))  #(3,3) are thickness, draws lines
# img_edge_E=cv2.erode(img_edge_D,np.ones((3,3),dtype=np.int8))
#
# cv2.imshow("Orignal Image",resize_img)
# cv2.imshow("Edge Image",img_edge)
# cv2.imshow("Dilate Edge Image",img_edge_D)    # use for Thicker
# cv2.imshow("Erosion Edge Image",img_edge_D)   # use for thiner
#
#
# cv2.waitKey(0)






############################################################################################
#...........................Custom Mask(Prewitt Edges)..........................................
# import cv2
# import numpy as np
#
# img = cv2.imread("B:\\images2_detection\\two-dogs.jpg", cv2.IMREAD_GRAYSCALE)
# resize_img=cv2.resize(img,(500,500))
#
# x_mask = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
# y_mask = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
#
#
# gx = cv2.filter2D(resize_img, cv2.CV_64F, x_mask)  # Ensure depth is CV_64F
# gy = cv2.filter2D(resize_img, cv2.CV_64F, y_mask)  # Ensure depth is CV_64F
#
#
# assert gx.shape == gy.shape, "gx and gy should have the same shape"
# assert gx.dtype == gy.dtype, "gx and gy should have the same data type"
#
#
# prewitt_magnitude = cv2.magnitude(gx, gy)  # This should now work without errors
#
# # Normalize the magnitude to the 0-255 range
# prewitt_edges = cv2.normalize(prewitt_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
# cv2.imshow('Orignal Image', resize_img)
# cv2.imshow('Prewitt Edges', prewitt_edges)
#
#
#
#
#
# #...........................Custom Mask(Sobel Edges)..........................................
#
#
#
#
# import cv2
# import numpy as np
#
# img = cv2.imread("B:\\images2_detection\\two-dogs.jpg", cv2.IMREAD_GRAYSCALE)
# resize_img=cv2.resize(img,(500,500))
#
# x_mask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
# y_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
#
#
# gx = cv2.filter2D(resize_img, cv2.CV_64F, x_mask)  # Ensure depth is CV_64F
# gy = cv2.filter2D(resize_img, cv2.CV_64F, y_mask)  # Ensure depth is CV_64F
#
#
# assert gx.shape == gy.shape, "gx and gy should have the same shape"
# assert gx.dtype == gy.dtype, "gx and gy should have the same data type"
#
#
# Sobel_magnitude = cv2.magnitude(gx, gy)  # This should now work without errors
#
# # Normalize the magnitude to the 0-255 range
# Sobel_edges = cv2.normalize(Sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
#
# cv2.imshow('Sobel Edges', Sobel_edges)
#
#
#
# #...........................Custom Mask(Canny Edges)..........................................
#
#
# import cv2
# import numpy as np
#
# img = cv2.imread("B:\\images2_detection\\two-dogs.jpg", cv2.IMREAD_GRAYSCALE)
# resize_img=cv2.resize(img,(500,500))
#
# #...........................................................................
# # x_mask = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
# # y_mask = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
# #
# #
# # gx = cv2.filter2D(resize_img, cv2.CV_64F, x_mask)  # Ensure depth is CV_64F
# # gy = cv2.filter2D(resize_img, cv2.CV_64F, y_mask)  # Ensure depth is CV_64F
# #
# #
# # assert gx.shape == gy.shape, "gx and gy should have the same shape"
# # assert gx.dtype == gy.dtype, "gx and gy should have the same data type"
# #
# #
# # Canny_magnitude = cv2.magnitude(gx, gy)  # This should now work without errors
# #
# # # Normalize the magnitude to the 0-255 range
# # Canny_edges = cv2.normalize(Canny_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
#
# #............................OR...........................
#
#
# Canny_edges=cv2.Canny(resize_img,100,150)
# cv2.imshow('Canny Edges', Canny_edges)
# # cv2.imshow('Orignal Image', resize_img)
# #
# #
# cv2.waitKey(0)
# cv2.destroyAllWindows()



####################################################################################
###############################################################################
# SUBPLOTING ALL : orignal,prewitt,sobel,canny


import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("B:\\images2_detection\\two-dogs.jpg", cv2.IMREAD_GRAYSCALE)
resize_img = cv2.resize(img, (500, 500))


x_mask_prewitt = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
y_mask_prewitt = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
gx_prewitt = cv2.filter2D(resize_img, cv2.CV_64F, x_mask_prewitt)
gy_prewitt = cv2.filter2D(resize_img, cv2.CV_64F, y_mask_prewitt)
prewitt_magnitude = cv2.magnitude(gx_prewitt, gy_prewitt)
prewitt_edges = cv2.normalize(prewitt_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


x_mask_sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
y_mask_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
gx_sobel = cv2.filter2D(resize_img, cv2.CV_64F, x_mask_sobel)
gy_sobel = cv2.filter2D(resize_img, cv2.CV_64F, y_mask_sobel)
sobel_magnitude = cv2.magnitude(gx_sobel, gy_sobel)
sobel_edges = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


canny_edges = cv2.Canny(resize_img, 100, 150)


fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(resize_img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(prewitt_edges, cmap='gray')
axs[0, 1].set_title('Prewitt Edges')
axs[0, 1].axis('off')

axs[1, 0].imshow(sobel_edges, cmap='gray')
axs[1, 0].set_title('Sobel Edges')
axs[1, 0].axis('off')

axs[1, 1].imshow(canny_edges, cmap='gray')
axs[1, 1].set_title('Canny Edges')
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()


