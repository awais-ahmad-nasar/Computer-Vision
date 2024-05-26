# import cv2
#
# image=cv2.imread('some_image.png')
# print(type(image))  #All images are numpy array
# print(image.shape)  # (720,1280,3) whereas (H , W , numberofchannels)



############################## <CHAPTER 2> #########################################
#......................................FOR IMAGES....................................

# import os
# import cv2
#
# #Read image
# # image_path=os.path.join('.',"E:\\images1_detection\maninhouse.jpg")
# image_path=(r"B:\images1_detection\maninhouse.jpg")
#
#
# img=cv2.imread(image_path)
#
# #Write image
# # cv2.imwrite(r"E:\images\maninhouse.jpg",img)
#
# #Visualize image
#
# cv2.imshow('image',img)
# cv2.waitKey(0)

############################## <CHAPTER 2> #########################################
#....................................FOR VEDIO.........................................

# import cv2
#
# #Read image
# vedio_path=r"B:\Videos\bikes.mp4"
# ved=cv2.VideoCapture(vedio_path)
#
# #Visualize image
# ret=True
# while ret:
#     ret,frame = ved.read()     # ret is boolean variable which check if frame works or not
#     if ret:
#         frame = cv2.resize(frame, (600, 600))
#         cv2.imshow('Frame',frame)
#         if cv2.waitKey(40) & 0xFF == ord('q'):
#              break
# i put 50 coz i check the detail of vedio , in 1second it has 50 frames


#...........................................................................
#If you want to give white color to your object
#...........................................................................

#
# import cv2
#
# vedio_path=r"B:\Videos\cars.mp4"
# ved = cv2.VideoCapture(vedio_path)
#
# whiteObjects = cv2.createBackgroundSubtractorMOG2()
#
# #Visualize image
# ret=True
# while ret:
#     ret,frame = ved.read()     # ret is boolean variable which check if frame works or not
#     if ret:
#         frame = cv2.resize(frame, (600, 600))
#         mask=whiteObjects.apply(frame)
#
#         cv2.imshow('Frame',frame)
#         cv2.imshow('White Objects', mask)
#
#         if cv2.waitKey(40) & 0xFF == ord('q'):
#              break


############################## <CHAPTER 2> #########################################
#....................................FOR WEBCAM.........................................

# import cv2
#
# #Read webcam
# webcam=cv2.VideoCapture(0)
#
# #Visualize webcam
# while True:
#     ret,frame=webcam.read()
#     frame = cv2.resize(frame, (600, 550))
#     cv2.imshow('Frame',frame)
#     if cv2.waitKey(40) & 0xFF==ord('q'):
#         break
#
# webcam.release()
# cv2.destroyAllWindows()





