
#.............Showing 2  img in one frame.................

# import cv2
# import matplotlib.pyplot as plt
#
# # Read the images using OpenCV
# img1 = cv2.imread(r"E:\images2_detection\doggo.jpg")
# img2 = cv2.imread(r"E:\images2_detection\two-dogs.jpg")
#
# #Task 1:Divide img in 4 parts
#
# print(len(img1),img1)
#
#
# # Create a figure and subplots
# fig, axes = plt.subplots(1, 2)
#
# # Display each image in a separate subplot
# axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
# axes[0].axis('off')  # Hide axes
# axes[0].set_title('Image 1')
#
# axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
# axes[1].axis('off')  # Hide axes
# axes[1].set_title('Image 2')
#
#
# # Adjust layout to prevent overlap of titles
# plt.tight_layout()
#
# # Display the figure
# plt.show()





#..........................single image(4 times) in one frame.......................
# import cv2
import numpy as np


# image_path=(r"E:\images1_detection\maninhouse.jpg")
# img=cv2.imread(image_path)

# horizontal=np.hstack((img,img))
# verticle=np.vstack((horizontal,horizontal))
#
# cv2.imshow('image',verticle)
# cv2.waitKey(0)

#..........................moving slide in one frame.......................

# import cv2
# import os
#
# image_path=(r"E:\images1_detection\maninhouse.jpg")
# img=cv2.imread(image_path)
# list_name=os.listdir(r"E:\images2_detection")
# print(list_name)    # ['Bear.jpg', 'birds.jpg', 'birds2.jpg', 'BusStudents.png' ,etc]
#
# for name in list_name:
#     path ="E:\\images2_detection"
#     img_name= path + "\\" + name
#     img1=cv2.imread(img_name)
#     img2=cv2.resize(img1,(500,600))
#     cv2.imshow('image', img2)
#     cv2.waitKey(1000)   # 1000 means 1 second ka bad new new image ati rhe gi
#
# cv2.destroyAllWindows()




#.............Showing 1 same img in 4 images(small parts) in one frame.................

#
# import cv2
# import matplotlib.pyplot as plt
#
#
# image_path= r"E:\images2_detection\doggo.jpg"
# img1 = cv2.imread(image_path)
# #
# hight,width = img1.shape[:2]
# print("hight", hight)
# print("width",width)
# #
# part1 = img1[0:hight//2, 0:width//2]
# part2 = img1[0:hight//2, (width//2)+1:]
# part3 = img1[(hight//2)+1:, 0:width//2 ]
# part4 = img1[(hight//2)+1:, (width//2)+1:]
#
# fig, axes = plt.subplots(2, 2)
#
#
# axes[0,0].imshow(cv2.cvtColor(part1, cv2.COLOR_BGR2RGB))
# axes[0,0].axis('off')  # Hide axes
# axes[0,0].set_title('Image 1')
#
# axes[0,1].imshow(cv2.cvtColor(part2, cv2.COLOR_BGR2RGB))
# axes[0,1].axis('off')  # Hide axes
# axes[0,1].set_title('Image 2')
#
# axes[1,0].imshow(cv2.cvtColor(part3, cv2.COLOR_BGR2RGB))
# axes[1,0].axis('off')  # Hide axes
# axes[1,0].set_title('Image 3')
#
# axes[1,1].imshow(cv2.cvtColor(part4, cv2.COLOR_BGR2RGB))
# axes[1,1].axis('off')  # Hide axes
# axes[1,1].set_title('Image 4')
#
#
# plt.show()




#.............Shuffling the images parts and then merge it.................

# import cv2
# import numpy as np
#
# image_path= r"E:\images2_detection\doggo.jpg"
# img1 = cv2.imread(image_path)
#
# hight,width = img1.shape[:2]
#
# p1 = img1[0:hight//2, 0:width//2]
# p2 = img1[0:hight//2, (width//2):]
# p3 = img1[(hight//2):, 0:width//2 ]
# p4 = img1[(hight//2):, (width//2):]
#
#
#
# a=np.concatenate((p1,p4),axis=1)
# b=np.concatenate((p2,p3),axis=1)
# con=np.concatenate((a,b),axis=0)
#
#
#
# cv2.imshow('Shuffled image',con)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

