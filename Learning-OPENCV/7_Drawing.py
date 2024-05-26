# import cv2
#
# img=cv2.imread(r"E:\images2_detection\white-board.jpg")
#
# print(img.shape)
# # Line
# cv2.line(img,(100,150),(250,300),(0,255,0),3)
#
# # Rectangle
# cv2.rectangle(img,(100,200),(250,350),(0,0,255),2)
#
# # circle
# cv2.circle(img,(300,350),100,(255,0,0),3)
# cv2.circle(img,(300,150),100,(255,0,0),3)
#
# # Text
# cv2.putText(img,'Awais',(225,170),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
#
# cv2.imshow("Orignal Image",img)
#
# cv2.waitKey(0)
#
#
#


##############################################################
#..........................TEXT...............................
##############################################################

# import cv2
# img=cv2.imread(r"E:\images2_detection\white-board.jpg")
#
# txt=cv2.putText(img,
# text="Awais",
# org=(160,260),
# fontFace=cv2.FONT_HERSHEY_DUPLEX,  #means font size
# fontScale=2,  # means font size
# color=(0,0,255),
# thickness=3,
# lineType=cv2.LINE_8,
# bottomLeftOrigin=False)
#
# cv2.imshow("Image",txt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#..................... For Reversing of text ...............

# import cv2
# img=cv2.imread(r"E:\images2_detection\white-board.jpg")
#
# txt1=cv2.putText(img,
# text="Awais",
# org=(160,260),
# fontFace=cv2.FONT_HERSHEY_DUPLEX,  #means font size
# fontScale=2,  # means font size
# color=(0,0,255),
# thickness=3,
# lineType=cv2.LINE_8,
# bottomLeftOrigin=False)
#
# txt2=cv2.putText(txt1,
# text="Awais",
# org=(160,270),
# fontFace=cv2.FONT_HERSHEY_DUPLEX,  #means font size
# fontScale=2,  # means font size
# color=(255,0,0),
# thickness=3,
# lineType=cv2.LINE_8,
# bottomLeftOrigin=True)
#
# cv2.imshow("Image",txt2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



##############################################################
#..........................LINE...............................
##############################################################

#
# import cv2
#
# image=cv2.imread(r"E:\images2_detection\person.jpg")
# img=cv2.resize(image,(600,600))
#
# new_img=cv2.line(img,pt1=(240,100),pt2=(330,100),color=(0,255,0),lineType=4,
#                  thickness=4)
#
# cv2.imshow("Image",new_img)
#
# cv2.waitKey(0)
#



##############################################################
#..........................Rectangle...............................
##############################################################

# import cv2
#
# image=cv2.imread(r"E:\images2_detection\person.jpg")
# img=cv2.resize(image,(600,600))
#
#
# txt1=cv2.putText(img,
# text="Habshi",
# org=(248,35),
# fontFace=1,  #means font size
# fontScale=1,  # means font size
# color=(0,0,255),
# thickness=2,
# lineType=16,
# bottomLeftOrigin=False)
#
# new_img=cv2.rectangle(img=txt1,pt1=(250,40),pt2=(325,105),color=(0,255,0),lineType=4,
#                  thickness=4)
#
# cv2.imshow("Image",new_img)
#
# cv2.waitKey(0)








##############################################################
#..........................Circle...............................
##############################################################


#
# import cv2
#
# image=cv2.imread(r"E:\images2_detection\person.jpg")
# img=cv2.resize(image,(600,600))
#
#
# txt1=cv2.putText(img,
# text="Habshi",
# org=(260,30),
# fontFace=1,
# fontScale=1,
# color=(0,0,255),
# thickness=2,
# lineType=16,
# bottomLeftOrigin=False)
#
# new_img=cv2.circle(img=txt1,color=(0,255,0),center=(287,70),radius=38,lineType=16,
#                  thickness=2)  # if you put thickness=-1 it will cover with color
#
# cv2.imshow("Image",new_img)
#
# cv2.waitKey(0)
