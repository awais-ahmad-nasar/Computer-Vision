# import cv2
#
# img=cv2.imread(r"E:\images2_detection\birds2.jpg")
# resize_img=cv2.resize(img,(500,500))
#
# ret,thresh=cv2.threshold(resize_img,127,255,cv2.THRESH_BINARY_INV)
#
#
# cv2.imshow("Orignal Image",resize_img)
# cv2.imshow("Thresh Image",thresh)
#
#
# cv2.waitKey(0)








import cv2

img=cv2.imread(r"B:\images2_detection\birds2.jpg")
resize_img=cv2.resize(img,(500,500))

img_gray=cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)

ret,thresh=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    print(cv2.contourArea(cnt))

    #if cv2.contourArea(cnt)<600:   #remove small numbers or u can say noice
    if cv2.contourArea(cnt) > 200:
        #cv2.drawContours(resize_img,cnt,-1,(0,255,0),1)  # boudry draw kre ga orignal image ka objects pa

        x1,y1,w,h=cv2.boundingRect(cnt)  #bounding box
        cv2.rectangle(resize_img,(x1,y1),(x1+h,y1+h),(0,255,0),2)  #bounding box


cv2.imshow("Orignal Image",resize_img)
#cv2.imshow("Gray Image",img_gray)
cv2.imshow("Thresh Image",thresh)


cv2.waitKey(0)




