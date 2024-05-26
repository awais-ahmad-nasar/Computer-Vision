# import cv2
# from utils import get_limits
#
# yellow=[0,255,255]   #yellow in BGR colospace
#
# webcam=cv2.VideoCapture(0)
#
# while True:
#     ret,frame=webcam.read()
#     hsvImage=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#
#     lower_limit,upper_limit=get_limits(color=yellow)
#
#     mask=cv2.inRange(hsvImage,lower_limit,upper_limit)
#
#
#     cv2.imshow('Frame',mask)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break
#
# webcam.release()
# cv2.destroyAllWindows()


'''
In above code if you run black screen will be shown but
if you put yellow color object it will show white and background will be black
'''





import cv2
from utils import get_limits
from PIL import Image

yellow=[0,255,255]   #yellow in BGR colospace

webcam=cv2.VideoCapture(0)

while True:
    ret,frame=webcam.read()
    hsvImage=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_limit,upper_limit=get_limits(color=yellow)

    mask=cv2.inRange(hsvImage,lower_limit,upper_limit)

    mask_ = Image.fromarray(mask)
    bbox=mask_.getbbox()
    if bbox is not None:
        x1,y1,x2,y2=bbox
        frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)

    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
