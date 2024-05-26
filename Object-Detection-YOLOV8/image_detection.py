from ultralytics import YOLO
import cv2


#model=YOLO(r"E:\Yolov8_Weights\yolov8n.pt")
model = YOLO(r"B:\Yolov8_Weights\yolov8l.pt")
#model = YOLO(r"E:\Yolov8_Weights\yolov8x.pt")

#result1=model(r"E:\images2_detection\BusStudents.png",show=True)
#result2=model(r"E:\images2_detection\crowdOfpeople.jpg",show=True)
result3=model(r"B:\images1_detection\maninhouse.jpg",show=True)
cv2.waitKey(0)







'''

.The first line imports the YOLO class from the ultralytics library.
.The second line imports the cv2 library, which is used for handling and displaying images.
.The third line creates an instance of the YOLO class, which loads the YOLO model from the specified weight file yolov8n.pt.
.The commented out lines result1=model(r"E:\images\BusStudents.png",show=True) and result2=model(r"E:\images\crowdOfpeople.jpg",show=True) are examples of how to use the model object to detect objects in an image. The model() method takes an image file path and an optional show argument, which if set to True, will display the image with the detected objects on the screen.
.The line result3=model(r"E:\OBJECT_DETECTION_ultralytic\images\Lightwood-Dental-Waiting-Room-People.jpg",show=True) uses the model object to detect objects in the image located at the specified file path. The show argument is set to True, so the image with the detected objects will be displayed on the screen.
.The cv2.waitKey(0) line waits for a key press before closing the window displaying the image.




cv2 (OpenCV):
cv2 stands for Open Source Computer Vision Library, and it is widely used for computer vision tasks in Python.It provides functions for image and video processing, including image manipulation, feature detection, object recognition, and more.The cv2 library is commonly used for reading and displaying images and videos, as well as for drawing shapes and annotations on images.

Cvzone:
cvzone is an open-source Python library built on top of OpenCV, providing additional functionalities and utilities for computer vision tasks.It simplifies the process of adding various elements to images and videos, such as bounding boxes, corner rectangles, and text annotations

YOLO:
YOLO is a popular object detection algorithm that divides an image into a grid and predicts bounding boxes and class probabilities for each grid cell.YOLO is known for its real-time object detection capabilities and efficiency.In your Python code, you are using the YOLO model from the Ultralytics library, which is an open-source computer vision library.

Ultralytics:
Ultralytics is a deep learning library that provides pre-trained models and utilities for various computer vision tasks.It is often used for object detection, image classification, and other tasks involving deep learning models.In your code, you are using the Ultralytics library to work with the YOLO model and perform object detection on images and videos.





'''

