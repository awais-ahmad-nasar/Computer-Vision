#####################################..FOR VIDEOS..##################################################
# from ultralytics import YOLO
# import cv2   #image , video ko resize , capture etc krny ka lia use krty
# import cvzone
# import math
# import time
# import os
# cap = cv2.VideoCapture(1)  # For Yolo_vedio
# cap.set(3, 1280)
# cap.set(4, 720)
#
# #vedio_path=os.path.join('.',r"E:\Videos\bikes.mp4")      #50 persecond frame
# #vedio_path=os.path.join('.',r"E:\Videos\cars.mp4")       #30 persecond frame
# vedio_path=os.path.join('.', r"B:\Videos\motorbikes.mp4")  #25 persecond fram
# cap = cv2.VideoCapture(vedio_path)
# #cap = cv2.VideoCapture("Videos/motorbikes.mp4")
#
#
# model=YOLO(r"B:\Yolov8_Weights\yolov8n.pt")
# #model = YOLO(r"E:\Yolov8_Weights\yolov8l.pt")
#
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]
#
# prev_frame_time = 0
# new_frame_time = 0
#
# while True:
#     new_frame_time = time.time()
#     success, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding Box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(img, (x1, y1, w, h))
#             # Confidence
#             conf = math.ceil((box.conf[0] * 100)) / 100
#             # Class Name
#             cls = int(box.cls[0])
#
#             cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
#
#     fps = 1 / (new_frame_time - prev_frame_time)
#     prev_frame_time = new_frame_time
#     print(fps)
#
#     cv2.imshow("Image", img)
#     # Terminate run when "Q" pressed
#     if cv2.waitKey(25) == ord("q"):
#         break


























'''
.The first few lines import the necessary libraries: YOLO for YOLO object detection, cv2 for handling ,displaying and processing the video frames, cvzone for drawing bounding boxes and text on the frames, math for some math operations, time for measuring elapsed time between frames, and os for joining the video file path.
.The line cap = cv2.VideoCapture(1) creates a cv2 video capture object that reads from the camera with index 1.
.The line cap.set(3, 1280) sets the width of the video frames to 1280 pixels.
.The line cap.set(4, 720) sets the height of the video frames to 720 pixels.
.The line cap = cv2.VideoCapture(vedio_path) opens the video file and creates a cv2 video capture object that reads from the video file.
.The lines prev_frame_time = 0 and new_frame_time = 0 are used to calculate the FPS (frames per second) of the video.
.The while loop runs until the user presses the "q" key.
.The line success, img = cap.read() reads a frame from the video and stores it in the img variable.
.The line results = model(img, stream=True) runs the object detection model on the current frame.And the stream=True argument allows the model to process the image in real-time, as a stream of data
.The nested for loops iterate over the detection results and draw bounding boxes and text on the frame.
.The line fps = 1 / (new_frame_time - prev_frame_time) calculates the FPS of the video.
.The line prev_frame_time = new_frame_time updates the prev_frame_time variable for the next iteration.
.The line cv2.imshow("Image", img) displays the current frame with the detected objects.
.The line if cv2.waitKey(25) == ord("q"): checks if the "q" key has been pressed and exits the loop if it has.


'''



#<><><><><><>Another method<><><><><><><>
#
import random
import os
import cv2
import numpy as np
from ultralytics import YOLO

# opening the file in read mode
my_file = open("B:\\Users\\AWAIS AHMAD\\Downloads\\coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO(r"B:\Users\AWAIS AHMAD\PycharmProjects\ObjectDetection_Yolo\Yolo_Weights\yolov8n.pt")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480


vedio_path=os.path.join('.',r"E:\Videos\motorbikes.mp4")  #25 persecond frame
cap = cv2.VideoCapture(vedio_path)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(25) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()






