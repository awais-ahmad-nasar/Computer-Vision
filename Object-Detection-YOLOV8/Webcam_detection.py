#####################################..FOR WEBCAM..##################################################


from ultralytics import YOLO
import  cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1980)
cap.set(4, 1080)

model=YOLO(r"E:\Yolov8_Weights\yolov8n.pt")
#model = YOLO(r"E:\Yolov8_Weights\yolov8l.pt")

classNames = ["person",
"bicycle",
"car",
"motorbike",
"aeroplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"sofa",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tvmonitor",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"]

while True:
    success, img  =cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]
            name = classNames[int(cls)]

            #cvzone.putTextRect(img, f'{name} 'f'{conf}', (max(0,x1), max(35,y1)), scale = 0.5)
            cvzone.putTextRect(img, f'{name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)


    cv2.imshow("Image", img)
    # Terminate run when "Q" pressed
    if cv2.waitKey(60) == ord("q"):
             break


cv2.destroyAllWindows()























'''

.The first few lines import the necessary libraries: Ultralytics for YOLO, OpenCV for handling the video feed and image processing, cvzone for some helper functions, and math for some math operations.
.The cap = cv2.VideoCapture(0) line sets up the video capture object, which is set to use the default camera (camera index 0). The next two lines set the width and height of the video feed to 1980 and 1080 pixels, respectively.
.The while loop is where the main program logic is executed. It runs until the user presses the "q" key.
.Inside the loop, the cap.read() function reads a frame from the video capture object and returns a success flag and the image.
.The results = model(img, stream=True) line uses the YOLO model to detect objects in the image. The stream=True argument allows the model to process the image in real-time, as a stream of data.
.The for loop iterates over each detection result in the results object.
.The boxes = r.boxes line extracts the bounding boxes for each detected object.
.The nested for loop iterates over each bounding box.
.The x1, y1, x2, y2 = box.xyxy[0] line extracts the coordinates of the top-left and bottom-right corners of the bounding box.
.The cvzone.cornerRect(img, (x1, y1, w, h)) line draws a rectangle around the bounding box on the image.
.The conf = math.ceil((box.conf[0]*100))/100 line calculates the confidence score of the detection and converts it to a percentage.
.The cls = box.cls[0] line extracts the class index of the detected object.
.The name = classNames[int(cls)] line maps the class index to the corresponding class name.
.The cvzone.putTextRect(img, f'{name} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2) line draws the class name and confidence score on the image, above the bounding box.
.Finally, the cv2.imshow("Image", img) line displays the image with the detected objects on the screen, and the cv2.waitKey(60) line waits for a key press for 60 milliseconds.


'''