import cv2
from ultralytics import YOLO
model = YOLO(r"B:\Yolov8_Weights\yolov8n.pt")
video_path = r"B:\Videos\motorbikes.mp4"
cap = cv2.VideoCapture(video_path)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = 900
height = 600
while cap.isOpened():
    success, frame = cap.read()
    if success:
        resized_frame = cv2.resize(frame, (width,height))
        results = model.track(resized_frame, persist=True)
        annotated_frame = results[0].plot()
        cv2.imshow("Tracking", annotated_frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
