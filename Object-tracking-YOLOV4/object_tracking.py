# import cv2
# import numpy as np
# from object_detection import ObjectDetection
# import math
#
# # Initialize Object Detection
# od = ObjectDetection()
#
# video=r"B:\Videos\cars.mp4"
#
# #video=r"B:\Videos\cars.mp4"
# cap = cv2.VideoCapture(video)
#
# # Initialize count
# count = 0
# center_points_prev_frame = []
#
# tracking_objects = {}
# track_id = 0
#
# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (600, 600))
#     count += 1
#     if not ret:
#         break
#
#     # Point current frame
#     center_points_cur_frame = []
#
#     # Detect objects on frame
#     (class_ids, scores, boxes) = od.detect(frame)
#     for box in boxes:
#         (x, y, w, h) = box
#         cx = int((x + x + w) / 2)
#         cy = int((y + y + h) / 2)
#         center_points_cur_frame.append((cx, cy))
#         #print("FRAME N°", count, " ", x, y, w, h)
#
#         # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Only at the beginning we compare previous and current frame
#     if count <= 2:
#         for pt in center_points_cur_frame:
#             for pt2 in center_points_prev_frame:
#                 distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
#
#                 if distance < 20:
#                     tracking_objects[track_id] = pt
#                     track_id += 1
#     else:
#
#         tracking_objects_copy = tracking_objects.copy()
#         center_points_cur_frame_copy = center_points_cur_frame.copy()
#
#         for object_id, pt2 in tracking_objects_copy.items():
#             object_exists = False
#             for pt in center_points_cur_frame_copy:
#                 distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
#
#                 # Update IDs position
#                 if distance < 20:
#                     tracking_objects[object_id] = pt
#                     object_exists = True
#                     if pt in center_points_cur_frame:
#                         center_points_cur_frame.remove(pt)
#                     continue
#
#             # Remove IDs lost
#             if not object_exists:
#                 tracking_objects.pop(object_id)
#
#         # Add new IDs found
#         for pt in center_points_cur_frame:
#             tracking_objects[track_id] = pt
#             track_id += 1
#
#     for object_id, pt in tracking_objects.items():
#         cv2.circle(frame, pt, 5, (0, 0, 255), -1)
#         cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
#
#     print("Tracking objects")
#     print(tracking_objects)
#
#
#     print("CUR FRAME LEFT PTS")
#     print(center_points_cur_frame)
#
#
#     cv2.imshow("Frame", frame)
#
#     # Make a copy of the points
#     center_points_prev_frame = center_points_cur_frame.copy()
#
#     if cv2.waitKey(50) & 0xFF == ord('q'):
#         break
#
#
# cap.release()
# cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
# import math
#
#
# class ObjectDetection:
#     def __init__(self, weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg"):
#         print("Loading Object Detection")
#         print("Running opencv dnn with YOLOv4")
#         self.nmsThreshold = 0.4
#         self.confThreshold = 0.5
#         self.image_size = 608
#
#         # Load Network
#         net = cv2.dnn.readNet(weights_path, cfg_path)
#
#         # Enable GPU CUDA
#         net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#         net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#         self.model = cv2.dnn_DetectionModel(net)
#
#         self.classes = []
#         self.load_class_names()
#         self.colors = np.random.uniform(0, 255, size=(80, 3))
#
#         self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)
#
#     def load_class_names(self, classes_path="dnn_model/classes.txt"):
#         with open(classes_path, "r") as file_object:
#             for class_name in file_object.readlines():
#                 class_name = class_name.strip()
#                 self.classes.append(class_name)
#         self.colors = np.random.uniform(0, 255, size=(80, 3))
#         return self.classes
#
#     def detect(self, frame):
#         return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
#
#
# # Initialize Object Detection
# od = ObjectDetection()
#
# video_path = r"B:\Videos\los_angeles.mp4"
# cap = cv2.VideoCapture(video_path)
#
# # Initialize count
# count = 0
# center_points_prev_frame = []
# tracking_objects = {}
# track_id = 0
#
# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (600, 600))
#     count += 1
#     if not ret:
#         break
#
#     # Point current frame
#     center_points_cur_frame = []
#
#     # Detect objects on frame
#     (class_ids, scores, boxes) = od.detect(frame)
#
#     # Manually select the target object by drawing a box around it
#     if count == 1:
#         bbox = cv2.selectROI("Select Object to Track", frame)
#         (x, y, w, h) = bbox
#         tracking_objects[track_id] = ((x + x + w) // 2, (y + y + h) // 2)
#         track_id += 1
#
#     for box in boxes:
#         (x, y, w, h) = box
#         cx = int((x + x + w) / 2)
#         cy = int((y + y + h) / 2)
#         center_points_cur_frame.append((cx, cy))
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Tracking logic
#     for object_id, pt in tracking_objects.items():
#         cv2.circle(frame, pt, 5, (0, 0, 255), -1)
#         cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
#
#     print("Tracking objects")
#     print(tracking_objects)
#
#     print("CUR FRAME LEFT PTS")
#     print(center_points_cur_frame)
#
#     cv2.imshow("Frame", frame)
#
#     if cv2.waitKey(50) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg"):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="dnn_model/classes.txt"):

        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)



# Initialize Object Detection
od = ObjectDetection()

video_path = r"B:\Videos\los_angeles.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize tracking variables
selected_box = None
tracker = cv2.TrackerCSRT_create()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if selected_box is None:
        # Allow user to draw a bounding box around the target object
        bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False)
        tracker.init(frame, bbox)
        selected_box = bbox

    # Update the tracker
    ok, bbox = tracker.update(frame)

    # Draw bounding box if tracking is successful
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

    cv2.imshow("Object Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()