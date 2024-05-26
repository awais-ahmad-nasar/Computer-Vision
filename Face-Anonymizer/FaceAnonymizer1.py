
#<><><><><><><><><><><><><><><><> PART A <><><><><><><><><><><><><><><><>

# import cv2
# import mediapipe as mp
#
# # Read image
#
# img=cv2.imread(r"B:\images2_detection\person.jpg")
#
# H , W , _ = img.shape
# #Detect Faces
#
# mp_face_detection = mp.solutions.face_detection
# with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     out = face_detection.process(img_rgb)
#
#     if out.detections is not None:
#         for detection in out.detections:
#             location_data = detection.location_data
#             bbox = location_data.relative_bounding_box
#
#             x1,y1,w,h = bbox.xmin, bbox.ymin,bbox.width,bbox.height
#
#             x1=int(x1*W)
#             y1 = int(y1 * H)
#             w = int(w * W)
#             h = int(h * H)
#
#             img=cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),5)
#
#     cv2.imshow("Image",img)
#     cv2.waitKey(0)


#<><><><><><><><><><><><><><><><> PART B <><><><><><><><><><><><><><><><>
##################################...BLUR FACE...######################################################
#
# import cv2
# import  mediapipe as mp
#
# # Read image
# image_path=r"E:\images2_detection\person.jpg"
# img=cv2.imread(image_path)
#
# H , W , _ = img.shape
#
#
# # detect then Blur Faces
#
# mp_face_detection = mp.solutions.face_detection
# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     out = face_detection.process(img_rgb)
#
#     if out.detections is not None:
#         for detection in out.detections:
#             location_data = detection.location_data
#             bbox = location_data.relative_bounding_box
#
#             x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
#
#             x1 = int(x1 * W)
#             y1 = int(y1 * H)
#             w = int(w * W)
#             h = int(h * H)
#
#             #blur face
#             img[ y1 : y1+h , x1 : x1+w , : ] = cv2.blur(
#                 img[ y1 : y1+h , x1 : x1+w , : ],(20,20))
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(0)


#<><><><><><><><><><><><><><><><> PART C <><><><><><><><><><><><><><><><>
##################################...SAVE BLUR...######################################################
#
# import cv2
# import  mediapipe as mp
# import os
#
# # Read image
# image_path=r"E:\images2_detection\person.jpg"
# img=cv2.imread(image_path)
#
# output_directory='./output'
# if os.path.exists(output_directory):
#     os.makedirs(output_directory)
#
#
# H , W , _ = img.shape
#
# # detect then Blur Faces
#
# mp_face_detection = mp.solutions.face_detection
# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     out = face_detection.process(img_rgb)
#
#     if out.detections is not None:
#         for detection in out.detections:
#             location_data = detection.location_data
#             bbox = location_data.relative_bounding_box
#
#             x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
#
#             x1 = int(x1 * W)
#             y1 = int(y1 * H)
#             w = int(w * W)
#             h = int(h * H)
#
#             #blur face
#             img[ y1 : y1+h , x1 : x1+w , : ] = cv2.blur(
#                 img[ y1 : y1+h , x1 : x1+w , : ],(20,20))
#
#
#
# # Save Image
# cv2.imwrite(("E:\\images2_detection\OutputOfPerson.jpg"),img)