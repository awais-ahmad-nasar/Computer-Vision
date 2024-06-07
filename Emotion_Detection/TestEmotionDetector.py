#.............................FOR VIDEOS....................................#

# import cv2
# import numpy as np
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
#
#
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#
# json_file = open('model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
#
# # Load the model, specifying the custom objects
# emotion_model = model_from_json(
#     loaded_model_json,
#     custom_objects={
#         'Sequential': Sequential,
#         'InputLayer': InputLayer,
#         'Conv2D': Conv2D,
#         'MaxPooling2D': MaxPooling2D,
#         'Dropout': Dropout,
#         'Flatten': Flatten,
#         'Dense': Dense
#     }
# )
# emotion_model.load_weights("model/emotion_model.h5")
# #print("Loaded model from disk")
#
# cap = cv2.VideoCapture(r"B:\images1_detection\EmotionDetectionTesting\video.mp4")
#
# while True:
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (1280, 720))
#     if not ret:
#         break
#
#     face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
#
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#
#         emotion_prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(emotion_prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#
#     cv2.imshow('Emotion Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()



#.............................FOR WEBCAM....................................#

# import cv2
# import numpy as np
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
#
# # Define the emotion dictionary
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#
# # Load json and create model
# json_file = open('model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
#
# # Load the model, specifying the custom objects
# emotion_model = model_from_json(
#     loaded_model_json,
#     custom_objects={
#         'Sequential': Sequential,
#         'InputLayer': InputLayer,
#         'Conv2D': Conv2D,
#         'MaxPooling2D': MaxPooling2D,
#         'Dropout': Dropout,
#         'Flatten': Flatten,
#         'Dense': Dense
#     }
# )
#
# # Load weights into new model
# emotion_model.load_weights("model/emotion_model.h5")
# print("Loaded model from disk")
#
# # Start the webcam feed
# cap = cv2.VideoCapture(0)  # Change from video file to webcam
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (1280, 720))
#     if not ret:
#         break
#
#     # Convert frame to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Load the face detector
#     face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#
#     # Detect faces in the grayscale frame
#     num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
#
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
#
#         # Predict the emotion
#         emotion_prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(emotion_prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
#                     cv2.LINE_AA)
#
#     # Display the resulting frame
#     cv2.imshow('Emotion Detection', frame)
#
#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# cap.release()
# cv2.destroyAllWindows()
#


#.............................FOR IMAGES....................................#

import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Load the model, specifying the custom objects
emotion_model = model_from_json(
    loaded_model_json,
    custom_objects={
        'Sequential': Sequential,
        'InputLayer': InputLayer,
        'Conv2D': Conv2D,
        'MaxPooling2D': MaxPooling2D,
        'Dropout': Dropout,
        'Flatten': Flatten,
        'Dense': Dense
    }
)

# Load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")


# Function to predict emotion from an image file
def predict_emotion_from_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read image")
        return

    # Resize the image
    img = cv2.resize(img, (900, 650))

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the face detector
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # Detect faces in the grayscale image
    faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray = gray_img[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]

        # Print the predicted emotion
        print(f"Predicted emotion: {emotion}")

        # Display the predicted emotion on the image
        cv2.putText(img, emotion, (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the image with predictions
    cv2.imshow('Emotion Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
#image_path = r"B:\images1_detection\EmotionDetectionTesting\FearfullMan.jpg"
#image_path = r"B:\images1_detection\EmotionDetectionTesting\SurpriseWoman.jpg"
#image_path = r"B:\images1_detection\EmotionDetectionTesting\maninhouse.jpg"

image_path = r"B:\images1_detection\EmotionDetectionTesting\testing.jpg"

predict_emotion_from_image(image_path)
