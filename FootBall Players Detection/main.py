import cv2
import os
import numpy as np

# Reading the video
vidcap = cv2.VideoCapture('B:\\B-DRIVE-Data\\Videos\\cutvideo.mp4')
success, image = vidcap.read()

# Define the codec and create a VideoWriter object to save the video
output_dir = 'B:\\B-DRIVE-Data\\6th Semister\\Computer Vision\\CV CODES\\FootBall Player Recognition'
output_path = os.path.join(output_dir, 'output_video.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
fps = int(vidcap.get(cv2.CAP_PROP_FPS))
frame_size = (1200, 700)  # Resized frame size
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

idx = 0

# Read the video frame by frame
while success:
    # Convert into HSV image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_red = np.array([0, 31, 255])
    upper_red = np.array([176, 255, 255])
    lower_white = np.array([0, 0, 0])
    upper_white = np.array([0, 0, 255])

    # Masking and processing
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(image, image, mask=mask)
    res_gray = cv2.cvtColor(cv2.cvtColor(res, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

    # Morphological operations
    kernel = np.ones((13, 13), np.uint8)
    thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Detect players
        if (h >= (1.5) * w) and (w > 15 and h >= 15):
            idx += 1
            player_img = image[y:y + h, x:x + w]
            player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)

            # Blue jersey detection
            mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
            res1 = cv2.cvtColor(cv2.cvtColor(cv2.bitwise_and(player_img, player_img, mask=mask1), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
            nzCount = cv2.countNonZero(res1)

            # Red jersey detection
            mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
            res2 = cv2.cvtColor(cv2.cvtColor(cv2.bitwise_and(player_img, player_img, mask=mask2), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
            nzCountred = cv2.countNonZero(res2)

            if nzCount >= 20:
                cv2.putText(image, 'Team A', (x - 2, y - 2), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            if nzCountred >= 20:
                cv2.putText(image, 'TEAM B', (x - 2, y - 2), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Detect the football
        if (1 <= h <= 30 and 1 <= w <= 30):
            player_img = image[y:y + h, x:x + w]
            player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
            res1 = cv2.cvtColor(cv2.cvtColor(cv2.bitwise_and(player_img, player_img, mask=mask1), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
            nzCount = cv2.countNonZero(res1)

            if nzCount >= 3:
                cv2.putText(image, 'football', (x - 2, y - 2), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Resize the image to 1200x700
    image_resized = cv2.resize(image, frame_size)

    # Write the frame to the output video
    out.write(image_resized)

    # Display the resized video
    cv2.imshow('Match Detection', image_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success, image = vidcap.read()

# Release resources
vidcap.release()
out.release()
cv2.destroyAllWindows()
