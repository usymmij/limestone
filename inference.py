import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = keras.models.load_model('olive-cosmos.h5')
cap = cv2.VideoCapture("testvid.mp4")
canvas = np.ones((900,1600), np.uint8)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    ret, img = cap.read()
    while (cap.isOpened()):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(np.rot90(img),1)
        cv2.imshow("vid",image)
        cv2.waitKey(1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        if not results.multi_hand_landmarks:
            print("nohands")
            continue
        h_landmarks = results.multi_hand_landmarks[0]

        skeleton = []
        for dp in h_landmarks.landmark:
            landmark = [dp.x, dp.y, dp.z]
            skeleton.append(landmark)

        skeleton = np.asarray([skeleton])

        dot = model.predict(skeleton)
        print(dot)
        canvas = cv2.rectangle(canvas, (int(dot[0][0]*80)+120, int(dot[0][1]*80)+120),
            (int(dot[0][0]*80)+121, int(dot[0][1]*80)+121),
            255,5)
        ret, img = cap.read()
        cv2.imshow("result", canvas)
        cv2.waitKey(0)

