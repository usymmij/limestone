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
canvas = np.ones((900,1600), np.uint8)
IMAGE_FILES = ["ic/1.JPEG","ic/2.JPEG","ic/3.JPEG","ic/4.JPEG","ic/5.JPEG","ic/6.JPEG","ic/7.JPEG","ic/8.JPEG","ic/9.JPEG","ic/10.JPEG","ic/11.JPEG","ic/12.JPEG","ic/13.JPEG","ic/14.JPEG","ic/15.JPEG","ic/16.JPEG","ic/17.JPEG","ic/18.JPEG","ic/19.JPEG","ic/20.JPEG","ic/21.JPEG","ic/22.JPEG","ic/23.JPEG","ic/24.JPEG","ic/25.JPEG","ic/26.JPEG","ic/27.JPEG","ic/28.JPEG","ic/29.JPEG","ic/30.JPEG","ic/31.JPEG","ic/32.JPEG","ic/33.JPEG","ic/34.JPEG","ic/35.JPEG","ic/36.JPEG","ic/37.JPEG","ic/38.JPEG","ic/39.JPEG","ic/40.JPEG","ic/41.JPEG","ic/42.JPEG"]

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(np.rot90(img),1)
    cv2.imshow("vid",cv2.flip(image,2))
    cv2.waitKey(1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    if not results.multi_hand_landmarks:
        print("nohands")
        exit()
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
    cv2.imshow("result", canvas)
    cv2.waitKey(0)

