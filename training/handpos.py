import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_FILES = ["ic/1.JPEG","ic/2.JPEG","ic/3.JPEG","ic/4.JPEG","ic/5.JPEG","ic/6.JPEG","ic/7.JPEG","ic/8.JPEG","ic/9.JPEG","ic/10.JPEG","ic/11.JPEG","ic/12.JPEG","ic/13.JPEG","ic/14.JPEG","ic/15.JPEG","ic/16.JPEG","ic/17.JPEG","ic/18.JPEG","ic/19.JPEG","ic/20.JPEG","ic/21.JPEG","ic/22.JPEG","ic/23.JPEG","ic/24.JPEG","ic/25.JPEG","ic/26.JPEG","ic/27.JPEG","ic/28.JPEG","ic/29.JPEG","ic/30.JPEG","ic/31.JPEG","ic/32.JPEG","ic/33.JPEG","ic/34.JPEG","ic/35.JPEG","ic/36.JPEG","ic/37.JPEG","ic/38.JPEG","ic/39.JPEG","ic/40.JPEG","ic/41.JPEG","ic/42.JPEG"]
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = np.rot90(cv2.flip(cv2.imread(file), 1)) 
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow("mp hands", cv2.flip(annotated_image, 1))
    cv2.waitKey(0)
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)