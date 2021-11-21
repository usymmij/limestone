import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

data= []
lbl = []
IMAGE_FILES = ["ic/1.JPEG","ic/2.JPEG","ic/3.JPEG","ic/4.JPEG","ic/5.JPEG","ic/6.JPEG","ic/7.JPEG","ic/8.JPEG","ic/9.JPEG","ic/10.JPEG","ic/11.JPEG","ic/12.JPEG","ic/13.JPEG","ic/14.JPEG","ic/15.JPEG","ic/16.JPEG","ic/17.JPEG","ic/18.JPEG","ic/19.JPEG","ic/20.JPEG","ic/21.JPEG","ic/22.JPEG","ic/23.JPEG","ic/24.JPEG","ic/25.JPEG","ic/26.JPEG","ic/27.JPEG","ic/28.JPEG","ic/29.JPEG","ic/30.JPEG","ic/31.JPEG","ic/32.JPEG","ic/33.JPEG","ic/34.JPEG","ic/35.JPEG","ic/36.JPEG","ic/37.JPEG","ic/38.JPEG","ic/39.JPEG","ic/40.JPEG","ic/41.JPEG","ic/42.JPEG"]
with mp_hands.Hands(
    # remember to set to true for static photos
    static_image_mode=True,
    max_num_hands=1, 
    min_detection_confidence=0.5) as hands:
    for i in range(41):
        if(i==25):
            continue
        image = cv2.imread(IMAGE_FILES[i])
        image = cv2.flip(np.rot90(image), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            h_landmarks = results.multi_hand_landmarks[0]
        else:
            print("an error occurred with the image data")
            continue
        skeleton = []
        for dp in h_landmarks.landmark:
            landmark = [dp.x, dp.y, dp.z]
            skeleton.append(landmark)

        skeleton = np.asarray(skeleton)
        #skeleton = np.rot90(skeleton)
        
        data.append(skeleton)
        lbl.append(i)

data = np.asarray(data)

ind = 0
for lb in lbl:
    # rescale the coordinates so that the width of the screen is 1.0 
    # height is first scaled to 1.0 then scaled by the aspect ratio 
    x = (lb % 20 + 1) / 21
    y = ((int(lb /  20) + 1) / 11)*(9/16) 
    #print(str(x), ",", str(y))
    lbl[ind] = [x,y]
    ind += 1
lbl = np.asarray(lbl)
print(lbl.shape)
print(data.shape)

np.save('data.npy',data, True)
np.save('lbl.npy',lbl, True)