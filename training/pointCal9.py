import cv2
import numpy as np

for i in range(9):
    img = np.ones((1080,1920,3),np.uint8)
    img = img*255

    j = (i % 3)
    k = int(i / 3)
    print("dot"+str(i))
    print(j)
    print(k)

    img = cv2.rectangle(img, ((j+1)*round(1920/4),(k+1)*round(1080/4)),((j+1)*round(1920/4)+1,(k+1)*round(1080/4)+1), 25,5)


    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('image', img)

    cv2.waitKey(0)

cv2.destroyAllWindows()