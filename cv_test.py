import numpy as np
import cv2 as cv


for i in range(1000):
    img = np.random.random((1024,720,3))
    cv.imshow('image',img)
    key = cv.waitKey(0)
    print(key)
    if key == ord('q'):
        break
