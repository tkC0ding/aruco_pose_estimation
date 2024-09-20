import cv2
import numpy as np

dictonary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
img = np.zeros((300, 300, 1), dtype='uint8')
cv2.aruco.generateImageMarker(dictonary, 7, 300, img, 1)

cv2.imwrite('tags/ARUCO_5X5_id7.png', img)
cv2.imshow('aruco', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()