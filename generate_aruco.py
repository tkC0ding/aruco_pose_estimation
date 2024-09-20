import cv2
import argparse
import numpy as np
import sys

ap = argparse.ArgumentParser()

ap.add_argument('-o', '--output', required=True, help='Output directory to save the aruco markers')
ap.add_argument('-i', '--id', type=int, required=True, help='id of the generated aruco marker')
ap.add_argument('-t', '--type', type=str, required=True, help='Type of aruco dictonary to use', default='DICT_ARUCO_ORIGINAL')

args = vars(ap.parse_args())

ARUCO_DICT = {
    'DICT_ARUCO_ORIGINAL' : cv2.aruco.DICT_ARUCO_ORIGINAL,
    'DICT_4X4_100' : cv2.aruco.DICT_4X4_100,
    'DICT_4X4_1000' : cv2.aruco.DICT_4X4_1000,
    'DICT_4X4_250' : cv2.aruco.DICT_4X4_250,
    'DICT_4X4_50' : cv2.aruco.DICT_4X4_50,
    'DICT_5X5_100' : cv2.aruco.DICT_5X5_100,
    'DICT_5X5_1000' : cv2.aruco.DICT_5X5_1000,
    'DICT_5X5_250' : cv2.aruco.DICT_5X5_250,
    'DICT_5X5_50' : cv2.aruco.DICT_5X5_50,
    'DICT_6X6_100' : cv2.aruco.DICT_6X6_100,
    'DICT_6X6_1000' : cv2.aruco.DICT_6X6_1000,
    'DICT_6X6_250' : cv2.aruco.DICT_6X6_250,
    'DICT_6X6_50' : cv2.aruco.DICT_6X6_50,
    'DICT_7X7_100' : cv2.aruco.DICT_7X7_100,
    'DICT_7X7_1000' : cv2.aruco.DICT_7X7_1000,
    'DICT_7X7_250' : cv2.aruco.DICT_7X7_250,
    'DICT_7X7_50' : cv2.aruco.DICT_7X7_50
}

if(ARUCO_DICT.get(args['type'], None) == None):
    print('invalid dictonary!')
    sys.exit(0)


dictonary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args['type']])
img = np.zeros((300, 300, 1), dtype='uint8')
cv2.aruco.generateImageMarker(dictonary, args['id'], 300, img, 1)

cv2.imwrite(args['output'], img)
cv2.imshow('aruco', img)
if cv2.waitKey(0) & 0xFF == ord(q):
    cv2.destroyAllWindows()