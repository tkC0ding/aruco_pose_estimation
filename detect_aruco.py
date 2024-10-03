import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from SOLVEPNP import SolvePnP

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

calibration_data = np.load('calibration_files/calibration.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeff = calibration_data['dist_coeff']

dictonary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictonary, detector_params)

aruco_side_length = 0.14 #m

object_points = np.array([
    [-aruco_side_length/2, aruco_side_length/2, 0],
    [aruco_side_length/2, aruco_side_length/2, 0],
    [aruco_side_length/2, -aruco_side_length/2, 0],
    [-aruco_side_length/2, -aruco_side_length/2, 0]
])

while True:
    _, frame = cap.read()

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (640,480), 1, (640,480))

    frame = cv2.undistort(frame, camera_matrix, dist_coeff, None, new_camera_matrix)
    x,y,w,h = roi
    frame = frame[y:y+h, x:x+w]

    corners, ids, rejected = detector.detectMarkers(frame)

    if(ids is not None):
        rvecs, tvecs = SolvePnP(object_points, corners[0][0], new_camera_matrix)
        if (rvecs.all() != None):
            success = True
        else:
            success = False
        if success:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))

            cv2.drawFrameAxes(frame, new_camera_matrix, dist_coeff, rvecs, tvecs, 0.1, 3)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()