import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        success, rvecs, tvecs = cv2.solvePnP(object_points, corners[0][0], new_camera_matrix, dist_coeff)
        if success:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))

            rotation_matrix, _ = cv2.Rodrigues(rvecs)
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()

            rotation_transform_x = quat[0]
            rotation_transform_y = quat[1]
            rotation_transform_z = quat[2]
            rotation_transform_w = quat[3]


cap.release()
cv2.destroyAllWindows()