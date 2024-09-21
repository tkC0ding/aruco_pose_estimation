import cv2
import numpy as np

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

while True:
    _, frame = cap.read()

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (640,480), 1, (640,480))

    frame = cv2.undistort(frame, camera_matrix, dist_coeff, None, new_camera_matrix)
    x,y,w,h = roi
    frame = frame[y:y+h, x:x+w]

    corners, ids, rejected = detector.detectMarkers(frame)

    if(len(corners) > 0):
        ids = ids.flatten().astype(int)
        for c,i in zip(corners, ids):
            top_left, top_right, bottom_right, bottom_left = c.reshape((4,2)).astype(int)

            top_left = (top_left[0], top_left[1])
            top_right = (top_right[0], top_right[1])
            bottom_right = (bottom_right[0], bottom_right[1])
            bottom_left = (bottom_left[0], bottom_left[1])

            cv2.line(frame, top_left, top_right, (0,255,0), 2)
            cv2.line(frame, top_right, bottom_right, (0,255,0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0,255,0), 2)
            cv2.line(frame, bottom_left, top_left, (0,255,0), 2)

            cx = (top_left[0] + bottom_right[0])//2
            cy = (top_left[1] + bottom_right[1])//2

            cv2.circle(frame, (cx,cy), 4, (0,0,255), -1)
        
        cv2.imshow("aruco", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()