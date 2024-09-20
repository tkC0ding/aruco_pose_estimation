import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

dictonary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
detector_parmas = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictonary, detector_parmas)

while True:
    _, frame = cap.read()

    corners, ids, reject = detector.detectMarkers(frame)
    
    if(len(corners) > 0):
        ids = ids.flatten().astype(int)
        for (c,id) in zip(corners,ids):
            c = c.reshape((4, 2)).astype(int)
            (top_left, top_right, bottom_right, bottom_left) = c

            top_left = (top_left[0], top_left[1])
            top_right = (top_right[0], top_right[1])
            bottom_right = (bottom_right[0], bottom_right[1])
            bottom_left = (bottom_left[0], bottom_left[1])

            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_left, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, bottom_right, (0, 255, 0), 2)

            cx = (top_left[0] + top_right[0])//2
            cy = (top_left[1] + top_right[1])//2

            cv2.circle(frame, (cx, cy), 2, (0,0,255), -1)
            

        cv2.imshow('frame', frame)
        if(cv2.waitKey(0) & 0xFF == ord('q')):
            break

cap.release()
cv2.destroyAllWindows()