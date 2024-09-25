import cv2
from edge_detection import canny

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

while True:
    _, frame = cap.read()

    final_edges = canny(frame, 3, 1)

    cv2.imshow("edge detection using canny", final_edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()