import cv2  # using cv only for starting video feed : ) nothing else
import numpy as np # does everything else other than starting the video : )
from numba import jit, prange
from scipy.ndimage import convolve


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

calibration_data = np.load('calibration_files/calibration.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeff']

def undistort(image, camera_matrix, dist_coeff):
    h,w = image.shape[:2]
    new_image = np.zeros_like(image) #create a base image and then later on add the pixel intensities

    fx, fy, ox, oy = camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]

    x_distorted, y_distorted = np.meshgrid(np.arange(w), np.arange(h))
    x_distorted = (x_distorted - ox)/fx
    y_distorted = (y_distorted - oy)/fy

    u, v = find_uv(x_distorted, y_distorted, camera_matrix, dist_coeff)

    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    new_image = image[v,u]

    return new_image

@jit(nopython=True, parallel=True)
def find_uv(x_distorted, y_distorted, camera_matrix, dist_coeff):

    fx, fy, ox, oy = camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]
    k1, k2, p1, p2, k3 = dist_coeff[0,0], dist_coeff[0,1], dist_coeff[0,2], dist_coeff[0,3], dist_coeff[0,4]

    x_guess = x_distorted
    y_guess = y_distorted

    for _ in range(2):
        r2 = (x_guess**2) + (y_guess**2)
        radial_distortion = 1 + (k1*r2) + (k2*(r2**2)) + (k3*(r2**3))
        x_new = (x_distorted - ((2*p1*x_guess*y_guess) + (p2*(r2 + (2*(x_guess**2))))))/radial_distortion
        y_new = (y_distorted - ((p1*(r2 + (2*(y_guess**2)))) + (2*p2*x_guess*y_guess)))/radial_distortion
        x_guess = x_new
        y_guess = y_new
    
    x_undistorted = x_guess
    y_undistorted = y_guess

    u = (fx*x_undistorted + ox).astype(np.int32)
    v = (fy*y_undistorted + oy).astype(np.int32)
    
    return (u,v)

def sobel(image):
    Gx = np.array([
        [-3, 0, 3],
        [-10, 0, 10],
        [-3, 0, 3]
    ]).astype(np.uint8)

    Gy = np.array([
        [-3, -10, -3],
        [0, 0, 0],
        [3, 10, 3]
    ]).astype(np.uint8)

    Ix = convolve(image, Gx).astype(np.uint8)
    Iy = convolve(image, Gy).astype(np.uint8)

    G = np.sqrt((Ix**2) + (Iy**2)).astype(np.uint8)
    theta = np.rad2deg(np.arctan2(Iy, Ix))

    theta[theta < 0] += 180

    return (G, theta)

while True:
    _, frame = cap.read()

    img = frame[:, ::-1, :]/255 # flip the image
    
    img_gray = (0.0722*img[:, :, 0]) + (0.7152*img[:, :, 1]) + (0.2126*img[:, :, 2]) #BGR, coverting BGR to GRAY image

    img_gray = undistort(img_gray, camera_matrix, dist_coeffs) # undistorting the image

    binary_img = np.astype(np.where(img_gray > 0.5, 1, 0), np.uint8) # applying thresholding to detect black and white easily

    gradient, theta = sobel(binary_img) # applying the sobel operator to detect edges in the binary image

    cv2.imshow("gradient", gradient*255)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()