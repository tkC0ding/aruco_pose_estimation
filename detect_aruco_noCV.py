import cv2  # using cv only for starting video feed : ) nothing else
import numpy as np # does everything else other than starting the video : )
from numba import jit, prange
from scipy.signal import convolve2d


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

def gaussian_filter(image, kernel_size, sigma=1):

    halfway = kernel_size//2

    x_coordinates, y_coordinates = np.meshgrid(np.arange(-halfway, halfway+1), np.arange(-halfway, halfway+1))

    kernel = (1/(2*np.pi*sigma*sigma))*np.exp(-((x_coordinates**2) + (y_coordinates**2))/(2*sigma*sigma))

    kernel = kernel/np.sum(kernel)

    filtered_img = convolve2d(image, kernel)

    return(filtered_img)

def laplacian(image):
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ]) #4 neighbour laplacian

    filtered_img = convolve2d(image, kernel)
    return(filtered_img)

while True:
    _, frame = cap.read()

    img = frame[:, ::-1, :]/255 # flip the image
    
    img_gray = (0.0722*img[:, :, 0]) + (0.7152*img[:, :, 1]) + (0.2126*img[:, :, 2]) #BGR, coverting BGR to GRAY image

    img_gray = undistort(img_gray, camera_matrix, dist_coeffs) # undistorting the image

    blurred_img = gaussian_filter(img_gray, 5, 8)

    edges = laplacian(blurred_img)*255

    cv2.imshow("laplacian", edges.astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()