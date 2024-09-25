import cv2  # using cv only for starting video feed : ) nothing else
import numpy as np # does everything else other than starting the video : )
from numba import jit
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

def sobel(image):
    Gx = np.array([
        [-1, 0 , 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    Gy = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    Ix = convolve2d(image, Gx)
    Iy = convolve2d(image, Gy)

    G = np.sqrt((Ix**2) + (Iy**2))
    G = G/G.max()
    theta = np.arctan2(Iy, Ix)

    theta = np.rad2deg(theta)

    return(G, theta)

def non_max_suppression(G, theta):
    theta[theta < 0] += 180

    mask_0 = ((0 <= theta) & (theta < 22.5)) | ((157.5 <= theta) & (theta < 180))
    mask_45 = (22.5 <= theta) & (theta < 67.5)
    mask_90 = (67.5 <= theta) & (theta < 112.5)
    mask_135 = (112.5 <= theta) & (theta < 157.5)

    shift_0 = np.roll(G, 1, 1)
    shift_180 = np.roll(G, -1, 1)

    shift_45_pos = np.roll(shift_0, -1, 0)
    shift_45_neg = np.roll(shift_180, 1, 0)

    shift_90_pos = np.roll(G, -1, 0)
    shift_90_neg = np.roll(G, 1, 0)

    shift_135_pos = np.roll(shift_180, -1, 0)
    shift_135_neg = np.roll(shift_0, 1, 0)

    condition = ((mask_0 & (G >= shift_0) & (G >= shift_180))|
                 (mask_45 & (G >= shift_45_pos) & (G >= shift_45_neg))|
                 (mask_90 & (G >= shift_90_pos) & (G >= shift_90_neg))|
                 (mask_135 & (G >= shift_135_pos) & (G >= shift_135_neg))
                 )
    
    nms_img = np.where(condition, G, 0)
    return(nms_img)

while True:
    _, frame = cap.read()

    img = frame[:, ::-1, :]/255 # flip the image
    
    img_gray = (0.0722*img[:, :, 0]) + (0.7152*img[:, :, 1]) + (0.2126*img[:, :, 2]) #BGR, coverting BGR to GRAY image

    img_gray = undistort(img_gray, camera_matrix, dist_coeffs) # undistorting the image

    blurred_img = gaussian_filter(img_gray, 3, 1)

    gradients, theta = sobel(img_gray)

    nms_image = non_max_suppression(gradients, theta) * 255

    cv2.imshow("nms", nms_image.astype(np.uint8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()