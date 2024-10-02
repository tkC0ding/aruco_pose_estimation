import numpy as np

def normalize_point(camera_matrix, image_points):
    img_points = np.hstack((image_points, np.ones((image_points.shape[0], 1))))

    K_inv = np.linalg.inv(camera_matrix)

    normalized_img_points = np.dot(K_inv, img_points.T).T

    return normalized_img_points[:, :2]