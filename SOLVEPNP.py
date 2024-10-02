import numpy as np

def normalize_point(camera_matrix, image_points):
    img_points = np.hstack((image_points, np.ones((image_points.shape[0], 1))))

    K_inv = np.linalg.inv(camera_matrix)

    normalized_img_points = np.dot(K_inv, img_points.T).T

    return normalized_img_points[:, :2]

def DLT(object_points, image_points):
    n = object_points.shape[0]
    A = []
    for i in range(n):
        X, Y, Z = object_points[i][0], object_points[i][1], object_points[i][2]
        u, v = image_points[i][0], image_points[i][1]

        A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, u*X, u*Y, u*Z, u])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, v*X, v*Y, v*Z, v])
    A = np.array(A)
    U, sigma, Vt = np.linalg.svd(A)

    P = np.reshape(Vt[-1], (3,4))
    return P