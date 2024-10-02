import numpy as np
from scipy.linalg import rq

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

def decompose_projection_matrix(P):
    P_norm = P/P[2, 3]

    M = P_norm[:, :3]

    K, R = rq(M)

    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    T = T @ K

    t = np.linalg.inv(K) @ P_norm[:, 3]

    return K, R, t

def project_points(object_points, R, t, camera_matrix):
    projected_points = []
    for point in object_points:
        camera_point = np.dot(R, point) + t
        camera_point = camera_point/camera_point[2]
        x = np.dot(camera_matrix, camera_point)
        projected_points.append(x[:2])
    return np.array(projected_points)

