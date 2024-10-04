import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def DLT(object_points, image_points):
    n = object_points.shape[0]
    A = []
    for i in range(n):
        X, Y, Z = object_points[i][0], object_points[i][1], object_points[i][2]
        u, v = image_points[i][0], image_points[i][1]

        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    A = np.array(A)

    P = np.reshape(Vt[-1, :], (3,4))
    return P

def decompose_projection_matrix(P):
    P_norm = P/P[2, 3]

    M = P_norm[:, :3]

    K, R = rq(M)

    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    K /= K[2,2]
    T = T @ K

    t = np.linalg.inv(K) @ P_norm[:, 3]

    K = np.hstack((K, np.zeros((K.shape[0], 1))))
    R = np.hstack((R, t.reshape((3, 1))))
    R = np.vstack((R, np.array([0,0,0,1])))

    return K, R

def project_points(object_points, R, camera_matrix):
    projected_points = []
    for point in object_points:
        camera_point = np.dot(R, point)
        x = np.dot(camera_matrix, camera_point)
        x = x/x[2]
        projected_points.append(x[:2])
    return np.array(projected_points)

def SolvePnP(object_points, image_points, camera_matrix):

    P = DLT(object_points, image_points)
    K, R, t = decompose_projection_matrix(P)

    def reprojection_error(params, object_points, image_points, camera_matrix):
        R_vec = params[:9].reshape(3, 3)
        t_vec = params[9:]
        projected_points = project_points(object_points, R_vec, t_vec, camera_matrix)
        return (projected_points - image_points).ravel()
      
    initial_params = np.hstack((R.ravel(), t))

    result = least_squares(reprojection_error, initial_params, args=(object_points, image_points, camera_matrix))

    R_refined = result.x[:9].reshape((3, 3))
    t_refined = result.x[9:].reshape((3,1))

    r = Rotation.from_matrix(R_refined)
    R_refined = r.as_rotvec().reshape((3, 1))

    return R_refined, t_refined