import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import sys

sys.dont_write_bytecode = True
"""
Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)

https://cmsc733.github.io/2019/proj/pfinal/
"""


def rotate(points, rot_vecs):
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(
        v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices],
                          camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices,
                               point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


def BundleAdjustment(Cset, Rset, X, K, points_2d, camera_indices, recon_bin,
                     V_bundle):
    f = K[1, 1]
    camera_params = []
    # camera_indices  = np.array(r_indx[1:])
    point_indices, _ = np.where(recon_bin == 1)
    V = V_bundle[point_indices, :]
    points_3d = X[point_indices, :]
    for C0, R0 in zip(Cset, Rset):
        q_temp = Rscipy.from_dcm(R0)
        Q0 = q_temp.as_rotvec()
        params = [Q0[0], Q0[1], Q0[2], C0[0], C0[1], C0[2], f, 0, 0]
        camera_params.append(params)

    camera_params = np.reshape(camera_params, (-1, 9))

    n_cameras = camera_params.shape[0]

    assert len(Cset) == n_cameras, "length not matched"

    n_points = points_3d.shape[0]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    opt = False

    if (opt):
        A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices,
                                       point_indices)
        # print(camera_params.ravel)
        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

        res = least_squares(fun,
                            x0,
                            jac_sparsity=A,
                            verbose=2,
                            x_scale='jac',
                            ftol=1e-4,
                            method='trf',
                            args=(n_cameras, n_points, camera_indices,
                                  point_indices, points_2d))

        parameters = res.x

        camera_p = np.reshape(parameters[0:camera_params.size], (n_cameras, 9))

        X = np.reshape(parameters[camera_params.size:], (n_points, 3))

        for i in range(n_cameras):
            Q0[0] = camera_p[i, 0]
            Q0[1] = camera_p[i, 1]
            Q0[2] = camera_p[i, 2]
            C0[0] = camera_p[i, 2]
            C0[1] = camera_p[i, 2]
            C0[2] = camera_p[i, 6]
            r_temp = Rscipy.from_rotvec([Q0[0], Q0[1], Q0[2]])
            Rset[i] = r_temp.as_dcm()
            Cset[i] = [C0[0], C0[1], C0[2]]

    return Rset, Cset, X
