#!/usr/bin/env python2
"""
LinearTriangulation

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)

Description:
This functtion estimates the 3D position of a matched pixel given
pixel coordinates in 2 images, intrinsic and extrinsic calibration parameters

Section 1.3.5
https://cmsc733.github.io/2019/proj/pfinal/

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def skew(x):
#     return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])


def LinearTriangulation(K, C, R, X1, X2, DataPath, imgn1, imgn2):
    img1 = cv2.imread(DataPath + str(imgn1) + ".jpg")
    img2 = cv2.imread(DataPath + str(imgn2) + ".jpg")
    I = np.identity(3)
    sz = X1.shape[0]
    C = np.reshape(C, (3, 1))
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    # P = np.dot(K, np.hstack((R, C)))
    P1t = np.reshape(P[0], (1, 4))
    P2t = np.reshape(P[1], (1, 4))
    P3t = np.reshape(P[2], (1, 4))
    # print(P1t)
    # print(P1t.shape)
    # print(P2t)
    # print(P3t)
    # print(X1.shape)
    # print('\n')

    #     print(P2.shape)
    # X1 = np.hstack((x1, np.ones((sz, 1))))
    # X2 = np.hstack((x2, np.ones((sz, 1))))

    X_arr = np.zeros((sz, 3))

    for i in range(sz):
        # print(X1[i])
        # print(X2[i])
        x1 = X1[i, 0]
        y1 = X1[i, 1]
        x2 = X2[i, 0]
        y2 = X2[i, 1]

        r1 = y1 * P3t - P2t
        r2 = P1t - x1 * P3t
        r3 = y2 * P3t - P2t
        r4 = P1t - x2 * P3t

        solMat = np.concatenate((r1, r2, r3, r4), axis=1)

        U, _, Vt = np.linalg.svd(solMat)

        X = Vt[-1, :]

        # A = np.vstack((np.dot(skew1, P), np.dot(skew2, P)))
        # _, _, v = np.linalg.svd(A)
        # x = v[-1] / v[-1, -1]
        # x = np.reshape(x, (len(x), -1))
        X_arr[i, :] = np.array([X[0] / X[3], X[1] / X[3], X[2] / X[3]])

        # # drawing circles
        # fig1 = plt.figure()
        # ax = Axes3D(fig1)
        # ax.set_xlim3d(-50, 50)
        # ax.set_ylim3d(-50, 50)
        # ax.set_zlim3d(0, 50)
        # ax.scatter(X_arr[:, 0], X_arr[:, 1], X_arr[:, 2])
        # print(np.array([X[0] / X[3], X[1] / X[3], X[2] / X[3]]))
        # cv2.circle(img1, (int(x1), int(y1)), 3, (0, 255, 0), -1)
        # cv2.circle(img2, (int(x2), int(y2)), 3, (0, 255, 0), -1)
        # cv2.imshow("img1", img1)
        # cv2.imshow("img2", img2)
        # cv2.waitKey(1)
        # plt.show()

    cv2.destroyAllWindows()
    return X_arr


"""
def LinearTriangulation(pt1, pt2, K, R, C):
    Ct = np.reshape(C, (3, 1))
    print(R.shape)
    print(Ct.shape)

    extCal = np.concatenate((R, Ct), axis=1)

    print('extrinsic calibration')
    print(extCal)

    p = np.matmul(K, extCal)

    p1 = p[0, :]
    p2 = p[1, :]
    p3 = p[2, :]

    x1 = pt1[0]
    y1 = pt1[1]

    x2 = pt2[0]
    y2 = pt2[1]

    r1 = y1 * p3 - p2
    R = p1 - x1 * p3
    r3 = y2 * p3 - p2
    r4 = p1 - x2 * p3

    r1 = np.reshape(r1, (4, 1))
    r2 = np.reshape(r2, (4, 1))
    r3 = np.reshape(r3, (4, 1))
    r4 = np.reshape(r4, (4, 1))

    solMat = np.concatenate((r1, r2, r3, r4), axis=1)

    U, _, Vt = np.linalg.svd(solMat)

    X = Vt[-1, :]
    print(X)

    return (np.array([X[0] / X[3], X[1] / X[3], X[2] / X[3]]))
"""
