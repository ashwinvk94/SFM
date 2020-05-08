#!/usr/bin/env python2
'''
LinearTriangulation

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)

Description:
This functtion estimates the 3D position of a matched pixel given
pixel coordinates in 2 images, intrinsic and extrinsic calibration parameters

Section 1.3.5
https://cmsc733.github.io/2019/proj/pfinal/

'''


def LinearTriangulation(pt1, pt2, K, R, C):
    inCal = np.concatenate((R, C), axis=1)

    print('intrinsic calibration')
    print(inCal)

    p = np.matmul(K, inCal)

    p1 = p[0, :]
    p2 = p[1, :]
    p3 = p[2, :]

    x1 = pt1[0]
    y1 = pt1[1]

    x2 = pt2[0]
    y2 = pt2[1]

    r1 = y1 * p3 - p2
    r2 = p1 - x1 * p3
    r3 = y2 * p3 - p2
    r4 = p1 - x2 * p3

    solMat = np.concatenate((r1, r2, r3, r4), axis=1)

    U, _, Vt = numpy.linalg.svd(E)

    X = Vt[-1, :]

    return (np.array([X[0] / X[3], X[1] / X[3], X[2] / X[3]]))
