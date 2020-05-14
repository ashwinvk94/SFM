#!/usr/bin/env python2
"""
DisambiguateCameraPose

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)

Description:
Implements chirality check for the 4 estimated extrinsic
calibration parameters. Returns true if the chirality
check is passed.

Section 1.3.5
https://cmsc733.github.io/2019/proj/pfinal/

http://cseweb.ucsd.edu/classes/fa11/cse252A-a/hw3/linear_triangulation.pdf
"""

import numpy as np
from LinearTriangulation import LinearTriangulation
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def DisambiguateCameraPose(Rs, Cs, pts1, pts2, K, DataPath, img1, img2):
    nPoints = pts1.shape[0]
    count = np.zeros(4)
    fig = plt.figure()
    ax = Axes3D(fig)
    # # ax.set_xlim3d(-50, 50)
    # # ax.set_ylim3d(-50, 50)
    # # ax.set_zlim3d(-50, 50)
    for k in range(4):
        R = Rs[k]
        C = Cs[k]
        X = LinearTriangulation(K, C, R, pts1, pts2, DataPath, img1, img2)
        # print(X)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2])

        r3 = R[2, :]
        for ind in range(X.shape[0]):
            checlVal = np.matmul(r3, X[ind, :] - C)
            if checlVal > 0:
                count[k] += 1
    # UNCOMMENT THIS LINE TO PLOT the 3D plots
    plt.show()
    ind = np.argmax(count)
    X = LinearTriangulation(K, Cs[ind], Rs[ind], pts1, pts2, DataPath, img1, img2)
    return Rs[ind], Cs[ind], X
