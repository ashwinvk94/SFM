#!/usr/bin/env python2
"""
Wrapper.py

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)

Description:
Main function which integrates the total traditional sfm pipeline
metioned in the below link

https://cmsc733.github.io/2019/proj/pfinal/
"""
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from featureMatching import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from EssentialMatrixFromFundamentalMatrix import findEssentialMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose
from NonlinearTriangulation import NonlinearTriangulation

# Camera Intrinsic Matrix
K = np.array(
    [[568.996140852, 0, 643.21055941], [0, 568.988362396, 477.982801038], [0, 0, 1]]
)

n_images = 6
img1 = 1
img2 = 2


def main():

    DataPath = "./Data/"
    visualize = True

    # pts1, pts2 = featureMatching(DataPath, visualize, img1, img2)
    F, pts1, pts2 = featureMatchingcv2(DataPath, visualize, img1, img2)
    print(F)
    # F = EstimateFundamentalMatrix(np.float32(pts1), np.float32(pts2))
    # print(F)
    print("fundamental matrix", F)

    E = findEssentialMatrix(F, K)
    print("essential matrix : ", E)
    # test from here
    Rs, Cs = ExtractCameraPose(E)

    # chrality check and linear triangulation
    R, C, X = DisambiguateCameraPose(Rs, Cs, pts1, pts2, K, DataPath, img1, img2)
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    ax.set_xlim3d(-50, 50)
    ax.set_ylim3d(-50, 50)
    ax.set_zlim3d(0, 50)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])

    print("running non linear triangulation")
    X = NonlinearTriangulation(K, pts1, pts2, X, R, C)
    fig2 = plt.figure()
    ax = Axes3D(fig2)
    ax.set_xlim3d(-50, 50)
    ax.set_ylim3d(-50, 50)
    ax.set_zlim3d(0, 50)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    plt.show()


if __name__ == "__main__":
    main()
