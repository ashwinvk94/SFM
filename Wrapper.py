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
from PnPRANSAC import *
from BundleAdjustment import *
from BuildVisibilityMatrix import *
from NonLinearPnP import *
from DrawCameras import *
from LinearTriangulation import *

# Camera Intrinsic Matrix
K = np.array([[568.996140852, 0, 643.21055941],
              [0, 568.988362396, 477.982801038], [0, 0, 1]])

n_images = 6
img1 = 1
img2 = 2


def main():

    DataPath = "./Data/"
    visualize = True

    pts1, pts2, matches, matchesX, matchesY, PixelVal, outlier_indices = featureMatching(
        DataPath, visualize, img1, img2)
    F = EstimateFundamentalMatrix(np.float32(pts1), np.float32(pts2))
    # This function was used to compare the feature extraction using cv2
    # F, pts1, pts2 = featureMatchingcv2(DataPath, visualize, img1, img2)
    print(F)
    print("fundamental matrix", F)

    E = findEssentialMatrix(F, K)
    print("essential matrix : ", E)
    # test from here
    Rs, Cs = ExtractCameraPose(E)

    # chrality check and linear triangulation
    R, C, X = DisambiguateCameraPose(Rs, Cs, pts1, pts2, K, DataPath, img1,
                                     img2)

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

    #  We have all inliers at this point in matches
    output = np.logical_and(matches[:, img1 - 1], matches[:, img2 - 1])
    outlier = np.logical_and(outlier_indices[:, img1 - 1],
                             outlier_indices[:, img2 - 1])
    outlier_idx = np.where(outlier == True)
    ind, = np.where(output == True)
    rgb_list = PixelVal[ind]

    pts1 = np.hstack((matchesX[ind, img1 - 1].reshape(
        (-1, 1)), matchesY[ind, img1 - 1].reshape((-1, 1))))
    pts2 = np.hstack((matchesX[ind, img2 - 1].reshape(
        (-1, 1)), matchesY[ind, img2 - 1].reshape((-1, 1))))
    best_F = EstimateFundamentalMatrix(np.float32(pts1), np.float32(pts2))

    outlier1 = np.hstack((matchesX[outlier_idx, img1 - 1].reshape(
        (-1, 1)), matchesY[outlier_idx, img1 - 1].reshape((-1, 1))))
    outlier2 = np.hstack((matchesX[outlier_idx, img2 - 1].reshape(
        (-1, 1)), matchesY[outlier_idx, img2 - 1].reshape((-1, 1))))

    recon_bin = np.zeros((matches.shape[0], 1))
    X_points = np.zeros((matches.shape[0], 3))
    Visibility = np.zeros((matches.shape[0], n_images))

    recon_bin[ind] = 1
    X_points[ind, :] = X
    Visibility[ind, img1 - 1] = 1
    Visibility[ind, img2 - 1] = 1

    Cset = []
    Rset = []

    Cset.append(C)
    Rset.append(R)

    r_indx = [img1, img2]

    for i in range(0, n_images):

        if (np.isin(r_indx, i)[0]):
            continue

        output = np.logical_and(recon_bin, matches[:, i].reshape((-1, 1)))
        ind, _ = np.where(output == True)
        if (len(ind) < 8):
            continue

        X = X_points[ind, :]
        x = np.transpose([matchesX[ind, i], matchesY[ind, i]])

        C, R = PnPRANSAC(X, x, K)
        C, R = NonLinearPnP(X, x, K, C, R)

        Cset.append(C)
        Rset.append(R)
        r_indx.append(i)
        Visibility[ind, i] = 1
        for j in range(0, len(r_indx) - 1):
            output = np.logical_and(
                np.logical_and(1 - recon_bin, matches[:, r_indx[j]].reshape(
                    (-1, 1))), matches[:, i].reshape((-1, 1)))
            ind, _ = np.where(output == True)
            if (len(ind) < 8):
                continue

            x1 = np.hstack((matchesX[ind, r_indx[j]].reshape(
                (-1, 1)), matchesY[ind, r_indx[j]].reshape((-1, 1))))
            x2 = np.hstack((matchesX[ind, i].reshape(
                (-1, 1)), matchesY[ind, i].reshape((-1, 1))))

            X = LinearTriangulation(K, Cset[j], Rset[j], x1, x2, DataPath, i,
                                    j)

            X_points[ind, :] = X
            recon_bin[ind] = 1
            Visibility[ind, r_indx[j]] = 1
            Visibility[ind, j] = 1

        for o in range(len(X_points)):
            if (X_points[o, 2] < 0):
                Visibility[o, :] = 0
                recon_bin[o] = 0

        V_bundle = BuildVisibilityMatrix(Visibility, r_indx)

        point_indices, _ = np.where(recon_bin == 1)
        camera_indices = i * np.ones((len(point_indices), 1))

        points_2d = np.hstack((matchesX[point_indices, i].reshape(
            (-1, 1)), matchesX[point_indices, i].reshape((-1, 1))))

        Rset, Cset, X_points = BundleAdjustment(Cset, Rset, X_points, K,
                                                points_2d, camera_indices,
                                                recon_bin, V_bundle)

    ind, _ = np.where(recon_bin == 1)
    X_points = X_points[ind, :]
    PixelVal = PixelVal[ind, :]

    # For 3D plotting
    ax = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    ax.scatter3D(X_points[:, 0],
                 X_points[:, 1],
                 X_points[:, 2],
                 c=PixelVal / 255.0,
                 s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([0, 20])
    plt.show()

    # For 2D plotting
    plt.scatter(X_points[:, 0], X_points[:, 2], c=PixelVal / 255.0, s=1)
    DrawCameras(Cset, Rset)
    ax1 = plt.gca()
    ax1.set_xlabel('x')
    ax1.set_ylabel('z')
    plt.show()


if __name__ == "__main__":
    main()
