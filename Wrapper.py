#!/usr/bin/env python2
'''
Wrapper.py

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)

Description:
Main function which integrates the total traditional sfm pipeline
metioned in the below link

https://cmsc733.github.io/2019/proj/pfinal/
'''
import sys
import numpy as np
import cv2

from LoadData import *
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from DrawCorrespondence import DrawCorrespondence
from EssentialMatrixFromFundamentalMatrix import findEssentialMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose

# Camera Intrinsic Matrix
K = np.array([[568.996140852, 0, 643.21055941],
              [0, 568.988362396, 477.982801038], [0, 0, 1]])

n_images = 6
img1 = 1
img2 = 2


def main():
    DataPath = './Data/'
    visualize = True

    # # Create matrices for All correspondences
    # Mx, My, M, Color = LoadData(DataPath)

    # #  Filter M for inliers
    # M, outlier_indices = inlier_filter(Mx, My, M, n_images)

    # # Save M matrix in case needed to lower the run time by loading
    # np.save('M', M)
    # np.save('Color', Color)
    # np.save('Mx', Mx)
    # np.save('My', My)
    # np.save('outlier_indices', outlier_indices)

    M = np.load('M.npy')
    Color = np.load('Color.npy')
    Mx = np.load('Mx.npy')
    My = np.load('My.npy')
    outlier_indices = np.load('outlier_indices.npy')

    recon_bin = np.zeros((M.shape[0], 1))
    X_3D = np.zeros((M.shape[0], 3))

    #  We have all inliers at this point in M
    output = np.logical_and(M[:, img1 - 1], M[:, img2 - 1])
    outlier = np.logical_and(outlier_indices[:, img1 - 1],
                             outlier_indices[:, img2 - 1])
    outlier_idx = np.where(outlier == True)
    indices, = np.where(output == True)
    rgb_list = Color[indices]

    pts1 = np.hstack((Mx[indices, img1 - 1].reshape(
        (-1, 1)), My[indices, img1 - 1].reshape((-1, 1))))
    pts2 = np.hstack((Mx[indices, img2 - 1].reshape(
        (-1, 1)), My[indices, img2 - 1].reshape((-1, 1))))
    F = EstimateFundamentalMatrix(np.float32(pts1), np.float32(pts2))
    print('fundamental matrix', F)

    outlier1 = np.hstack((Mx[outlier_idx, img1 - 1].reshape(
        (-1, 1)), My[outlier_idx, img1 - 1].reshape((-1, 1))))
    outlier2 = np.hstack((Mx[outlier_idx, img2 - 1].reshape(
        (-1, 1)), My[outlier_idx, img2 - 1].reshape((-1, 1))))

    if (visualize):
        out = DrawCorrespondence(img1,
                                 img2,
                                 pts1,
                                 pts2,
                                 outlier1,
                                 outlier2,
                                 DrawOutliers=False)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1000, 600)
        cv2.imshow('image', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    E = findEssentialMatrix(F, K)
    print('essential matrix : ', E)
    Rs, Cs = ExtractCameraPose(E)

    # chrality check and linear triangulation
    R, C, X = DisambiguateCameraPose(Rs, Cs, pts1, pts2, K)


if __name__ == '__main__':
    main()