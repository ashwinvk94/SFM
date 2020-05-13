import numpy as np
from LoadData import *
from EstimateFundamentalMatrix import *
from DrawCorrespondence import DrawCorrespondence
import cv2
from GetInlierRANSAC import *

"""
return 2  numpy array of size Nx2 each. Corresponding
elements of each array are the matching pixel coordinates
"""


def featureMatching(DataPath, visualize, img1, img2):
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

    M = np.load("M.npy")
    Color = np.load("Color.npy")
    Mx = np.load("Mx.npy")
    My = np.load("My.npy")
    outlier_indices = np.load("outlier_indices.npy")

    recon_bin = np.zeros((M.shape[0], 1))
    X_3D = np.zeros((M.shape[0], 3))

    #  We have all inliers at this point in M
    output = np.logical_and(M[:, img1 - 1], M[:, img2 - 1])
    outlier = np.logical_and(outlier_indices[:, img1 - 1], outlier_indices[:, img2 - 1])
    outlier_idx = np.where(outlier == True)
    (indices,) = np.where(output == True)
    rgb_list = Color[indices]

    pts1 = np.hstack(
        (Mx[indices, img1 - 1].reshape((-1, 1)), My[indices, img1 - 1].reshape((-1, 1)))
    )
    pts2 = np.hstack(
        (Mx[indices, img2 - 1].reshape((-1, 1)), My[indices, img2 - 1].reshape((-1, 1)))
    )
    outlier1 = np.hstack(
        (
            Mx[outlier_idx, img1 - 1].reshape((-1, 1)),
            My[outlier_idx, img1 - 1].reshape((-1, 1)),
        )
    )
    outlier2 = np.hstack(
        (
            Mx[outlier_idx, img2 - 1].reshape((-1, 1)),
            My[outlier_idx, img2 - 1].reshape((-1, 1)),
        )
    )

    if visualize:
        out = DrawCorrespondence(
            img1, img2, pts1, pts2, outlier1, outlier2, DrawOutliers=False
        )
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 1000, 600)
        cv2.imshow("image", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pts1, pts2


def featureMatchingcv2(DataPath, visualize, imgn1, imgn2):
    img1 = cv2.imread(DataPath + str(imgn1) + ".jpg")
    img2 = cv2.imread(DataPath + str(imgn2) + ".jpg")
    kp1, kp2, good = get_feature_matches(img1, img2)

    pts1, pts2, points = featureToPoints(kp1, kp2, good)

    # print(pts1.shape)
    # print(len(points))
    # print(pts2.shape)

    F, inliers, foundFlag = get_F_with_ransac(points)
    inliers_matches = getMatches(points, inliers, good)
    if visualize:
        # Draw first 10 matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
        img4 = cv2.drawMatches(img1, kp1, img2, kp2, inliers_matches, None)
        #     out = DrawCorrespondence(img1,
        #                              img2,
        #                              pts1,
        #                              pts2,
        #                              outlier1,
        #                              outlier2,
        #                              DrawOutliers=False)
        #     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow('image', 1000, 600)
        cv2.imshow("before ransac", img3)
        cv2.imshow("after ransac", img4)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return None, None

    # return pts1, pts2
