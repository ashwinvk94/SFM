import numpy as np
import random


def get_point(kp1, kp2, match):
    x1, y1 = kp1[match.queryIdx].pt
    x2, y2 = kp2[match.trainIdx].pt

    return [x1, y1, x2, y2]


def featureToPoints(kp1, kp2, matches):
    pts1 = []
    pts2 = []
    points = []
    for match in matches:
        pts = get_point(kp1, kp2, match)
        points.append(pts)
        pts1.append([pts[0], pts[1]])
        pts2.append([pts[2], pts[3]])

    return np.array(pts1), np.array(pts2), points


def get_F_util(points):
    A = []
    for pt in points:
        x1, y1, x2, y2 = pt
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    A = np.asarray(A)

    U, S, Vh = np.linalg.svd(A, full_matrices=True)

    f = Vh[-1, :]
    F = np.reshape(f, (3, 3)).transpose()

    # Correct rank of F
    U, S, Vh = np.linalg.svd(F)
    S_ = np.eye(3)
    for i in range(3):
        S_[i, i] = S[i]
    S_[2, 2] = 0

    F = np.matmul(U, np.matmul(S_, Vh))

    return F


def get_inliers(points, F):
    inliers = []
    err_thresh = 0.005

    for p in points:
        x1 = np.asarray([p[0], p[1], 1])
        x2 = np.asarray([p[2], p[3], 1])

        e = np.matmul(x2, np.matmul(F, x1.T))

        # print(e)
        if abs(e) < err_thresh:
            inliers.append(p)

    inliers = np.asarray(inliers)

    return inliers


def get_F_with_ransac(points):
    foundFlag = False
    nPoints = len(points)
    inlierThresh = 0.6

    for k in range(200):
        randindx = random.sample(range(0, len(points)), 8)
        randpts = []
        for i in randindx:
            randpts.append(points[i])
        F = get_F_util(randpts)
        inliers = get_inliers(points, F)
        nInliers = len(inliers)
        perc_inliers = float(nInliers) / nPoints
        print("for iteration", str(k), "percentage inliers", str(perc_inliers))
        if perc_inliers > inlierThresh:
            foundFlag = True
            break
    if not foundFlag:
        print("not found")
    # final estimation of F
    F = get_F_util(inliers)
    return F, inliers, foundFlag


def getInd(points, checkpt):
    for i, pt in enumerate(points):
        if (np.array(pt) == np.array(checkpt)).all():
            return i
    return None


def getMatches(points, inliers, good):
    inlier_matches = []
    for in_pt in inliers:
        ind = getInd(points, in_pt)
        # print("found at ", str(ind))
        inlier_matches.append(good[ind])
    return inlier_matches


def convertPts(points):
    pts1 = []
    pts2 = []
    for pt in points:
        pts1.append([pt[0], pt[1]])
        pts2.append([pt[2], pt[3]])
    return np.array(pts1), np.array(pts2)
