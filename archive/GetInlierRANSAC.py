import numpy as np


def get_point(kp1, kp2, match):
    x1, y1 = kp1[match.queryIdx].pt
    x2, y2 = kp2[match.trainIdx].pt
    
    return [x1, y1, x2, y2]


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


def get_inliers(kp1, kp2, matches, F):
    inliers = []
    err_thresh = 0.01
    
    for m in matches:
        p = get_point(kp1, kp2, m)
        x1 = np.asarray([p[0], p[1], 1])
        x2 = np.asarray([p[2], p[3], 1])
        
        e = np.matmul(x1, np.matmul(F, x2.T))
        if abs(e) < err_thresh:
            inliers.append(m)
    
    inliers = np.asarray(inliers)
    
    return inliers

def get_F_with_ransac(F, points, kp1, kp2, matches):
    max_in = 0
    max_inliers = []
    F_ = []
    inliers = get_inliers(kp1, kp2, matches, F)
    
    if inliers.shape[0] > max_in:
        max_inliers = inliers
        max_in = inliers.shape[0]
        F_ = F


    points = []
    
    # Recompute F with all the inliers
    for m in max_inliers:
        pt = get_point(kp1, kp2, m)
        points.append(pt)
    
    points = np.asarray(points)
    F = get_F_util(points)
    
    return F, points