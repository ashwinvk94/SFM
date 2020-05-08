import cv2
import numpy as np
import sys


def get_feature_matches(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT in current as well as next frame
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Ratio test as per Lowe's paper
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good.append(m)
    
    return kp1, kp2, good


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


def get_F(kp1, kp2, matches):
    np.random.seed(30)
    
    for i in range(50):
        idx = np.random.choice(len(matches), 8, replace=False)
        points = []
        for i in idx:
            pt = get_point(kp1, kp2, matches[i])
            points.append(pt)
        
        points = np.asarray(points)
        
        F = get_F_util(points)
        
    
    return F, points


"""
def applyANMS(img1, nbest):
    coordinates = cv2.goodFeaturesToTrack(img1, 3 * nbest, 0.05, 20)
    Nstrong = len(coordinates)
    # print(Nstrong)
    rcord = []
    eucDist = 0
    for i in range(Nstrong):
        rcord.append([sys.maxsize, [coordinates[i][0][0], coordinates[i][0][1]]])
    for i in range(Nstrong):
        for j in range(Nstrong):
            yi = int(coordinates[i][0][0])
            xi = int(coordinates[i][0][1])
            yj = int(coordinates[j][0][0])
            xj = int(coordinates[j][0][1])
            if img1[xj][yj] > img1[xi][yi]:
                eucDist = (xj - xi) ** 2 + (yj - yi) ** 2
            if eucDist < rcord[i][0]:
                rcord[i][0] = eucDist
                rcord[i][1] = [xi, yi]
    rcord.sort()
    rcord = rcord[::-1]
    rcord = rcord[:nbest]
    result = []
    for r in rcord:
        result.append(r[1])
    return np.asarray(result)


def match_features(featureVec1, featureVec2):
    matches = []
    matching_ratio_threshold = 0.8
    for vec1_ind, vec1 in enumerate(featureVec1):
        sum_sq_dist = []
        for vec2 in featureVec2:
            sum_sq_dist.append(sum((vec1 - vec2) ** 2))
        min_val = min(sum_sq_dist)[0]
        second_min_val = find_second_smallest(sum_sq_dist)
        # print('matching ratio for corner ',str(vec1_ind),'is = ',(min_val/second_min_val))
        if ((min_val / second_min_val) < matching_ratio_threshold):
            vec2_ind = sum_sq_dist.index(min_val)
            matches.append([vec1_ind, vec2_ind])
    return matches


def find_second_smallest(arr):
    arr1 = arr[:]
    minVal = min(arr1)[0]
    arr1.remove(minVal)
    minVal = min(arr1)[0]
    return minVal


def get_feature_vectors(img, corners):
    # parameters
    feature_crop_half_size = 20
    gaussian_blur_filter_size = 3
    resize_arr_size = 8
    
    # smoothing the image
    img = cv2.GaussianBlur(img, (gaussian_blur_filter_size, gaussian_blur_filter_size), cv2.BORDER_DEFAULT)
    
    # padding iamge by copying the border
    imgPadded = np.pad(img, (20, 20), mode='edge')
    
    featureArrs = []
    for cornerInd in range(corners.shape[0]):
        pixelCoordinates = corners[cornerInd, :]
        
        # cropping image
        featureArr = imgPadded[pixelCoordinates[0]:pixelCoordinates[0] + 2 * feature_crop_half_size,
                     pixelCoordinates[1]:pixelCoordinates[1] + 2 * feature_crop_half_size]
        featureArr = cv2.resize(featureArr, (resize_arr_size, resize_arr_size), interpolation=cv2.INTER_AREA)
        featureArr = featureArr.reshape((resize_arr_size * resize_arr_size, 1))
        featureArr = normalize_features(featureArr)
        featureArrs.append(featureArr)
    
    return featureArrs


def normalize_features(featureArr):
    mean = np.mean(featureArr)
    sd = np.std(featureArr)
    featureArr = (featureArr - mean) / (sd + 10 ** -7)
    return featureArr
"""