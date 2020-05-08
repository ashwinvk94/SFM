import cv2
import numpy as np
import sys


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
