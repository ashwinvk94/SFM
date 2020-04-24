#!/usr/bin/env python2
'''
DisambiguateCameraPose

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)

Description:
Implements chirality check for the 4 estimated extrinsic
calibration parameters

Section 1.3.5
https://cmsc733.github.io/2019/proj/pfinal/
'''

def DisambiguateCameraPose(X,R,C):
	r3 = R[2,:]
	checlVal = np.matmul(r3,X-C)
	if checlVal>0:
		return True
	else:
		return False