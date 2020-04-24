#!/usr/bin/env python2
'''
Triangulation

Author:
Ashwin Varghese Kuruttukulam(ashwinvk94@gmail.com)

Description:
This functtion estimates the camera pose given the essential matrix

Section 1.3.4
https://cmsc733.github.io/2019/proj/pfinal/
'''

def ExtractCameraPose(E):
	U, _, Vt = numpy.linalg.svd(E)
	# because numpy gives V transpose
	V = Vt.T

	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

	C1 = U[:,2]
	C2 = -U[:,2]
	C3 = U[:,2]
	C4 = -U[:,2]

	R1 = np.matmul(np.matmul(U,W),V.T)
	print(numpy.linalg.det(R1))
	if(numpy.linalg.det(R1)):
		R1 = -R1
		C1 = -C1
	R2 = np.matmul(np.matmul(U,W),V.T)
	print(numpy.linalg.det(R2))
	if(numpy.linalg.det(R2)):
		R2 = -R2
		C2 = -C2
	R3 = np.matmul(np.matmul(U,W.T),V.T)
	print(numpy.linalg.det(R3))
	if(numpy.linalg.det(R3)):
		R3 = -R3
		C3 = -C3
	R4 = np.matmul(np.matmul(U,W.T),V.T)
	print(numpy.linalg.det(R4))
	if(numpy.linalg.det(R4)):
		R4 = -R4
		C4 = -C4

	return [R1,R2,R3,R4],[C1,C2,C3,C4]


