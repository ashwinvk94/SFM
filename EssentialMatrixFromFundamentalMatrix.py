import numpy as np


def findEssentialMatrix(F, K):
	E = np.matmul(K.T, np.matmul(F, K))

	# Correct rank of E
	U, S, Vh = np.linalg.svd(E, full_matrices = True)
	S_ = np.eye(3)
	S_[2, 2] = 0

	E = np.matmul(U, np.matmul(S_, Vh))

	return E