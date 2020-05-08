import numpy as np

def nonlinearerror(X, P, points):
	X = X.reshape((len(points), 4))
	X_homo = X.T
	# X = X[:, :3].T
	s = np.square(np.asarray(points[:, 0]) - (np.matmul(P[0, :], X_homo) / np.matmul(P[2, :], X_homo))) + np.square(
		np.asarray(points[:, 1]) - (np.matmul(P[1, :], X_homo) / np.matmul(P[2, :], X_homo)))
	s = s + np.square(np.asarray(points[:, 2]) - (np.matmul(P[0, :], X_homo) / np.matmul(P[2, :], X_homo))) + np.square(
		np.asarray(points[:, 3]) - (np.matmul(P[1, :], X_homo) / np.matmul(P[2, :], X_homo)))
	
	return np.asarray(s, dtype=np.float32)