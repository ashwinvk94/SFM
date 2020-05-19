import numpy as np
import matplotlib.pyplot as plt
import math
import sys

sys.dont_write_bytecode = True


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)

    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def DrawCameras(C_set, R_set):
    for i in range(0, len(C_set)):
        R1 = rotationMatrixToEulerAngles(R_set[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set[i][0],
                 C_set[i][2],
                 marker=(3, 0, int(R1[1])),
                 markersize=15,
                 linestyle='None')
