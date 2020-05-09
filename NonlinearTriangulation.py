import numpy as np
import scipy.optimize as opt

# def nonlinearerror(X, P, points):
#     X = X.reshape((len(points), 4))
#     X_homo = X.T
#     # X = X[:, :3].T
#     s = np.square(
#         np.asarray(points[:, 0]) -
#         (np.matmul(P[0, :], X_homo) / np.matmul(P[2, :], X_homo))) + np.square(
#             np.asarray(points[:, 1]) -
#             (np.matmul(P[1, :], X_homo) / np.matmul(P[2, :], X_homo)))
#     s = s + np.square(
#         np.asarray(points[:, 2]) -
#         (np.matmul(P[0, :], X_homo) / np.matmul(P[2, :], X_homo))) + np.square(
#             np.asarray(points[:, 3]) -
#             (np.matmul(P[1, :], X_homo) / np.matmul(P[2, :], X_homo)))

#     return np.asarray(s, dtype=np.float32)

# def NonLinearTriangulation(K, x1, x2, X_init, R, C, R, C):
#     optimized_params = opt.least_squares(fun=nonlinearerror,
#                                          x0=init,
#                                          method="dogbox",
#                                          args=[K, x1, x2, R, C, R, C])

#     X = np.reshape(optimized_params.x, (sz, 3))


#     return X
def NonlinearTriangulation(K, x1, x2, X_init, R, C):
    """Summary

    Args:
        K (TYPE): Description
        x1 (TYPE): Description
        x2 (TYPE): Description
        X_init (TYPE): Description
        R (TYPE): Description
        C (TYPE): Description
        R (TYPE): Description
        C (TYPE): Description

    Returns:
        TYPE: Description
    """
    sz = x1.shape[0]
    # print(R)
    # print(C)
    assert x1.shape[0] == x2.shape[0] == X_init.shape[
        0], "2D-3D corresspondences have different shape "
    X = np.zeros((sz, 3))

    init = X_init.flatten()
    #     Tracer()()
    optimized_params = opt.least_squares(fun=nonlinearerror,
                                         x0=init,
                                         method="dogbox",
                                         args=[K, x1, x2, R, C])

    X = np.reshape(optimized_params.x, (sz, 3))

    return X


def nonlinearerror(init, K, x1, x2, R, C):
    """Summary

    Args:
        init (TYPE): Description
        K (TYPE): Description
        x1 (TYPE): Description
        x2 (TYPE): Description
        R (TYPE): Description
        C (TYPE): Description
        R (TYPE): Description
        C (TYPE): Description

    Returns:
        TYPE: Description
    """
    sz = x1.shape[0]
    X = np.reshape(init, (sz, 3))

    I = np.identity(3)
    C = np.reshape(C, (3, -1))

    X = np.hstack((X, np.ones((sz, 1))))

    P = np.dot(K, np.dot(R, np.hstack((I, -C))))

    error1 = 0
    error2 = 0
    error = []

    u1 = np.divide((np.dot(P[0, :], X.T).T), (np.dot(P[2, :], X.T).T))
    v1 = np.divide((np.dot(P[1, :], X.T).T), (np.dot(P[2, :], X.T).T))
    u2 = np.divide((np.dot(P[0, :], X.T).T), (np.dot(P[2, :], X.T).T))
    v2 = np.divide((np.dot(P[1, :], X.T).T), (np.dot(P[2, :], X.T).T))

    #     print(u1.shape,x1.shape)
    assert u1.shape[0] == x1.shape[0], "shape not matched"

    error1 = ((x1[:, 0] - u1) + (x1[:, 1] - v1))
    error2 = ((x2[:, 0] - u2) + (x2[:, 1] - v2))
    #     print(error1.shape)
    error = sum(error1, error2)

    return sum(error)