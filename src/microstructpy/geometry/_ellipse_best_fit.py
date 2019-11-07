import numpy as np


def _best_fit(points, ellipse):
    # Unpack the input points
    pts = np.array(points, dtype='float')
    x, y = pts.T

    # Quadratic part of design matrix
    D1 = np.mat(np.vstack([x * x, x * y, y * y])).T

    # Linear part of design matrix
    D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T

    # Scatter matrix
    S1 = D1.T * D1
    S2 = D1.T * D2
    S3 = D2.T * D2

    # Constraint matrix
    C1inv = np.mat([[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]])

    # Reduced scatter matrix
    M = C1inv * (S1 - S2 * S3.I * S2.T)

    # Find eigenvalues
    _, evec = np.linalg.eig(M)

    # Mask
    cond = 4 * np.multiply(evec[0, :], evec[2, :])
    cond -= np.multiply(evec[1, :], evec[1, :])
    a1 = evec[:, np.nonzero(cond.A > 0)[1]]

    a2 = -S3.I * S2.T * a1

    # Coefficients
    a = a1[0, 0]
    b = 0.5 * a1[1, 0]
    c = a1[2, 0]

    d = 0.5 * a2[0, 0]
    f = 0.5 * a2[1, 0]
    g = a2[2, 0]

    # Center of ellipse
    k = b * b - a * c
    xc = (c * d - b * f) / k
    yc = (a * f - b * d) / k

    # Semi-axes lengths
    numer = a * f * f
    numer += c * d * d
    numer += g * b * b
    numer -= 2 * b * d * f
    numer -= a * c * g
    numer *= 2

    tan2 = 2 * b / (a - c)
    sq_val = np.sqrt(1 + tan2 * tan2)
    denom1 = k * ((c - a) * sq_val - (c + a))
    denom2 = k * ((a - c) * sq_val - (c + a))
    width = np.sqrt(numer / denom1)
    height = np.sqrt(numer / denom2)

    # Angle of rotation
    phi = 0.5 * np.arctan(tan2)

    return width, height, phi, xc, yc
