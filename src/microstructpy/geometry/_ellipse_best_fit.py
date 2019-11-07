import numpy as np


def _best_fit(points, ellipse):
    # Unpack the input points
    pts = np.array(points, dtype='float')
    x, y = pts.T

    xx = x * x
    yy = y * y
    xy = x * y
    g = 1.0
    ones = -g * np.ones(len(x))
    coeffs = np.array([xx, xy, yy, x, y]).T
    a, b, c, d, f = np.linalg.lstsq(coeffs, ones, rcond=None)[0]
    b *= 0.5
    d *= 0.5
    f *= 0.5

    mj = b * b - a * c
    x0 = (c * d - b * f) / mj
    y0 = (a * f - b * d) / mj

    numer = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    denom1 = np.sign(c - a) * np.sqrt((a - c) * (a - c) + 4 * b * b)
    denom2 = a + c
    a2_d = mj * (denom1 - denom2)
    a2 = numer / a2_d
    b2_d = mj * (-denom1 - denom2)
    b2 = numer / b2_d

    a = alt_sqrt(a2)
    b = alt_sqrt(b2)

    if np.isclose(b, 0) and a < c:
        phi = 0
    elif np.isclose(b, 0):
        phi = 0.5 * np.pi
    else:
        phi = 0.5 * np.arctan2(2 * b, a - c)

    return a, b, phi, x0, y0


def alt_sqrt(x):
    if x >= 0:
        return np.sqrt(x)
    return - np.sqrt(-x)
