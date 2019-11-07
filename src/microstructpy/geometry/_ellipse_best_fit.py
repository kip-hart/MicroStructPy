import numpy as np

from microstructpy.geometry import ellipses


def _best_fit(points, ellipse):
    data = np.array(points).T
    lsqe = ellipses.LSqEllipse()
    lsqe.fit(data)
    center, width, height, phi = lsqe.parameters()

    return width, height, phi, center[0], center[1]
