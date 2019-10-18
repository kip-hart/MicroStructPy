import numpy as np
import pytest

import matplotlib.pyplot as plt

from microstructpy.geometry import ellipse


@pytest.mark.parametrize('kwargs', [{'center': (2, -3), 'a': 10, 'b': 2,
                                     'angle': -12},
                                    {'axes': [5, 1.2], 'angle_deg': 381},
                                    {'size': 2.3, 'aspect_ratio': 3.1},
                                    {'area': 129, 'aspect_ratio': 12},
                                    {'area': 1e-3, 'a': 5, 'angle_rad': -2},
                                    {'area': 102, 'b': 0.1},
                                    {'size': 2, 'a': 1,
                                     'matrix': [[0, -1], [1, 0]],
                                     'position': (2.31, 501.2)},
                                    {'size': 12, 'b': 2,
                                     'orientation': [[1/2, np.sqrt(3)/2],
                                                     [-np.sqrt(3)/2, 1/2]]},
                                    {'a': 201, 'aspect_ratio': 1},
                                    {'b': 2.13, 'aspect_ratio': 51.3},
                                    {'area': 1e-3}, {'a': 3}, {'b': 4.3},
                                    {'size': 6.413},
                                    ])
def test_ellipse___init__(kwargs):
    e = ellipse.Ellipse(**kwargs)
    for key in kwargs:
        attr = getattr(e, key)
        val = kwargs[key]
        assert np.isclose(attr, val).all()


def test_ellipse___init___default():
    e = ellipse.Ellipse()
    defaults = {'a': 1, 'b': 1, 'center': (0, 0), 'angle': 0}
    for key in defaults:
        attr = getattr(e, key)
        val = defaults[key]
        assert np.isclose(attr, val).all()

@pytest.mark.parametrize('kwargs', [{'center': (2, -3), 'a': 10, 'b': 2,
                                     'angle': -110},
                                    {'axes': [5, 1.2], 'angle_deg': 381},
                                    {'size': 2.3, 'aspect_ratio': 3.1},
                                    {'area': 129, 'aspect_ratio': 12},
                                    {'area': 1e-3, 'a': 5e-3, 'angle_rad': -2},
                                    {'area': 4, 'b': 0.2, 'angle': -134},
                                    {'area': 4, 'b': 0.2, 'angle': 130},
                                    {'size': 2, 'a': 1,
                                     'matrix': [[0, -1], [1, 0]],
                                     'position': (2.31, 501.2)},
                                    {'size': 12, 'b': 2,
                                     'orientation': [[1/2, np.sqrt(3)/2],
                                                     [-np.sqrt(3)/2, 1/2]]},
                                    {'a': 201, 'aspect_ratio': 1},
                                    {'b': 2.13, 'aspect_ratio': 51.3},
                                    {'area': 1e-3}, {'a': 3}, {'b': 4.3},
                                    {'size': 6.413},
                                    ])
def test_ellipse_best_fit_exact(kwargs):
    e = ellipse.Ellipse(**kwargs)
    a, b = e.axes
    center = e.center
    ang = e.angle_rad

    t = np.linspace(-0.5, 2 * np.pi + 0.5, 200)
    xp = a * np.cos(t)
    yp = b * np.sin(t)
    x = center[0] + xp * np.cos(ang) - yp * np.sin(ang)
    y = center[1] + xp * np.sin(ang) + yp * np.cos(ang)

    pts = np.array([x, y]).T
    e_fit = e.best_fit(pts)

    if not e_fit == e:
        e.plot(facecolor='none', edgecolor='C0')
        plt.plot(x, y, '.')
        e_fit.plot(facecolor='none', edgecolor='k')
        plt.axis('equal')
        plt.show()

    assert e_fit == e


