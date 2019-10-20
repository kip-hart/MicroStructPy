from __future__ import division

import numpy as np
import pytest

from microstructpy.geometry import ellipse


@pytest.fixture(params=[{'center': (2, -3), 'a': 10, 'b': 2, 'angle': -110},
                        {'axes': [5, 1.2], 'angle_deg': 381},
                        {'size': 2.3, 'aspect_ratio': 3.1},
                        {'area': 129, 'aspect_ratio': 12},
                        {'area': 1e-3, 'a': 5e-3, 'angle_rad': -2},
                        {'area': 4, 'b': 0.2, 'angle': -134},
                        {'area': 4, 'b': 0.2, 'angle': 130},
                        {'area': 4, 'b': 0.2, 'angle': 180},
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
                        ], scope='module')
def kwargs(request):
    yield request.param


@pytest.fixture(scope='module')
def ellipse_geom(kwargs):
    yield ellipse.Ellipse(**kwargs)


@pytest.fixture(scope='module')
def ellipse_default():
    yield ellipse.Ellipse()


def test_ellipse___init__(kwargs):
    e = ellipse.Ellipse(**kwargs)
    for key in kwargs:
        attr = getattr(e, key)
        val = kwargs[key]
        assert np.isclose(attr, val).all()

@pytest.mark.parametrize('kw, val', [('a', 1), ('b', 1), ('center', (0, 0)),
                                     ('angle', 0)])
def test_ellipse___init___default(ellipse_default, kw, val):
    assert hasattr(ellipse_default, kw)
    actual_val = getattr(ellipse_default, kw)
    assert np.isclose(actual_val, val).all()


def test_ellipse_best_fit_exact(ellipse_geom):
    a, b = ellipse_geom.axes
    center = ellipse_geom.center
    ang = ellipse_geom.angle_rad

    t = np.linspace(-0.5, 2 * np.pi + 0.5, 200)
    xp = a * np.cos(t)
    yp = b * np.sin(t)
    x = center[0] + xp * np.cos(ang) - yp * np.sin(ang)
    y = center[1] + xp * np.sin(ang) + yp * np.cos(ang)

    pts = np.array([x, y]).T
    e_fit = ellipse_geom.best_fit(pts)
    assert e_fit == ellipse_geom


def test_ellipse___str__(ellipse_geom):
    assert isinstance(ellipse_geom.__str__(), str)


def test_ellipse___repr__(ellipse_geom):
    r = ellipse_geom.__repr__()
    assert eval('ellipse.' + r) == ellipse_geom


@pytest.mark.parametrize('key', [0, 1, 2, 3, 4])
def test_ellipse___eq__(ellipse_geom, key):
    if key % 2 == 0:
        a, b = ellipse_geom.axes
    else:
        b, a = ellipse_geom.axes
    ang = ellipse_geom.angle_deg + key * 90
    cen = ellipse_geom.center
    equiv_geom = ellipse.Ellipse(a=a, b=b, angle_deg=ang, center=cen)
    assert ellipse_geom.__eq__(equiv_geom)


@pytest.mark.parametrize('value', ['foo', 2.1, -3, (2, 4)])
def test_ellipse___eq__class(ellipse_default, value):
    with pytest.raises(TypeError):
        ellipse_default.__eq__(value)


@pytest.mark.parametrize('key', [0, 1, 2, 3, 4])
def test_ellipse___ne__(ellipse_geom, key):
    if key % 2 == 1:
        a, b = ellipse_geom.axes
    else:
        b, a = ellipse_geom.axes

    ang = ellipse_geom.angle_deg + key * 90 + 10
    cen = ellipse_geom.center
    equiv_geom = ellipse.Ellipse(a=a, b=b, angle_deg=ang, center=cen)
    if not np.isclose(a, b):
        assert ellipse_geom.__ne__(equiv_geom)


def test_ellipse___ne___default(ellipse_default, ellipse_geom):
    assert ellipse_geom.__ne__(ellipse_default)


def test_ellipse_n_dim(ellipse_default):
    assert ellipse_default.n_dim == 2


def test_ellipse_area_expectation(kwargs):
    e = ellipse.Ellipse(**kwargs)
    if ('size' in kwargs) or ('area' in kwargs):
        exp_area = ellipse.Ellipse.area_expectation(**kwargs)
        assert np.isclose(e.area, exp_area)
    elif ('a' in kwargs) and ('b' in kwargs):
        exp_area = ellipse.Ellipse.area_expectation(**kwargs)
        assert np.isclose(e.area, exp_area)
    elif ('a' in kwargs) and ('aspect_ratio' in kwargs):
        exp_area = ellipse.Ellipse.area_expectation(**kwargs)
        assert np.isclose(e.area, exp_area)
    elif ('b' in kwargs) and ('aspect_ratio' in kwargs):
        exp_area = ellipse.Ellipse.area_expectation(**kwargs)
        assert np.isclose(e.area, exp_area)


def test_ellipse_area_expectation_error():
    with pytest.raises(KeyError):
        ellipse.Ellipse.area_expectation()


def test_ellipse_approximate(ellipse_geom):
    approx = ellipse_geom.approximate()
    assert np.array(approx).shape[1] == 3

    rel_pos = approx[:, :-1] - np.array(ellipse_geom.center).reshape(1, -1)
    rel_dist = np.sqrt(np.sum(rel_pos * rel_pos, axis=-1))
    max_dist = rel_dist + approx[:, -1]
    assert np.all(max_dist <= max(ellipse_geom.axes))


def test_ellipse_limits(ellipse_geom):
    # Limits are centered on the ellipse
    lims = np.array(ellipse_geom.limits)
    lims_cen = lims.mean(axis=1)
    cen = ellipse_geom.center
    assert np.isclose(lims_cen, cen).all()

    # Limits are within the major radius of the ellipse
    rel_lims = lims - np.array(cen).reshape(-1, 1)
    assert np.all(np.abs(rel_lims) <= max(ellipse_geom.axes))


def test_ellipse_within_center(ellipse_geom):
    cen = ellipse_geom.center
    assert ellipse_geom.within(cen)


def test_ellipse_within_limits(ellipse_geom):
    lims = ellipse_geom.limits
    pts = []
    for i in range(2):
        for j in range(2):
            x = lims[0][i]
            y = lims[1][j]
            pts.append([x, y])
    assert not np.any(ellipse_geom.within(pts))
