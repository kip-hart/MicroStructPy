"""Regression tests for the geometry module bug fixes."""
import itertools

import matplotlib
import numpy as np
import pytest
import scipy.stats
from matplotlib import pyplot as plt
from pyquaternion import Quaternion

from microstructpy.geometry import Box
from microstructpy.geometry import Circle
from microstructpy.geometry import Ellipse
from microstructpy.geometry import Ellipsoid
from microstructpy.geometry import Rectangle
from microstructpy.geometry import Sphere
from microstructpy.geometry import Square

matplotlib.use('agg')


def _rot2d(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# --------------------------------------------------------------------------- #
# Ellipse                                                                     #
# --------------------------------------------------------------------------- #
def test_ellipse_axes_keyword():
    e = Ellipse(axes=[2, 1])
    assert e.a == 2 and e.b == 1
    with pytest.raises(ValueError):
        Ellipse(axes=[2, -1])


def test_ellipse_matrix_keyword_accepts_rotations():
    for deg in (0, 30, 90, -135):
        e = Ellipse(a=2, b=1, matrix=_rot2d(deg))
        assert np.isclose((e.angle - deg + 180) % 360 - 180, 0)
        e = Ellipse(a=2, b=1, orientation=_rot2d(deg))
        assert np.isclose((e.angle - deg + 180) % 360 - 180, 0)
    with pytest.raises(ValueError):
        Ellipse(matrix=[[1, 1], [0, 1]])


def test_ellipse_reflect_broadcasting():
    e = Ellipse(a=2, b=1)
    pts = e.reflect([[1, 0], [0, 0.5], [2, 2]])
    assert pts.shape == (3, 2)
    # a point on the boundary is its own reflection
    assert np.allclose(e.reflect([2, 0]), [2, 0])


def test_ellipse_area_expectation_numpy_scalars():
    exp = 0.25 * np.pi * 4
    assert np.isclose(Ellipse.area_expectation(size=np.float64(2),
                                               aspect_ratio=2), exp)
    assert np.isclose(Ellipse.area_expectation(axes=[2, np.float32(1)]),
                      2 * np.pi)
    assert np.isclose(Ellipse.area_expectation(a=np.float64(2), b=1),
                      2 * np.pi)


def test_ellipse_area_expectation_deterministic():
    kw = {'a': scipy.stats.uniform(1, 1),
          'aspect_ratio': scipy.stats.uniform(1, 2)}
    assert Ellipse.area_expectation(**kw) == Ellipse.area_expectation(**kw)


def test_ellipse_equality():
    e1 = Ellipse(a=2, b=1, angle_deg=30, center=(1, 2))
    e2 = Ellipse(a=2, b=1, angle_deg=390, center=(1, 2))
    e3 = Ellipse(a=2, b=1.1, angle_deg=30, center=(1, 2))
    assert e1 == e2
    assert e1 != e3
    assert e1 != Circle(r=1)


# --------------------------------------------------------------------------- #
# Ellipsoid                                                                   #
# --------------------------------------------------------------------------- #
def test_ellipsoid_ratio_bc_with_c():
    e = Ellipsoid(c=1, ratio_bc=2)
    assert np.allclose(e.axes, (1, 2, 1))
    e = Ellipsoid(size=2, c=0.5, ratio_bc=2)
    assert np.isclose(e.ratio_bc, 2)
    assert np.isclose(e.size, 2)


def test_ellipsoid_str_round_trip_rot_seq():
    import ast
    e = Ellipsoid(a=3, b=2, c=1, matrix=Quaternion.random().rotation_matrix)
    line = [ln for ln in str(e).split('\n') if ln.startswith('rot_seq')][0]
    rot_seq = ast.literal_eval(line.split(':', 1)[1].strip())
    e2 = Ellipsoid(a=3, b=2, c=1, rot_seq=list(rot_seq))
    assert np.allclose(e2.matrix, e.matrix)


@pytest.mark.parametrize('axes', list(itertools.permutations([5, 3, 1])))
def test_ellipsoid_approximate_all_axis_orderings(axes):
    e = Ellipsoid(a=axes[0], b=axes[1], c=axes[2])
    sph = e.approximate()
    cen, r = sph[:, :3], sph[:, 3]
    # union of spheres spans exactly the ellipsoid extents
    assert np.allclose(np.max(np.abs(cen) + r[:, None], axis=0), axes)
    # every sphere lies inside the ellipsoid
    rng = np.random.RandomState(0)
    u = rng.normal(size=(200, 3))
    u /= np.linalg.norm(u, axis=1).reshape(-1, 1)
    worst = 0
    for c, rad in zip(cen, r):
        p = c + rad * u
        worst = max(worst, np.max(np.sum((p / np.array(axes)) ** 2, axis=1)))
    assert worst < 1.01


def test_ellipsoid_limits_exact_for_rotations():
    rng = np.random.RandomState(1)
    for _ in range(20):
        axes = rng.uniform(0.2, 5, 3)
        R = Quaternion.random().rotation_matrix
        cen = rng.uniform(-2, 2, 3)
        e = Ellipsoid(axes=axes, matrix=R, center=cen)
        lims = np.array(e.limits)
        # brute force over a fine sampling of the surface
        u = rng.normal(size=(20000, 3))
        u /= np.linalg.norm(u, axis=1).reshape(-1, 1)
        pts = (u * axes).dot(R.T) + cen
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        assert np.all(lims[:, 0] <= lo + 1e-12)
        assert np.all(lims[:, 1] >= hi - 1e-12)
        assert np.allclose(lims[:, 0], lo, atol=0.02 * axes.max())
        assert np.allclose(lims[:, 1], hi, atol=0.02 * axes.max())
    e = Ellipsoid(a=3, b=2, c=1, center=(1, 1, 1))
    assert np.allclose(e.limits, [(-2, 4), (-1, 3), (0, 2)])


def test_ellipsoid_volume_expectation_numpy_scalars():
    v = Ellipsoid.volume_expectation(size=np.float64(2), ratio_ab=2)
    assert np.isclose(v, 4 * np.pi / 3)


def test_ellipsoid_volume_expectation_deterministic():
    kw = {'a': scipy.stats.uniform(1, 1),
          'ratio_ab': scipy.stats.uniform(1, 2),
          'ratio_ac': scipy.stats.uniform(1, 2)}
    v1 = Ellipsoid.volume_expectation(**kw)
    v2 = Ellipsoid.volume_expectation(**kw)
    assert v1 == v2


def test_ellipsoid_reflect_broadcasting():
    e = Ellipsoid(a=2, b=1, c=1)
    pts = e.reflect(np.zeros((5, 3)) + [1, 0, 0])
    assert pts.shape == (5, 3)
    assert np.allclose(pts, [3, 0, 0])


def test_ellipsoid_equality():
    R = Quaternion.random().rotation_matrix
    e1 = Ellipsoid(a=3, b=2, c=1, matrix=R, center=(1, 2, 3))
    e2 = Ellipsoid(axes=(3, 2, 1), matrix=R, center=(1, 2, 3))
    assert e1 == e2
    assert e1 != Ellipsoid(a=3, b=2, c=1.5, matrix=R, center=(1, 2, 3))


# --------------------------------------------------------------------------- #
# Spheres, boxes, rectangles                                                  #
# --------------------------------------------------------------------------- #
def test_sphere_volume_expectation_numpy_scalars():
    assert np.isclose(Sphere.volume_expectation(r=np.float64(1)),
                      4 * np.pi / 3)
    assert np.isclose(Sphere.volume_expectation(size=np.float64(2)),
                      4 * np.pi / 3)


def test_sphere_plot_on_fresh_figure():
    plt.figure()
    Sphere(r=1).plot()
    plt.close('all')


def test_circle_best_fit_singular():
    # collinear points: the linear system is singular, must not raise
    pts = [[0, 0], [1, 0], [2, 0], [3, 0]]
    c = Circle.best_fit(pts)
    assert np.isfinite(c.r)


def test_rectangle_within_respects_rotation():
    r = Rectangle(length=4, width=1, angle=90)
    assert r.within([0, 1.5])
    assert not r.within([1.5, 0])
    assert np.all(r.within([[0, 1.9], [0.4, -1.9]]))
    r = Rectangle(length=4, width=1)
    assert r.within([1.5, 0]) and not r.within([0, 1.5])


def test_rectangle_length_only_matches_expectation():
    r = Rectangle(length=2)
    assert np.allclose(r.side_lengths, [2, 1])
    assert np.isclose(r.area, Rectangle.area_expectation(length=2))


def test_square_area_expectation_returns_value():
    assert np.isclose(Square.area_expectation(side_lengths=[2, 2]), 4)
    assert np.isclose(Square.area_expectation(side_length=2), 4)


def test_box_str_round_trip_precision():
    import ast
    b = Box(center=(0.1, 0.2, 0.3), side_lengths=(1 / 3, 2 / 3, 1.1))
    vals = {}
    for line in str(b).split('\n'):
        k, v = line.split(':', 1)
        vals[k.strip().lower().replace(' ', '_')] = ast.literal_eval(v.strip())
    assert vals['center'] == (0.1, 0.2, 0.3)
    assert vals['side_lengths'] == (1 / 3, 2 / 3, 1.1)
    assert Box(**vals) == b


def test_nbox_equality():
    r1 = Rectangle(center=(1, 1), length=2, width=1, angle=30)
    r2 = Rectangle(center=(1, 1), side_lengths=(2, 1), angle=30)
    assert r1 == r2
    assert r1 != Rectangle(center=(1, 1), length=2, width=1, angle=31)
