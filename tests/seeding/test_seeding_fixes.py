"""Regression tests for the seeding module bug fixes."""
import copy

import matplotlib
import numpy as np
import pytest
import scipy.stats
from matplotlib import pyplot as plt

import microstructpy as msp
from microstructpy.seeding import Seed
from microstructpy.seeding import SeedList
from microstructpy.seeding.seedlist import calc_rtol
from microstructpy.seeding.seedlist import sample_pos_within

matplotlib.use('agg')


def _phases_2d():
    return [{'shape': 'ellipse', 'size': scipy.stats.lognorm(s=0.3, scale=0.5),
             'aspect_ratio': scipy.stats.uniform(1, 2),
             'angle_deg': scipy.stats.uniform(0, 180), 'fraction': 0.6},
            {'shape': 'circle', 'size': scipy.stats.uniform(0.2, 0.3),
             'fraction': 0.4}]


def _signature(seeds):
    return [(s.phase, round(s.volume, 10), tuple(np.round(s.position, 8)))
            for s in seeds]


# --------------------------------------------------------------------------- #
# Seed generation                                                             #
# --------------------------------------------------------------------------- #
def test_from_info_keyword_order_independent():
    """The RNG chain must not depend on the (hash-dependent) set order."""
    phases = _phases_2d()
    seeds_1 = SeedList.from_info(copy.deepcopy(phases), 30.0)
    # same phases with the dictionary keys inserted in another order
    reordered = [dict(reversed(list(p.items()))) for p in phases]
    seeds_2 = SeedList.from_info(reordered, 30.0)
    assert _signature(seeds_1) == _signature(seeds_2)


def test_from_info_repeated_calls_identical():
    phases = _phases_2d()
    seeds_1 = SeedList.from_info(copy.deepcopy(phases), 30.0)
    seeds_2 = SeedList.from_info(copy.deepcopy(phases), 30.0)
    assert _signature(seeds_1) == _signature(seeds_2)


def test_from_info_does_not_mutate_rng_seeds():
    rng_seeds = {'size': 3, 'fraction': 1}
    SeedList.from_info(_phases_2d(), 20.0, rng_seeds)
    assert rng_seeds == {'size': 3, 'fraction': 1}
    assert SeedList.from_info.__func__.__defaults__[0] == {}


# --------------------------------------------------------------------------- #
# Overlap tolerance                                                           #
# --------------------------------------------------------------------------- #
def test_calc_rtol_values():
    rng = np.random.RandomState(0)
    areas = np.exp(-9 + 0.5 * rng.normal(size=4000))
    seeds = [Seed.factory('circle', area=a) for a in areas]
    cv_s = scipy.stats.variation(areas)
    expected = ((0.362954 * cv_s ** 2 - 0.419069 * cv_s + 0.184959) /
                (cv_s ** 2 - 1.05989 * cv_s + 0.365096))
    assert np.isclose(calc_rtol(seeds), expected)

    seeds_3d = [Seed.factory('sphere', volume=a) for a in areas]
    expected_3d = ((0.471115 * cv_s ** 2 - 0.602324 * cv_s + 0.297562) /
                   (cv_s ** 2 - 1.08469 * cv_s + 0.428216))
    assert np.isclose(calc_rtol(seeds_3d), expected_3d)

    # constant sizes: cv = 0 (a single seed as well)
    same = [Seed.factory('circle', r=1) for _ in range(3)]
    assert np.isclose(calc_rtol(same), 0.184959 / 0.365096)
    assert np.isclose(calc_rtol(same[:1]), 0.184959 / 0.365096)


def test_position_uses_calc_rtol(monkeypatch):
    seen = {}
    real = msp.seeding.seedlist.calc_rtol

    def spy(seeds):
        seen['rtol'] = real(seeds)
        return seen['rtol']

    monkeypatch.setattr(msp.seeding.seedlist, 'calc_rtol', spy)
    domain = msp.geometry.Square(side_length=4)
    seeds = SeedList.from_info(_phases_2d(), domain.area)
    seeds.position(domain)
    assert np.isclose(seen['rtol'], real(seeds))


# --------------------------------------------------------------------------- #
# Seeds and files                                                             #
# --------------------------------------------------------------------------- #
def test_seed_factory_position_and_center_consistent():
    s = Seed.factory('circle', r=0.5, position=(1, 2))
    assert np.allclose(s.breakdown, [[1, 2, 0.5]])
    assert np.allclose(s.geometry.center, [1, 2])

    s = Seed.factory('circle', r=0.5, center=(3, 4))
    assert np.allclose(s.position, [3, 4])
    assert np.allclose(s.breakdown, [[3, 4, 0.5]])

    s.position = (5, 5)
    assert np.allclose(s.breakdown, [[5, 5, 0.5]])
    assert np.allclose(s.geometry.center, [5, 5])

    e = Seed.factory('ellipse', a=2, b=1, angle_deg=30, position=(1, -1))
    assert np.allclose(np.mean(np.array(e.breakdown)[:, :2], axis=0),
                       [1, -1])


def test_seed_update_breakdown_exists():
    s = Seed.factory('ellipse', a=2, b=1)
    s.breakdown = []
    s.update_breakdown()
    assert len(s.breakdown) > 1


def test_seed_factory_unsupported_shape():
    with pytest.raises(ValueError):
        Seed.factory('box', side_lengths=(1, 1, 1))


def test_seed_equality_different_breakdowns():
    s1 = Seed.factory('ellipse', a=2, b=1)
    s2 = Seed.factory('ellipse', a=3, b=1)
    assert s1 != s2
    assert s1 == Seed.factory('ellipse', a=2, b=1)


def test_seedlist_mutable_default():
    x = SeedList()
    x.append(Seed.factory('circle', r=1))
    assert len(SeedList()) == 0


def test_seed_file_round_trip_and_reposition(tmp_path):
    phases = [{'shape': 'circle', 'size': 0.3},
              {'shape': 'ellipse', 'size': 0.4, 'aspect_ratio': 2,
               'angle_deg': scipy.stats.uniform(0, 180)},
              {'shape': 'rectangle', 'length': 0.3, 'width': 0.2,
               'angle_deg': scipy.stats.uniform(0, 90)},
              {'shape': 'square', 'side_length': 0.25}]
    domain = msp.geometry.Square(side_length=3)
    seeds = SeedList.from_info(phases, 0.5 * domain.area)
    seeds.position(domain)
    fname = str(tmp_path / 'seeds.txt')
    seeds.write(fname)
    loaded = SeedList.from_file(fname)
    assert loaded == seeds
    for s1, s2 in zip(seeds, loaded):
        assert np.allclose(s1.breakdown, s2.breakdown)
        assert s1.geometry == s2.geometry

    # seeds loaded from a file can be repositioned
    loaded[0].position = [1.0, 2.0]
    assert np.allclose(np.array(loaded[0].breakdown)[0, :2], [1.0, 2.0])
    loaded.position(domain, rng_seed=1)


def test_ellipsoid_seed_file_round_trip(tmp_path):
    phases = [{'shape': 'ellipsoid', 'size': 0.5, 'ratio_ab': 2,
               'ratio_ac': 1.5, 'orientation': 'random'},
              {'shape': 'sphere', 'size': 0.4}]
    seeds = SeedList.from_info(phases, 1.0)
    for i, s in enumerate(seeds):
        s.position = [0.1 * i, -0.2 * i, 0.3 * i]
    fname = str(tmp_path / 'seeds.txt')
    seeds.write(fname)
    loaded = SeedList.from_file(fname)
    assert loaded == seeds
    for s1, s2 in zip(seeds, loaded):
        if isinstance(s1.geometry, msp.geometry.Ellipsoid):
            assert np.allclose(s1.geometry.matrix, s2.geometry.matrix)
            assert np.allclose(s1.geometry.limits, s2.geometry.limits)


# --------------------------------------------------------------------------- #
# Positioning                                                                 #
# --------------------------------------------------------------------------- #
def test_position_random_axis_distribution():
    phases = [{'shape': 'circle', 'size': 0.3},
              {'shape': 'circle', 'size': 0.3}]
    domain = msp.geometry.Square(side_length=5)
    seeds = SeedList.from_info(phases, 0.3 * domain.area)
    pos_dists = {1: ['random', scipy.stats.norm(0, 0.3)]}
    seeds.position(domain, pos_dists=pos_dists, rng_seed=0)
    pos = np.array([s.position for s in seeds if s.phase == 1])
    assert pos[:, 0].std() > 1.0
    assert pos[:, 1].std() < 0.8
    assert np.abs(pos[:, 1]).mean() < 0.7


def test_sample_pos_within_raises_when_unreachable():
    domain = msp.geometry.Square(side_length=2)
    with pytest.raises(ValueError):
        sample_pos_within([5.0, 5.0], 3, domain, max_rounds=10)
    mvn = scipy.stats.multivariate_normal([50, 50], np.eye(2))
    with pytest.raises(ValueError):
        sample_pos_within(mvn, 3, domain, max_rounds=10)
    pts = sample_pos_within([scipy.stats.uniform(-1, 2),
                             scipy.stats.uniform(-1, 2)], 5, domain)
    assert pts.shape == (5, 2)
    assert np.all(domain.within(pts))


# --------------------------------------------------------------------------- #
# Plotting                                                                    #
# --------------------------------------------------------------------------- #
def test_plot_breakdown_3d_fresh_figure():
    seeds = SeedList([Seed.factory('sphere', r=0.5, position=(0, 0, 0)),
                      Seed.factory('ellipsoid', a=1, b=0.5, c=0.5,
                                   position=(2, 0, 0))])
    plt.figure()
    seeds.plot_breakdown()
    plt.close('all')
    plt.figure()
    seeds.plot()
    plt.close('all')


def test_plot_accepts_numpy_array_colors():
    seeds = SeedList([Seed.factory('circle', r=0.5, position=(i, 0))
                      for i in range(3)])
    plt.figure()
    seeds.plot(facecolors=np.array(['r', 'g', 'b']))
    colls = [c for c in plt.gca().collections
             if type(c).__name__ == 'EllipseCollection']
    assert len(colls[0].get_facecolors()) == 3
    plt.close('all')
