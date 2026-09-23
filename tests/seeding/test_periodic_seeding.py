"""Tests for the periodic placement of seeds."""
import itertools

import numpy as np
import pytest
import scipy.stats
from scipy.spatial import distance

import microstructpy as msp
from microstructpy import _misc
from microstructpy.seeding import Seed
from microstructpy.seeding import SeedList
from microstructpy.seeding.seedlist import _periodic_images


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _translations(lengths, per_axes):
    """All translations by 0 or +/- the domain length along periodic axes."""
    options = [[0.0, length, -length] if flag else [0.0]
               for length, flag in zip(lengths, per_axes)]
    return [np.array(t) for t in itertools.product(*options)]


def _worst_overlap(seeds, lengths, per_axes, rtol=0.0):
    """Largest violation of the overlap condition over all pairs of seeds,
    including the periodic images of the second seed (Eq. 3 of the paper).
    Positive values are overlaps beyond the tolerance."""
    bkdwns = [np.array(s.breakdown) for s in seeds]
    n_dim = bkdwns[0].shape[1] - 1
    worst = -np.inf
    for i, j in itertools.combinations(range(len(seeds)), 2):
        c_i, r_i = bkdwns[i][:, :n_dim], bkdwns[i][:, n_dim].reshape(-1, 1)
        c_j, r_j = bkdwns[j][:, :n_dim], bkdwns[j][:, n_dim].reshape(1, -1)
        for t in _translations(lengths, per_axes):
            dists = distance.cdist(c_i, c_j + t)
            viol = r_i + r_j - rtol * np.minimum(r_i, r_j) - dists
            worst = max(worst, viol.max())
    return worst


def _cross_face_overlap(seeds, lengths, per_axes):
    """Largest overlap of a seed with a periodic image (non-zero
    translation) of another seed."""
    bkdwns = [np.array(s.breakdown) for s in seeds]
    n_dim = bkdwns[0].shape[1] - 1
    worst = -np.inf
    for i, j in itertools.permutations(range(len(seeds)), 2):
        c_i, r_i = bkdwns[i][:, :n_dim], bkdwns[i][:, n_dim].reshape(-1, 1)
        c_j, r_j = bkdwns[j][:, :n_dim], bkdwns[j][:, n_dim].reshape(1, -1)
        for t in _translations(lengths, per_axes):
            if not np.any(t):
                continue
            dists = distance.cdist(c_i, c_j + t)
            worst = max(worst, (r_i + r_j - dists).max())
    return worst


# --------------------------------------------------------------------------- #
# Periodicity specification                                                   #
# --------------------------------------------------------------------------- #
def test_periodic_axes_parsing():
    assert _misc.periodic_axes(True, 2) == [True, True]
    assert _misc.periodic_axes(False, 3) == [False, False, False]
    assert _misc.periodic_axes(None, 3) == [False, False, False]
    assert _misc.periodic_axes('xy', 3) == [True, True, False]
    assert _misc.periodic_axes('z', 3) == [False, False, True]
    assert _misc.periodic_axes('x, z', 3) == [True, False, True]
    assert _misc.periodic_axes('True', 2) == [True, True]
    assert _misc.periodic_axes([True, False], 2) == [True, False]
    assert _misc.periodic_axes(np.array([0, 1, 1]), 3) == [False, True, True]
    with pytest.raises(ValueError):
        _misc.periodic_axes('xw', 3)
    with pytest.raises(ValueError):
        _misc.periodic_axes('z', 2)
    with pytest.raises(ValueError):
        _misc.periodic_axes([True], 2)


def test_periodic_images_translations():
    dom_lims = [(0.0, 2.0), (0.0, 3.0)]
    # crosses the -x face and the +y face: 3 images (edge, edge, corner)
    limits = [(-0.1, 0.5), (2.5, 3.2)]
    images = _periodic_images(limits, dom_lims, [True, True])
    assert sorted(images) == sorted([(2.0, 0.0), (0.0, -3.0), (2.0, -3.0)])
    # only x periodic: one image
    assert _periodic_images(limits, dom_lims, [True, False]) == [(2.0, 0.0)]
    # inside the domain: no images
    assert _periodic_images([(0.5, 1.0), (0.5, 1.0)], dom_lims,
                            [True, True]) == []
    # non-periodic domain
    assert _periodic_images(limits, None, [False, False],
                            include_zero=True) == [(0.0, 0.0)]


# --------------------------------------------------------------------------- #
# Positioning                                                                 #
# --------------------------------------------------------------------------- #
def _phases_2d():
    return [{'shape': 'circle', 'size': scipy.stats.uniform(0.15, 0.15)},
            {'shape': 'ellipse', 'size': scipy.stats.uniform(0.2, 0.15),
             'aspect_ratio': scipy.stats.uniform(1.5, 1.5),
             'angle_deg': scipy.stats.uniform(0, 180)}]


def test_periodic_position_2d_no_overlaps():
    domain = msp.geometry.Square(side_length=2.5, corner=(1, -1))
    lengths = domain.side_lengths

    seeds = SeedList.from_info(_phases_2d(), 0.6 * domain.area)
    seeds.position(domain, rtol=0.0, rng_seed=1, periodic=True)
    # seed centers stay inside the domain
    assert np.all(domain.within([s.position for s in seeds]))
    # no overlaps, including through the periodic faces
    assert _worst_overlap(seeds, lengths, [True, True]) <= 1e-9
    # some seeds do cross the faces (the test is not vacuous)
    lims = np.array([s.geometry.limits for s in seeds])
    dom = np.array(domain.limits)
    crossing = np.any((lims[:, :, 0] < dom[:, 0]) |
                      (lims[:, :, 1] > dom[:, 1]), axis=1)
    assert crossing.sum() > 0

    # without periodicity, the same packing overlaps through the faces
    seeds_np = SeedList.from_info(_phases_2d(), 0.6 * domain.area)
    seeds_np.position(domain, rtol=0.0, rng_seed=1)
    assert _cross_face_overlap(seeds_np, lengths, [True, True]) > 1e-6


def test_periodic_position_2d_single_axis():
    domain = msp.geometry.Rectangle(length=3, width=2)
    lengths = domain.side_lengths
    seeds = SeedList.from_info(_phases_2d(), 0.6 * domain.area)
    seeds.position(domain, rtol=0.0, rng_seed=2, periodic='x')
    # no overlaps through the x faces (translations along x only)
    assert _worst_overlap(seeds, lengths, [True, False]) <= 1e-9


def test_periodic_position_with_tolerance():
    domain = msp.geometry.Square(side_length=2)
    seeds = SeedList.from_info(_phases_2d(), domain.area)
    seeds.position(domain, rtol=0.3, rng_seed=3, periodic=[True, True])
    assert _worst_overlap(seeds, domain.side_lengths, [True, True],
                          rtol=0.3) <= 1e-9


def test_periodic_position_held_seed_images():
    domain = msp.geometry.Square(side_length=2)
    # a held seed in the corner, crossing the -x and -y faces
    held = Seed.factory('circle', r=0.3, position=(0.05, 0.05))
    others = SeedList.from_info([{'shape': 'circle', 'size': 0.3}],
                                0.5 * domain.area)
    others.position(domain, rtol=0.0, rng_seed=0, periodic=True)
    seeds = SeedList([held]) + others
    hold = [True] + [False for _ in others]
    seeds.position(domain, rtol=0.0, rng_seed=4, hold=hold,
                   periodic=True)
    assert np.allclose(seeds[0].position, [0.05, 0.05])
    assert _worst_overlap(seeds, domain.side_lengths, [True, True]) <= 1e-9


def test_periodic_position_3d():
    domain = msp.geometry.Cube(side_length=2)
    phases = [{'shape': 'sphere', 'size': scipy.stats.uniform(0.4, 0.3)}]
    seeds = SeedList.from_info(phases, 0.45 * domain.volume)
    seeds.position(domain, rtol=0.0, rng_seed=5, periodic=True)
    assert _worst_overlap(seeds, domain.side_lengths,
                          [True, True, True]) <= 1e-9
    seeds.position(domain, rtol=0.0, rng_seed=6, periodic='xz')
    assert _worst_overlap(seeds, domain.side_lengths,
                          [True, False, True]) <= 1e-9


def test_periodic_requires_rectangular_domain():
    seeds = SeedList.from_info([{'shape': 'circle', 'size': 0.3}], 2.0)
    with pytest.raises(ValueError):
        seeds.position(msp.geometry.Circle(r=1), periodic=True)
    with pytest.raises(ValueError):
        seeds.position(msp.geometry.Rectangle(length=2, width=2, angle=30),
                       periodic=True)
    # a non-periodic call on a circular domain still works
    seeds.position(msp.geometry.Circle(r=1), periodic=False)
