"""Tests for periodic polygonal meshes (2D)."""
import copy
import itertools

import numpy as np
import pytest
import scipy.stats

import microstructpy as msp
from microstructpy.meshing import PolyMesh
from microstructpy.meshing.polymesh import kp_loop
from microstructpy.seeding import Seed
from microstructpy.seeding import SeedList


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _phases():
    return [{'shape': 'circle', 'size': scipy.stats.uniform(0.15, 0.15)},
            {'shape': 'ellipse', 'size': scipy.stats.uniform(0.2, 0.15),
             'aspect_ratio': scipy.stats.uniform(1.5, 1.5),
             'angle_deg': scipy.stats.uniform(0, 180)}]


def _periodic_seeds(domain, per_axes, rng_seed=0, fill=0.55):
    seeds = SeedList.from_info(_phases(), fill * domain.area)
    seeds.position(domain, rtol=0.0, rng_seed=rng_seed, periodic=per_axes)
    return seeds


def _seed_areas(pmesh, n_seeds):
    areas = np.zeros(n_seeds)
    for seed_num, vol in zip(pmesh.seed_numbers, pmesh.volumes):
        areas[seed_num] += vol
    return areas


def _tiled_reference_areas(seeds, domain, per_axes):
    """Areas of the cells of the seeds in a periodic tessellation, computed
    as a non-periodic tessellation of the seeds tiled across the periodic
    axes (3 copies per periodic axis)."""
    lims = np.array(domain.limits)
    lengths = lims[:, 1] - lims[:, 0]
    options = [[-length, 0.0, length] if flag else [0.0]
               for length, flag in zip(lengths, per_axes)]
    tiled = SeedList()
    for t in itertools.product(*options):
        for seed in seeds:
            copy_seed = copy.deepcopy(seed)
            copy_seed.position = list(np.array(seed.position) + np.array(t))
            tiled.append(copy_seed)
    n_seeds = len(seeds)
    big_lims = [(lb - length, ub + length) if flag else (lb, ub)
                for (lb, ub), length, flag in zip(lims, lengths, per_axes)]
    big_domain = msp.geometry.Rectangle(limits=big_lims)
    pmesh = PolyMesh.from_seeds(tiled, big_domain)
    # the cells of the original copies (the zero translation)
    i_zero = [i for i, t in enumerate(itertools.product(*options))
              if not any(t)][0]
    areas = np.zeros(n_seeds)
    for seed_num, vol in zip(pmesh.seed_numbers, pmesh.volumes):
        block, local = divmod(seed_num, n_seeds)
        if block == i_zero:
            areas[local] += vol
    return areas


def _check_periodic_structure(pmesh, domain, per_axes):
    """Points and facets on the periodic faces are paired and are exact
    translates of each other."""
    pts = np.array(pmesh.points)
    lims = np.array(domain.limits)
    lengths = lims[:, 1] - lims[:, 0]
    assert pmesh.periodic_axes == list(per_axes)
    for axis, flag in enumerate(per_axes):
        if not flag:
            assert axis not in pmesh.periodic_points
            continue
        lb, ub = lims[axis]
        shift = np.zeros(2)
        shift[axis] = lengths[axis]
        pairs = pmesh.periodic_points[axis]
        low = set(np.nonzero(np.isclose(pts[:, axis], lb))[0])
        high = set(np.nonzero(np.isclose(pts[:, axis], ub))[0])
        assert len(pairs) == len(low) == len(high)
        assert set([lo for lo, _ in pairs]) == low
        assert set([hi for _, hi in pairs]) == high
        for lo, hi in pairs:
            assert np.array_equal(pts[hi], pts[lo] + shift)
        # facets on the lower face are paired with facets on the upper face
        kp_map = dict(pairs)
        f_pairs = dict(pmesh.periodic_facets[axis])
        for f_num, facet in enumerate(pmesh.facets):
            if all([kp in low for kp in facet]):
                assert f_num in f_pairs
                image = pmesh.facets[f_pairs[f_num]]
                assert set(image) == set([kp_map[kp] for kp in facet])
        # every boundary facet on the faces is a wall facet
        for f_num, neighs in enumerate(pmesh.facet_neighbors):
            facet = pmesh.facets[f_num]
            if all([kp in low for kp in facet]):
                assert min(neighs) == -(2 * axis + 1)
            if all([kp in high for kp in facet]):
                assert min(neighs) == -(2 * axis + 2)


def _region_loops(pmesh):
    pts = np.array(pmesh.points)
    return [pts[kp_loop([pmesh.facets[f] for f in r])]
            for r in pmesh.regions]


# --------------------------------------------------------------------------- #
# Analytic cases                                                              #
# --------------------------------------------------------------------------- #
def test_two_seeds_periodic_in_x():
    domain = msp.geometry.Square(side_length=1, corner=(0, 0))
    seeds = SeedList([Seed.factory('circle', r=0.2, position=(0.1, 0.5)),
                      Seed.factory('circle', r=0.2, position=(0.6, 0.5))])
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='x')

    # cell 0 is cut by the periodic face: [0, 0.35] and [0.85, 1]
    assert len(pmesh.regions) == 3
    assert sorted(pmesh.seed_numbers) == [0, 0, 1]
    areas = _seed_areas(pmesh, 2)
    assert np.allclose(areas, [0.5, 0.5])
    pieces = sorted([v for v, s in zip(pmesh.volumes, pmesh.seed_numbers)
                     if s == 0])
    assert np.allclose(pieces, [0.15, 0.35])
    _check_periodic_structure(pmesh, domain, [True, False])
    assert len(pmesh.periodic_points[0]) == 2
    assert len(pmesh.periodic_facets[0]) == 1


def test_single_seed_tiles_the_domain():
    domain = msp.geometry.Square(side_length=1, corner=(0, 0))
    seeds = SeedList([Seed.factory('circle', r=0.2, position=(0.3, 0.7))])
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=True)
    assert len(pmesh.regions) == 4
    assert np.isclose(sum(pmesh.volumes), 1.0)
    assert np.allclose(sorted(pmesh.volumes), [0.04, 0.16, 0.16, 0.64])
    _check_periodic_structure(pmesh, domain, [True, True])


def test_non_periodic_unchanged():
    domain = msp.geometry.Square(side_length=1, corner=(0, 0))
    seeds = SeedList([Seed.factory('circle', r=0.2, position=(0.1, 0.5)),
                      Seed.factory('circle', r=0.2, position=(0.6, 0.5))])
    pmesh = PolyMesh.from_seeds(seeds, domain)
    assert pmesh.periodic_axes is None
    assert pmesh.periodic_points is None
    assert len(pmesh.regions) == 2
    assert np.allclose(sorted(pmesh.volumes), [0.35, 0.65])


# --------------------------------------------------------------------------- #
# Random microstructures against a tiled reference                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('per_axes', [[True, True], [True, False],
                                      [False, True]])
def test_periodic_matches_tiled_reference(per_axes):
    domain = msp.geometry.Rectangle(limits=[(-1, 2), (0.5, 2.5)])
    seeds = _periodic_seeds(domain, per_axes, rng_seed=1)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=per_axes)

    assert np.isclose(sum(pmesh.volumes), domain.area)
    assert np.all(np.array(pmesh.volumes) > 0)
    assert set(pmesh.seed_numbers) == set(range(len(seeds)))
    areas = _seed_areas(pmesh, len(seeds))
    ref = _tiled_reference_areas(seeds, domain, per_axes)
    assert np.allclose(areas, ref, rtol=1e-9, atol=1e-12)
    _check_periodic_structure(pmesh, domain, per_axes)

    # every piece is a convex polygon inside the domain
    lims = np.array(domain.limits)
    for loop in _region_loops(pmesh):
        assert np.all(loop >= lims[:, 0] - 1e-9)
        assert np.all(loop <= lims[:, 1] + 1e-9)
        d_edge = np.roll(loop, -1, axis=0) - loop
        d_next = np.roll(d_edge, -1, axis=0)
        cross = d_edge[:, 0] * d_next[:, 1] - d_edge[:, 1] * d_next[:, 0]
        assert np.all(cross >= -1e-12) or np.all(cross <= 1e-12)


def test_periodic_seeds_crossing_faces_are_split():
    domain = msp.geometry.Square(side_length=2)
    seeds = _periodic_seeds(domain, [True, True], rng_seed=2)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=True)
    lims = np.array(domain.limits)
    n_pieces = np.bincount(pmesh.seed_numbers, minlength=len(seeds))
    # seeds crossing a face have more than one piece
    crossing = 0
    for seed_num, seed in enumerate(seeds):
        s_lims = np.array(seed.geometry.limits)
        if np.any(s_lims[:, 0] < lims[:, 0]) or \
                np.any(s_lims[:, 1] > lims[:, 1]):
            crossing += 1
            assert n_pieces[seed_num] >= 2
    assert crossing > 0


# --------------------------------------------------------------------------- #
# Files and errors                                                            #
# --------------------------------------------------------------------------- #
def test_periodic_file_round_trip(tmp_path):
    domain = msp.geometry.Square(side_length=2)
    seeds = _periodic_seeds(domain, [True, False], rng_seed=3, fill=0.5)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='x')
    fname = str(tmp_path / 'polymesh.txt')
    pmesh.write(fname)
    loaded = PolyMesh.from_file(fname)
    assert loaded == pmesh
    assert np.array_equal(np.array(loaded.points), np.array(pmesh.points))
    assert loaded.periodic_axes == pmesh.periodic_axes
    assert {k: [tuple(p) for p in v] for k, v in
            loaded.periodic_points.items()} == pmesh.periodic_points
    assert {k: [tuple(p) for p in v] for k, v in
            loaded.periodic_facets.items()} == pmesh.periodic_facets

    # a non-periodic mesh has no periodic sections
    pmesh_np = PolyMesh.from_seeds(seeds, domain)
    assert 'Periodic' not in str(pmesh_np)


def test_periodic_errors():
    seeds = SeedList([Seed.factory('circle', r=0.2, position=(0.5, 0.5))])
    with pytest.raises(ValueError):
        PolyMesh.from_seeds(seeds, msp.geometry.Circle(r=1), periodic=True)
    seeds_3d = SeedList([Seed.factory('sphere', r=0.2,
                                      position=(0.5, 0.5, 0.5))])
    with pytest.raises(NotImplementedError):
        PolyMesh.from_seeds(seeds_3d, msp.geometry.Cube(side_length=2),
                            periodic=True)
