"""Tests for periodic polygonal and polyhedral meshes."""
import numpy as np
import pytest
import scipy.stats
from periodic_helpers import check_periodic_pairs
from periodic_helpers import seed_volumes
from periodic_helpers import tiled_reference_volumes

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


def _check_periodic_structure(pmesh, domain, per_axes):
    """Points and facets on the periodic faces are paired and are exact
    translates of each other, and the facets on the faces are wall
    facets."""
    assert pmesh.periodic_axes == list(per_axes)
    faces = check_periodic_pairs(pmesh.points, pmesh.facets,
                                 pmesh.periodic_points,
                                 pmesh.periodic_facets, per_axes, domain)
    for axis, (low, high, _) in faces.items():
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
    areas = seed_volumes(pmesh, 2)
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
    areas = seed_volumes(pmesh, len(seeds))
    ref = tiled_reference_volumes(seeds, domain, per_axes)
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
    # only boxes can be periodic in 3D
    seeds_3d = SeedList([Seed.factory('sphere', r=0.2,
                                      position=(0.5, 0.5, 0.5))])
    with pytest.raises(ValueError):
        PolyMesh.from_seeds(seeds_3d, msp.geometry.Sphere(r=1),
                            periodic=True)


# --------------------------------------------------------------------------- #
# 3D                                                                          #
# --------------------------------------------------------------------------- #


def test_two_spheres_periodic_in_x():
    domain = msp.geometry.Cube(side_length=1, corner=(0, 0, 0))
    seeds = SeedList([Seed.factory('sphere', r=0.2, position=(0.2, .5, .5)),
                      Seed.factory('sphere', r=0.2, position=(0.7, .5, .5))])
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='x')
    assert sorted(pmesh.seed_numbers) == [0, 0, 1]
    assert np.allclose(seed_volumes(pmesh, 2), [0.5, 0.5])
    pieces = sorted([v for v, s in zip(pmesh.volumes, pmesh.seed_numbers)
                     if s == 0])
    assert np.allclose(pieces, [0.05, 0.45])
    _check_periodic_structure(pmesh, domain, [True, False, False])


def test_single_sphere_tiles_the_cube():
    domain = msp.geometry.Cube(side_length=1, corner=(0, 0, 0))
    seeds = SeedList([Seed.factory('sphere', r=0.2, position=(.3, .6, .8))])
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=True)
    assert len(pmesh.regions) == 8
    assert len(pmesh.points) == 27
    assert len(pmesh.facets) == 36
    assert np.isclose(sum(pmesh.volumes), 1.0)
    _check_periodic_structure(pmesh, domain, [True, True, True])
    for axis in range(3):
        assert len(pmesh.periodic_points[axis]) == 9
        assert len(pmesh.periodic_facets[axis]) == 4


@pytest.mark.parametrize('per_axes', [[True, True, True],
                                      [True, False, True]])
def test_periodic_3d_matches_tiled_reference(per_axes):
    phases = [{'shape': 'sphere', 'size': scipy.stats.uniform(0.3, 0.2)},
              {'shape': 'ellipsoid', 'size': scipy.stats.uniform(0.35, 0.15),
               'ratio_ab': 2, 'ratio_ac': 1.5, 'orientation': 'random'}]
    domain = msp.geometry.Box(limits=[(0, 2), (-1, 1), (0.5, 2.5)])
    seeds = SeedList.from_info(phases, 0.4 * domain.volume)
    seeds.position(domain, rtol=0.0, rng_seed=1, periodic=per_axes)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=per_axes)

    assert np.isclose(sum(pmesh.volumes), domain.volume)
    assert np.all(np.array(pmesh.volumes) > 0)
    assert set(pmesh.seed_numbers) == set(range(len(seeds)))
    vols = seed_volumes(pmesh, len(seeds))
    ref = tiled_reference_volumes(seeds, domain, per_axes)
    # vertices within 1e-5 of the faces are snapped onto them
    assert np.allclose(vols, ref, rtol=1e-6, atol=1e-6)
    _check_periodic_structure(pmesh, domain, per_axes)


def test_periodic_3d_file_round_trip(tmp_path):
    domain = msp.geometry.Cube(side_length=2)
    phases = [{'shape': 'sphere', 'size': scipy.stats.uniform(0.4, 0.2)}]
    seeds = SeedList.from_info(phases, 0.4 * domain.volume)
    seeds.position(domain, rtol=0.0, rng_seed=2, periodic='xy')
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='xy')
    fname = str(tmp_path / 'polymesh.txt')
    pmesh.write(fname)
    loaded = PolyMesh.from_file(fname)
    assert loaded == pmesh
    assert loaded.periodic_axes == [True, True, False]
    assert {k: [tuple(p) for p in v] for k, v in
            loaded.periodic_points.items()} == pmesh.periodic_points
