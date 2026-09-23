"""Tests for the PolyMesh bug fixes.

Each test covers one of the fixes to :mod:`microstructpy.meshing.polymesh`:
the area of a vertex loop, the clipping of cells to non-rectangular 2D
domains, the short-edge optimization, the precision of the text file,
the poly file writer, the segment/boundary crossing at large coordinates,
3D plotting on a fresh figure, per-region numpy arrays in plot keyword
arguments, and the silence of the mesh comparison.
"""
from __future__ import division

import copy
import os
import time

import numpy as np
from matplotlib import path as mpath
from matplotlib import pyplot as plt

from microstructpy import geometry
from microstructpy.meshing.polymesh import PolyMesh
from microstructpy.meshing.polymesh import _edge_lengths
from microstructpy.meshing.polymesh import _loop_area
from microstructpy.meshing.polymesh import _segment_cross
from microstructpy.meshing.polymesh import _shortest_edge
from microstructpy.meshing.polymesh import kp_loop
from microstructpy.seeding import Seed
from microstructpy.seeding import SeedList


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _circle_seeds(positions, r=0.3):
    """Circle seeds at the given positions (breakdowns translated)."""
    seeds = SeedList([])
    for pos in positions:
        seed = Seed.factory('circle', r=r)
        seed.position = list(pos)
        seeds.append(seed)
    return seeds


def _random_seeds(domain, n_seeds, r=0.1, rng_seed=0):
    seeds = SeedList([Seed.factory('circle', r=r) for _ in range(n_seeds)])
    seeds.position(domain, rng_seed=rng_seed)
    return seeds


def _region_paths(pmesh):
    pts = np.array(pmesh.points)
    loops = [kp_loop([pmesh.facets[f] for f in r]) for r in pmesh.regions]
    return [mpath.Path(pts[loop]) for loop in loops]


def _random_interior_points(domain, n_pts, frac=0.9, rng_seed=1):
    """Random points within ``frac`` of the domain boundary."""
    rng = np.random.RandomState(rng_seed)
    pts = []
    while len(pts) < n_pts:
        pt = rng.uniform(-1, 1, 2)
        if np.linalg.norm(pt) >= frac:
            continue
        if isinstance(domain, geometry.Ellipse):
            pt = domain.matrix.dot(pt * np.array([domain.a, domain.b]))
        else:
            pt = pt * domain.r
        pts.append(np.array(domain.center) + pt)
    return pts


def _check_partition(pmesh, domain, n_seeds):
    """Check that the mesh partitions the domain."""
    pts = np.array(pmesh.points)
    vols = np.array(pmesh.volumes)
    assert len(pmesh.regions) == n_seeds
    assert set(pmesh.seed_numbers) == set(range(n_seeds))
    assert np.all(np.isfinite(pts))
    assert np.all(vols > 0)
    assert abs(vols.sum() - domain.area) < 0.02 * domain.area

    # the stored volumes match the polygons
    recomputed = PolyMesh(pmesh.points, pmesh.facets, pmesh.regions).volumes
    assert np.allclose(vols, recomputed)

    # each interior point is in exactly one region
    paths = _region_paths(pmesh)
    for pt in _random_interior_points(domain, 200):
        n_in = sum([path.contains_point(pt) for path in paths])
        assert n_in == 1


def _min_edge_length(pmesh):
    edge_lens = _edge_lengths(pmesh)
    return edge_lens[_shortest_edge(edge_lens)]['length']


# --------------------------------------------------------------------------- #
# 1. _loop_area                                                               #
# --------------------------------------------------------------------------- #
def test_loop_area_unordered_points():
    # unit square, vertices stored out of loop order
    pts = [(0, 0), (1, 1), (1, 0), (0, 1)]
    loop = [0, 2, 1, 3]
    assert np.isclose(_loop_area(pts, loop), 1.0)
    assert np.isclose(_loop_area(pts, loop[::-1]), 1.0)


# --------------------------------------------------------------------------- #
# 2. Clipping cells to circular and elliptical domains                        #
# --------------------------------------------------------------------------- #
def test_clip_cells_without_interior_vertices():
    # the top and bottom cells have no vertex within the circle
    domain = geometry.Circle(r=1)
    seeds = _circle_seeds([(0, 0.4), (0, -0.4), (0, 0)])
    pmesh = PolyMesh.from_seeds(seeds, domain)

    assert len(pmesh.regions) == 3
    assert sorted(pmesh.seed_numbers) == [0, 1, 2]
    assert np.all(np.isfinite(np.array(pmesh.points)))
    assert all([len(r) >= 3 for r in pmesh.regions])
    assert np.all(np.array(pmesh.volumes) > 0)

    # the boundary arcs are approximated by chords (one per gap in the
    # cell), which under-estimates the area - significantly with only
    # three cells
    total_area = sum(pmesh.volumes)
    assert 0.7 * np.pi < total_area < np.pi


def test_clip_domain_within_cell():
    # a single seed: the Voronoi cell is the bounding box of the circle
    domain = geometry.Circle(r=2)
    pmesh = PolyMesh.from_seeds(_circle_seeds([(0, 0)]), domain)

    assert len(pmesh.regions) == 1
    assert abs(pmesh.volumes[0] - 4 * np.pi) < 0.01 * 4 * np.pi
    assert len(pmesh.regions[0]) >= 3
    assert np.allclose(np.linalg.norm(np.array(pmesh.points), axis=1), 2)


def test_clip_single_edge_crossing():
    # the cell of the second seed is a circular segment: the circle crosses
    # a single edge of the cell twice
    domain = geometry.Circle(r=1)
    seeds = _circle_seeds([(0, 0), (0, -0.99)], r=0.3)
    pmesh = PolyMesh.from_seeds(seeds, domain)

    assert len(pmesh.regions) == 2
    assert sorted(pmesh.seed_numbers) == [0, 1]
    assert all([len(r) >= 3 for r in pmesh.regions])
    assert np.all(np.array(pmesh.volumes) > 0)


def test_clip_partition_circle():
    domain = geometry.Circle(r=1)
    n_seeds = 60
    seeds = _random_seeds(domain, n_seeds)
    pmesh = PolyMesh.from_seeds(seeds, domain)
    _check_partition(pmesh, domain, n_seeds)


def test_clip_partition_ellipse():
    domain = geometry.Ellipse(a=1.5, b=1)
    n_seeds = 100
    seeds = _random_seeds(domain, n_seeds)
    pmesh = PolyMesh.from_seeds(seeds, domain)
    _check_partition(pmesh, domain, n_seeds)


def test_clip_partition_rotated_ellipse():
    domain = geometry.Ellipse(a=1.5, b=1, angle=30, center=(0.3, -0.2))
    n_seeds = 100
    seeds = _random_seeds(domain, n_seeds)
    pmesh = PolyMesh.from_seeds(seeds, domain)
    _check_partition(pmesh, domain, n_seeds)


def test_rectangular_domain_unchanged():
    domain = geometry.Square(side_length=2)
    seeds = _random_seeds(domain, 30)
    pmesh = PolyMesh.from_seeds(seeds, domain)

    assert len(pmesh.regions) == 30
    assert abs(sum(pmesh.volumes) - domain.area) < 1e-10


# --------------------------------------------------------------------------- #
# 3. Short edge optimization                                                  #
# --------------------------------------------------------------------------- #
def test_edge_opt(capsys):
    np.random.seed(0)
    domain = geometry.Square(side_length=2)
    seeds = _random_seeds(domain, 12, r=0.15, rng_seed=3)
    seeds_orig = copy.deepcopy(seeds)

    pmesh_0 = PolyMesh.from_seeds(seeds, domain)
    min_len_0 = _min_edge_length(pmesh_0)

    capsys.readouterr()
    pmesh = PolyMesh.from_seeds(seeds, domain, edge_opt=True, n_iter=10,
                                verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''

    # the minimum edge length does not decrease
    assert _min_edge_length(pmesh) >= min_len_0

    # the seeds are in the accepted state: re-tessellating them
    # reproduces the returned mesh
    pmesh_re = PolyMesh.from_seeds(seeds, domain)
    assert len(pmesh_re.regions) == len(pmesh.regions)
    assert np.allclose(np.sort(pmesh_re.volumes), np.sort(pmesh.volumes),
                       rtol=0, atol=1e-9)

    # the seeds were displaced rigidly
    for seed in seeds:
        assert np.allclose(seed.breakdown[0][:-1], seed.position)
        assert np.allclose(seed.geometry.center, seed.position)

    # the seeds were displaced (the mesh changed)
    n_moved = sum([not np.allclose(s.position, s0.position)
                   for s, s0 in zip(seeds, seeds_orig)])
    assert n_moved > 0
    assert n_moved <= 2 * 10 * 2  # at most 2 seeds per accepted trial


# --------------------------------------------------------------------------- #
# 4. Full precision text files                                                #
# --------------------------------------------------------------------------- #
def test_write_read_full_precision(tmp_path):
    domain = geometry.Square(side_length=1.7)
    seeds = _random_seeds(domain, 15, r=0.1)
    pmesh = PolyMesh.from_seeds(seeds, domain)

    filename = str(tmp_path / 'polymesh.txt')
    pmesh.write(filename)
    pmesh_rw = PolyMesh.from_file(filename)

    assert np.array_equal(np.array(pmesh.points, dtype='float'),
                          np.array(pmesh_rw.points, dtype='float'))
    assert np.array_equal(np.array(pmesh.volumes, dtype='float'),
                          np.array(pmesh_rw.volumes, dtype='float'))
    assert pmesh == pmesh_rw


# --------------------------------------------------------------------------- #
# 5. Poly file writer                                                         #
# --------------------------------------------------------------------------- #
def test_write_poly(tmp_path):
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (1.5, 0)]
    facets = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 2)]
    regions = [(0, 1, 2, 3), (4, 5, 1)]
    pmesh = PolyMesh(pts, facets, regions)

    filename = str(tmp_path / 'polymesh.poly')
    pmesh.write(filename, format='poly')

    assert os.path.exists(filename)
    with open(filename, 'r') as file:
        lines = [ln for ln in file.read().split('\n') if ln.strip()]
    data_lines = [ln for ln in lines if not ln.startswith('#')]
    assert int(data_lines[0].split()[0]) == len(pts)


# --------------------------------------------------------------------------- #
# 6. Segment crossing at large coordinates                                    #
# --------------------------------------------------------------------------- #
def test_segment_cross_large_coordinates():
    center = np.array([1e5, 1e5])
    domain = geometry.Circle(center=center, r=1)
    pts = [center, center + np.array([2, 0])]

    t_start = time.time()
    crossing = _segment_cross(pts, domain)
    elapsed = time.time() - t_start

    expected = center + np.array([1, 0])
    assert np.linalg.norm(crossing - expected) < 1e-9 * np.linalg.norm(
        expected)
    assert elapsed < 1

    # the result does not depend on the order of the end points
    assert np.array_equal(_segment_cross(pts[::-1], domain), crossing)


# --------------------------------------------------------------------------- #
# 7. 3D plot on a fresh figure                                                #
# --------------------------------------------------------------------------- #
def test_plot_3d_fresh_figure():
    domain = geometry.Cube(side_length=1)
    positions = [(-0.2, -0.2, -0.2), (0.2, 0.2, 0.2), (0.2, -0.2, 0.1)]
    seeds = SeedList([Seed.factory('sphere', r=0.25, position=p)
                      for p in positions])
    pmesh = PolyMesh.from_seeds(seeds, domain)
    plt.close('all')
    plt.figure()
    try:
        pmesh.plot()
        assert plt.gca().name == '3d'
    finally:
        plt.close('all')


# --------------------------------------------------------------------------- #
# 8. Per-region numpy arrays in plot keyword arguments                        #
# --------------------------------------------------------------------------- #
def test_plot_facecolors_array():
    domain = geometry.Square(side_length=2)
    seeds = _random_seeds(domain, 8, r=0.15)
    phases = [0, 0, 0, 1, 1, 1, 1, 1]
    for seed, phase in zip(seeds, phases):
        seed.phase = phase
    pmesh = PolyMesh.from_seeds(seeds, domain)
    n_regions = len(pmesh.regions)

    # one colour per region, indexed by seed
    colors = np.array([plt.cm.viridis(i / n_regions)
                       for i in range(n_regions)])
    plt.close('all')
    plt.figure()
    try:
        pmesh.plot(facecolors=colors)
        collection = plt.gca().collections[-1]
        assert len(collection.get_facecolors()) == n_regions
        assert np.allclose(collection.get_facecolors(),
                           colors[pmesh.seed_numbers])
    finally:
        plt.close('all')

    # one colour per material phase
    colors = np.array([[1, 0, 0, 1], [0, 0, 1, 1]], dtype='float')
    plt.figure()
    try:
        pmesh.plot(index_by='material', facecolors=colors)
        collection = plt.gca().collections[-1]
        assert len(collection.get_facecolors()) == n_regions
        assert np.allclose(collection.get_facecolors(),
                           colors[pmesh.phase_numbers])
    finally:
        plt.close('all')


# --------------------------------------------------------------------------- #
# 9. Mesh comparison is silent                                                #
# --------------------------------------------------------------------------- #
def test_eq_silent(capsys):
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (1.5, 0)]
    facets = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 2)]
    regions = [(0, 1, 2, 3), (4, 5, 1)]
    pmesh = PolyMesh(pts, facets, regions, [0, 1], [2, 2])
    pmesh_2 = PolyMesh(pts, facets, [(0, 3, 2, 1), (4, 1, 5)], [0, 1],
                       [2, 2])
    pmesh_3 = PolyMesh(pts, facets, regions, [0, 1], [2, 3])

    capsys.readouterr()
    assert pmesh == pmesh_2
    assert pmesh != pmesh_3
    assert pmesh != pts
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''
