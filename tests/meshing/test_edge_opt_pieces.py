"""Tests of the edge optimization with thin pieces at the periodic faces."""
import copy

import numpy as np

from microstructpy import geometry
from microstructpy.meshing import PolyMesh
from microstructpy.meshing.polymesh import _accept_trial
from microstructpy.meshing.polymesh import _displace_seed
from microstructpy.meshing.polymesh import _edge_lengths
from microstructpy.meshing.polymesh import _mesh_features
from microstructpy.meshing.polymesh import _nearest_image
from microstructpy.meshing.polymesh import _select_target
from microstructpy.meshing.polymesh import _wedge_geometry
from microstructpy.seeding import Seed
from microstructpy.seeding import SeedList


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _grid_seeds_2d(x_shift, rng_seed=0):
    """Circles on a 3 x 3 grid (spacing 1, at 0.75, 1.75, 2.75) in a
    periodic square of side 3: the cells cross the faces by 0.25. The seed
    at (0.75, 1.75) is moved to ``x_shift``: its cell then crosses the
    face x = 0 by about (0.25 - x_shift) / 2 and leaves a piece that thick
    on the face x = 3. The other rows are jittered to avoid degenerate
    vertices."""
    rng = np.random.RandomState(rng_seed)
    seeds = SeedList()
    for i in range(3):
        for j in range(3):
            x, y = 0.75 + i, 0.75 + j
            if (i, j) == (0, 1):
                x = x_shift
            else:
                y += 0.03 * (2 * rng.rand() - 1)
            seeds.append(Seed.factory('circle', r=0.2, position=[x, y]))
    return seeds


def _grid_seeds_3d(x_shift, rng_seed=0):
    """Spheres on a 2 x 2 x 2 grid (at 0.75 and 1.75) in a periodic cube
    of side 2, with the seed at (0.75, 0.75, 0.75) moved to ``x_shift``."""
    rng = np.random.RandomState(rng_seed)
    seeds = SeedList()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x, y, z = 0.75 + i, 0.75 + j, 0.75 + k
                if (i, j, k) == (0, 0, 0):
                    x = x_shift
                else:
                    y += 0.03 * (2 * rng.rand() - 1)
                    z += 0.03 * (2 * rng.rand() - 1)
                seeds.append(Seed.factory('sphere', r=0.2,
                                          position=[x, y, z]))
    return seeds


def _pieces(pmesh, domain):
    n_dim = domain.n_dim
    scale = max([ub - lb for lb, ub in domain.limits])
    feats = _mesh_features(pmesh, [True] * n_dim, domain.limits, scale)
    return [f for f in feats if f['kind'] == 'piece']


def _min_edge(pmesh):
    return min([e['length'] for e in _edge_lengths(pmesh).values()])


def _feature(kind, size, key, seeds=()):
    return {'kind': kind, 'size': size, 'key': key, 'seeds': list(seeds)}


# --------------------------------------------------------------------------- #
# Features                                                                    #
# --------------------------------------------------------------------------- #
def test_piece_features_2d():
    domain = geometry.Square(side_length=3, corner=(0, 0))
    seeds = _grid_seeds_2d(0.05)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=True)
    pieces = _pieces(pmesh, domain)

    # the shifted seed (number 1) has a piece about 0.1 thick on the face
    # x = 3 (the part of its cell beyond x = 0) and its main part on the
    # face x = 0, about 0.9 thick; the other pieces are about 0.25 thick
    thin = [p for p in pieces if p['seed'] == 1 and p['axis'] == 0
            and p['side'] == 1]
    assert len(thin) == 1
    assert np.isclose(thin[0]['size'], 0.1, atol=0.02)
    assert 1 in thin[0]['seeds']
    assert len(thin[0]['seeds']) > 1  # the neighbors across its facets
    main = [p for p in pieces if p['seed'] == 1 and p['axis'] == 0
            and p['side'] == 0]
    assert len(main) == 1
    assert np.isclose(main[0]['size'], 0.9, atol=0.02)
    others = [p['size'] for p in pieces if p['seed'] != 1]
    assert np.all(np.array(others) > 0.15)

    # every piece touches its face, and the keys are distinct
    pts = np.array(pmesh.points)
    lims = np.array(domain.limits)
    for p in pieces:
        assert p['size'] > 0
    keys = [p['key'] for p in pieces]
    assert len(set(keys)) == len(keys)
    for f, neighs in enumerate(pmesh.facet_neighbors):
        if min(neighs) < 0:
            axis, side = divmod(-min(neighs) - 1, 2)
            assert np.allclose(pts[pmesh.facets[f], axis], lims[axis, side])

    # edge features cover every edge of the mesh
    scale = 3.0
    feats = _mesh_features(pmesh, [True, True], domain.limits, scale)
    n_edges = len([f for f in feats if f['kind'] == 'edge'])
    assert n_edges == len(_edge_lengths(pmesh))


def test_no_piece_features_without_periodicity():
    domain = geometry.Square(side_length=3, corner=(0, 0))
    seeds = _grid_seeds_2d(0.05)
    pmesh = PolyMesh.from_seeds(seeds, domain)
    feats = _mesh_features(pmesh, [False, False], None, 3.0)
    assert all([f['kind'] == 'edge' for f in feats])
    # pieces on the periodic axes only
    pmesh_x = PolyMesh.from_seeds(seeds, domain, periodic='x')
    feats = _mesh_features(pmesh_x, [True, False], domain.limits, 3.0)
    assert all([f['axis'] == 0 for f in feats if f['kind'] == 'piece'])


# --------------------------------------------------------------------------- #
# Acceptance and target selection                                             #
# --------------------------------------------------------------------------- #
def test_accept_trial():
    old = [_feature('edge', 0.5, 'a'), _feature('edge', 0.02, 'b'),
           _feature('piece', 0.01, 'c'), _feature('edge', 0.3, 'd')]
    # the thin piece is replaced by longer features: accepted
    new = [_feature('edge', 0.5, 'a'), _feature('edge', 0.02, 'b'),
           _feature('piece', 0.05, 'e'), _feature('edge', 0.25, 'f')]
    assert _accept_trial(new, old, 1e-9)
    # a feature shorter than the removed ones appears: rejected
    new = [_feature('edge', 0.5, 'a'), _feature('edge', 0.02, 'b'),
           _feature('piece', 0.05, 'e'), _feature('edge', 0.005, 'f')]
    assert not _accept_trial(new, old, 1e-9)
    # nothing changes: rejected
    assert not _accept_trial(copy.deepcopy(old), old, 1e-9)
    # a feature is only removed: accepted; only added: rejected
    assert _accept_trial(old[:3], old, 1e-9)
    assert not _accept_trial(old + [_feature('edge', 0.9, 'g')], old, 1e-9)
    # a removed and an added feature of the same size cancel out, so the
    # comparison is between the piece (0.01) and the new edge (0.015)
    new = [_feature('edge', 0.5, 'a'), _feature('edge', 0.02, 'b2'),
           _feature('edge', 0.015, 'e'), _feature('edge', 0.3, 'd')]
    assert _accept_trial(new, old, 1e-9)
    new[2]['size'] = 0.009
    assert not _accept_trial(new, old, 1e-9)


def test_select_target():
    feats = [_feature('edge', 0.5, 'a'), _feature('edge', 0.02, 'b'),
             _feature('piece', 0.03, 'c'), _feature('piece', 0.08, 'd'),
             _feature('piece', 0.2, 'e')]
    # without a margin: the shortest feature only
    assert _select_target(feats, 0.0, set())['key'] == 'b'
    assert _select_target(feats, 0.0, {'b'}) is None
    # with a margin: the shortest feature, then the pieces under the
    # margin from the thinnest
    assert _select_target(feats, 0.1, set())['key'] == 'b'
    assert _select_target(feats, 0.1, {'b'})['key'] == 'c'
    assert _select_target(feats, 0.1, {'b', 'c'})['key'] == 'd'
    assert _select_target(feats, 0.1, {'b', 'c', 'd'}) is None
    # a piece that is the shortest feature is the target even without a
    # margin
    feats[2]['size'] = 0.01
    assert _select_target(feats, 0.0, set())['key'] == 'c'


def test_displace_seed_wraps():
    seed = Seed.factory('circle', r=0.2, position=[2.9, 1.0])
    seed.update_breakdown()
    dom_lims = [(0.0, 3.0), (0.0, 3.0)]
    _displace_seed(seed, [0.3, -1.2], dom_lims, [True, False])
    assert np.allclose(seed.position, [0.2, -0.2])
    assert np.allclose(seed.geometry.center, seed.position)
    assert np.allclose(seed.breakdown[0][:-1], seed.position)
    # without periodicity, no wrapping
    _displace_seed(seed, [3.0, 0.0])
    assert np.allclose(seed.position, [3.2, -0.2])


def test_nearest_image():
    dom_lims = [(0.0, 3.0), (0.0, 3.0)]
    pts = np.array([[2.9, 1.0], [2.8, 1.5]])
    ref = np.array([0.1, 1.2])
    image = _nearest_image(pts, ref, dom_lims, [True, True])
    assert np.allclose(image, [[-0.1, 1.0], [-0.2, 1.5]])
    image = _nearest_image(pts, ref, dom_lims, [False, True])
    assert np.allclose(image, pts)


# --------------------------------------------------------------------------- #
# Optimization                                                                #
# --------------------------------------------------------------------------- #
def test_edge_opt_fixes_thin_piece_2d():
    np.random.seed(0)
    domain = geometry.Square(side_length=3, corner=(0, 0))
    margin = 0.1
    seeds = _grid_seeds_2d(0.2)
    seeds_orig = copy.deepcopy(seeds)
    pmesh_0 = PolyMesh.from_seeds(seeds, domain, periodic=True)
    thin_0 = [p['size'] for p in _pieces(pmesh_0, domain)
              if p['size'] < margin]
    assert len(thin_0) == 1
    assert np.isclose(thin_0[0], 0.025, atol=0.01)
    min_edge_0 = _min_edge(pmesh_0)

    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=True, edge_opt=True,
                                n_iter=2, periodic_margin=margin)

    # no piece thinner than the margin is left, and the shortest edge of
    # the mesh did not get shorter
    thin = [p['size'] for p in _pieces(pmesh, domain) if p['size'] < margin]
    assert thin == []
    assert _min_edge(pmesh) >= min_edge_0 - 1e-9

    # the mesh is periodic and the seeds reproduce it
    assert pmesh.periodic_axes == [True, True]
    pmesh_re = PolyMesh.from_seeds(seeds, domain, periodic=True)
    assert np.allclose(np.sort(pmesh_re.volumes), np.sort(pmesh.volumes),
                       rtol=0, atol=1e-9)
    assert np.isclose(sum(pmesh.volumes), domain.area)

    # the shifted seed moved (into the domain, normal to the face) and
    # every seed is in the domain
    assert not np.allclose(seeds[1].position, seeds_orig[1].position)
    assert np.isclose(seeds[1].position[1], seeds_orig[1].position[1])
    for seed in seeds:
        assert np.all(np.array(seed.position) >= 0)
        assert np.all(np.array(seed.position) <= 3)
        assert np.allclose(seed.geometry.center, seed.position)


def test_edge_opt_fixes_thin_piece_3d():
    np.random.seed(0)
    domain = geometry.Cube(side_length=2, corner=(0, 0, 0))
    margin = 0.1
    seeds = _grid_seeds_3d(0.2)
    pmesh_0 = PolyMesh.from_seeds(seeds, domain, periodic=True)
    # the piece of the shifted seed, and a corner of its neighbor along x
    # that pokes through the face x = 2 (the bisector with the image of
    # the shifted seed is tilted by the jitter)
    thin_0 = [p['size'] for p in _pieces(pmesh_0, domain)
              if p['size'] < margin]
    assert len(thin_0) == 2
    assert min(thin_0) < 0.01
    min_edge_0 = _min_edge(pmesh_0)

    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=True, edge_opt=True,
                                n_iter=5, periodic_margin=margin)
    thin = [p['size'] for p in _pieces(pmesh, domain) if p['size'] < margin]
    assert thin == []
    assert _min_edge(pmesh) >= min_edge_0 - 1e-9
    assert np.isclose(sum(pmesh.volumes), domain.volume)
    pmesh_re = PolyMesh.from_seeds(seeds, domain, periodic=True)
    assert np.allclose(np.sort(pmesh_re.volumes), np.sort(pmesh.volumes),
                       rtol=0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Wedges (corners at the periodic faces narrower than the mesh angle)         #
# --------------------------------------------------------------------------- #
def _wedge_seeds_2d(angle_deg=15.0):
    """Circles in a square of side 3, periodic in x. The seeds A and B are
    0.5 apart along a line tilted by ``angle_deg`` from the x axis, so
    their facet (normal to that line) meets the face x = 3 at that angle,
    at about (3, 0.1): the cell of B has a wedge there. The other seeds
    form a jittered grid away from them."""
    rng = np.random.RandomState(0)
    ang = np.radians(angle_deg)
    positions = [[2.5, 1.0],
                 [2.5 + 0.5 * np.cos(ang), 1.0 + 0.5 * np.sin(ang)]]
    for x in (0.75, 1.75):
        for y in (0.75, 1.75, 2.75):
            if (x, y) == (0.75, 0.75):
                continue  # its image would cut the corner of the wedge
            positions.append([x + 0.03 * (2 * rng.rand() - 1),
                              y + 0.03 * (2 * rng.rand() - 1)])
    positions.append([2.75, 2.75])
    return SeedList([Seed.factory('circle', r=0.2, position=p)
                     for p in positions])


def _wedge_seeds_3d(angle_deg=15.0):
    """The 2D configuration extruded along z in a cube of side 3, periodic
    in x: the facet of A and B contains the z direction and meets the
    face x = 3 along a line, with a dihedral angle of ``angle_deg``."""
    rng = np.random.RandomState(0)
    ang = np.radians(angle_deg)
    positions = [[2.5, 1.0, 1.5],
                 [2.5 + 0.5 * np.cos(ang), 1.0 + 0.5 * np.sin(ang), 1.5]]
    for x in (0.75, 1.75):
        for y in (0.75, 1.75, 2.75):
            for z in (0.75, 2.25):
                positions.append([x + 0.03 * (2 * rng.rand() - 1),
                                  y + 0.03 * (2 * rng.rand() - 1),
                                  z + 0.03 * (2 * rng.rand() - 1)])
    positions.append([2.75, 2.75, 0.75])
    positions.append([2.75, 2.75, 2.25])
    return SeedList([Seed.factory('sphere', r=0.2, position=p)
                     for p in positions])


def _wedges(pmesh, domain, per_axes, min_angle):
    scale = max([ub - lb for lb, ub in domain.limits])
    feats = _mesh_features(pmesh, per_axes, domain.limits, scale, min_angle)
    return [f for f in feats if f['kind'] == 'wedge']


def test_wedge_geometry():
    # 2D: a vertex on the wall, the wall edge along -x and a facet at 20
    # degrees from it, 0.5 long
    ang = np.radians(20)
    pts = np.array([[0.0, 0.0], [-1.0, 0.0],
                    [-0.5 * np.cos(ang), 0.5 * np.sin(ang)]])
    angle, length, u_vec, where = _wedge_geometry(pts, [0, 1], [0, 2], [0],
                                                  np.array([-0.5, 0.05]))
    assert np.isclose(angle, ang)
    assert np.isclose(length, 0.5)
    assert np.allclose(u_vec, [-np.cos(ang), np.sin(ang)])
    assert np.allclose(where, [0, 0])

    # 3D: the wall facet in the plane y = 0, the cell above it, and a
    # facet leaving their common edge (along x) at 20 degrees
    d = np.array([0.0, np.sin(ang), np.cos(ang)])
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1],
                    [1, 0, 0] + 0.5 * d, [0, 0, 0] + 0.5 * d], dtype=float)
    cen = np.array([0.5, 0.1, 0.6])
    angle, length, u_vec, where = _wedge_geometry(pts, [0, 1, 2, 3],
                                                  [0, 1, 4, 5], [0, 1], cen)
    assert np.isclose(angle, ang)
    assert np.isclose(length, 0.5)
    assert np.allclose(u_vec, d)
    assert np.allclose(where, [0.5, 0, 0])
    # the same with the facets listed in the other order round the edge
    angle_2, _, _, _ = _wedge_geometry(pts, [3, 2, 1, 0], [5, 4, 1, 0],
                                       [1, 0], cen)
    assert np.isclose(angle_2, ang)


def test_wedge_features_2d():
    domain = geometry.Square(side_length=3, corner=(0, 0))
    seeds = _wedge_seeds_2d(15.0)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='x')
    # no wedges narrower than 10 degrees; two narrower than 25: the corner
    # of the cell of B (seed 1) on the face x = 3, and its image on the
    # face x = 0, where the facet continues into the piece of A (seed 0)
    assert _wedges(pmesh, domain, [True, False], 10.0) == []
    wedges = _wedges(pmesh, domain, [True, False], 25.0)
    assert len(wedges) == 2
    by_seed = {w['seed']: w for w in wedges}
    assert set(by_seed) == {0, 1}
    w = by_seed[1]
    assert w['neighbor'] == 0
    assert (w['axis'], w['side']) == (0, 1)
    assert np.isclose(np.degrees(w['angle']), 15.0, atol=1.0)
    assert 0 < w['size'] < 0.25 * 3 * np.sin(w['angle'])
    # the direction is along the facet, away from the face
    assert w['u_vec'][0] < 0
    assert np.isclose(np.linalg.norm(w['u_vec']), 1)
    w_0 = by_seed[0]
    assert w_0['neighbor'] == 1
    assert (w_0['axis'], w_0['side']) == (0, 0)
    assert np.isclose(np.degrees(w_0['angle']), 15.0, atol=1.0)
    assert w_0['u_vec'][0] > 0
    # no wedges at all without a minimum angle
    assert _wedges(pmesh, domain, [True, False], 0.0) == []
    # no wedges on non-periodic faces (y is not periodic)
    assert all([f['axis'] == 0 for f in wedges])


def test_wedge_features_3d():
    domain = geometry.Cube(side_length=3, corner=(0, 0, 0))
    seeds = _wedge_seeds_3d(15.0)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='x')
    wedges = _wedges(pmesh, domain, [True, False, False], 25.0)
    mine = [w for w in wedges if w['seed'] == 1]
    assert len(mine) >= 1
    for w in mine:
        assert w['neighbor'] == 0
        assert (w['axis'], w['side']) == (0, 1)
        assert np.isclose(np.degrees(w['angle']), 15.0, atol=1.0)
        assert w['u_vec'][0] < 0
        assert np.isclose(w['u_vec'][2], 0, atol=0.05)
    assert _wedges(pmesh, domain, [True, False, False], 10.0) == []


def test_edge_opt_opens_wedge_2d():
    np.random.seed(0)
    domain = geometry.Square(side_length=3, corner=(0, 0))
    seeds = _wedge_seeds_2d(15.0)
    pmesh_0 = PolyMesh.from_seeds(seeds, domain, periodic='x')
    assert len(_wedges(pmesh_0, domain, [True, False], 25.0)) == 2
    min_edge_0 = _min_edge(pmesh_0)

    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='x', edge_opt=True,
                                n_iter=5, periodic_margin=0.05,
                                min_angle=25.0)
    assert _wedges(pmesh, domain, [True, False], 25.0) == []
    assert _min_edge(pmesh) >= min_edge_0 - 1e-9
    assert np.isclose(sum(pmesh.volumes), domain.area)
    pmesh_re = PolyMesh.from_seeds(seeds, domain, periodic='x')
    assert np.allclose(np.sort(pmesh_re.volumes), np.sort(pmesh.volumes),
                       rtol=0, atol=1e-9)


def test_edge_opt_opens_wedge_3d():
    np.random.seed(0)
    domain = geometry.Cube(side_length=3, corner=(0, 0, 0))
    seeds = _wedge_seeds_3d(15.0)
    pmesh_0 = PolyMesh.from_seeds(seeds, domain, periodic='x')
    n_0 = len(_wedges(pmesh_0, domain, [True, False, False], 25.0))
    assert n_0 >= 1
    min_edge_0 = _min_edge(pmesh_0)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='x', edge_opt=True,
                                n_iter=5, periodic_margin=0.05,
                                min_angle=25.0)
    assert len(_wedges(pmesh, domain, [True, False, False], 25.0)) < n_0
    assert _min_edge(pmesh) >= min_edge_0 - 1e-9
    assert np.isclose(sum(pmesh.volumes), domain.volume)
