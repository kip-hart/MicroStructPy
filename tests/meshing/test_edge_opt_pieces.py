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
