"""Helpers shared by the tests of periodic microstructures."""
import copy
import itertools

import numpy as np

import microstructpy as msp
from microstructpy.meshing import PolyMesh
from microstructpy.meshing.polymesh import _edge_lengths
from microstructpy.seeding import Seed
from microstructpy.seeding import SeedList


def min_edge(pmesh):
    """Length of the shortest edge of a polygonal/polyhedral mesh."""
    return min([e['length'] for e in _edge_lengths(pmesh).values()])


def wedge_seeds_2d(angle_deg):
    """Circles in a square of side 3 to be tessellated periodic in x.

    The seeds A and B are 0.5 apart along a line tilted by ``angle_deg``
    from the x axis, so their facet (normal to that line) meets the face
    x = 3 at that angle, at about (3, 0.1): the cell of B (seed 1) has a
    wedge there, and the cell of A (seed 0) its image on the face x = 0.
    The other seeds form a jittered grid away from them.
    """
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


def check_periodic_pairs(points, facets, point_pairs, facet_pairs, per_axes,
                         domain):
    """Check the pairs of points and facets on the periodic faces.

    Every point on a periodic face is paired, exactly once, with a point
    on the opposite face that is its exact translate by the domain
    length, and every facet whose points all lie on a face is paired with
    the facet made of their images.

    Args:
        points (list or numpy.ndarray): The points of the mesh.
        facets (list): The facets of the mesh.
        point_pairs (dict): Axis -> list of (lower, upper) point numbers.
        facet_pairs (dict): Axis -> list of (lower, upper) facet numbers.
        per_axes (list): The periodicity flags.
        domain (from :mod:`microstructpy.geometry`): The domain.

    Returns:
        dict: Axis -> (set of the points on the lower face, set of the
        points on the upper face, number of facets on the lower face).

    """
    pts = np.array(points)
    lims = np.array(domain.limits)
    n_dim = len(lims)
    faces = {}
    for axis, flag in enumerate(per_axes):
        if not flag:
            assert axis not in point_pairs
            continue
        lb, ub = lims[axis]
        shift = np.zeros(n_dim)
        shift[axis] = ub - lb
        pairs = point_pairs[axis]
        low = set(np.nonzero(np.isclose(pts[:, axis], lb))[0])
        high = set(np.nonzero(np.isclose(pts[:, axis], ub))[0])
        assert len(pairs) == len(low) == len(high) > 0
        assert set([lo for lo, _ in pairs]) == low
        assert set([hi for _, hi in pairs]) == high
        for lo, hi in pairs:
            assert np.array_equal(pts[hi], pts[lo] + shift)
        kp_map = dict(pairs)
        f_pairs = dict(facet_pairs[axis])
        n_low = 0
        for f_num, facet in enumerate(facets):
            if all([kp in low for kp in facet]):
                n_low += 1
                assert f_num in f_pairs
                image = facets[f_pairs[f_num]]
                assert set(image) == set([kp_map[kp] for kp in facet])
        faces[axis] = (low, high, n_low)
    return faces


def seed_volumes(pmesh, n_seeds):
    """Volume (area) of the cells of each seed, pieces added up."""
    vols = np.zeros(n_seeds)
    for seed_num, vol in zip(pmesh.seed_numbers, pmesh.volumes):
        vols[seed_num] += vol
    return vols


def tiled_reference_volumes(seeds, domain, per_axes):
    """Volumes of the cells of the seeds in a periodic tessellation,
    computed as a non-periodic tessellation of the seeds tiled across the
    periodic axes (3 copies per periodic axis)."""
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
    if len(lims) == 2:
        big_domain = msp.geometry.Rectangle(limits=big_lims)
    else:
        big_domain = msp.geometry.Box(limits=big_lims)
    pmesh = PolyMesh.from_seeds(tiled, big_domain)
    # the cells of the original copies (the zero translation)
    i_zero = [i for i, t in enumerate(itertools.product(*options))
              if not any(t)][0]
    vols = np.zeros(n_seeds)
    for seed_num, vol in zip(pmesh.seed_numbers, pmesh.volumes):
        block, local = divmod(seed_num, n_seeds)
        if block == i_zero:
            vols[local] += vol
    return vols
