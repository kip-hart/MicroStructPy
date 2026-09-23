"""Geometric checks of periodic meshes.

The cells of a periodic polygonal/polyhedral mesh must partition the
domain: each cell is a closed convex polytope inside the domain, the cell
volumes add up to the domain volume and every point of the domain lies in
exactly one cell (no gaps, no overlaps). The elements of the triangular/
tetrahedral mesh must partition the cells in the same way.
"""
from collections import Counter

import numpy as np
import pytest
import scipy.stats

import microstructpy as msp
from microstructpy.meshing import PolyMesh
from microstructpy.meshing import TriMesh
from microstructpy.meshing.trimesh import _amorphous_seed_numbers
from microstructpy.seeding import SeedList

PHASES = {
    2: [{'shape': 'circle', 'size': scipy.stats.uniform(0.15, 0.15),
         'material_type': 'crystalline'},
        {'shape': 'ellipse', 'size': scipy.stats.uniform(0.2, 0.15),
         'aspect_ratio': scipy.stats.uniform(1.5, 1.5),
         'angle_deg': scipy.stats.uniform(0, 180),
         'material_type': 'amorphous'}],
    3: [{'shape': 'sphere', 'size': scipy.stats.uniform(0.3, 0.2),
         'material_type': 'crystalline'},
        {'shape': 'sphere', 'size': 0.4, 'material_type': 'amorphous'}],
}

CASES = [
    ('square-xy', msp.geometry.Square(side_length=2, corner=(0, 0)), 1, True),
    ('rect-x', msp.geometry.Rectangle(length=3, width=2, corner=(0, 0)), 2,
     'x'),
    ('cube-xyz-1', msp.geometry.Cube(side_length=1.5, corner=(0, 0, 0)), 1,
     True),
    ('cube-xyz-2', msp.geometry.Cube(side_length=1.5, corner=(0, 0, 0)), 2,
     True),
    ('cube-xz', msp.geometry.Cube(side_length=1.5, corner=(0, 0, 0)), 3,
     'xz'),
    ('box-y', msp.geometry.Box(limits=[(0, 2), (0, 1), (0, 1.3)]), 7, 'y'),
    ('box-none', msp.geometry.Box(limits=[(0, 2), (0, 1), (0, 1.3)]), 9,
     False),
]


@pytest.fixture(scope='module', params=CASES, ids=[c[0] for c in CASES])
def case(request):
    name, domain, rng_seed, periodic = request.param
    n_dim = len(domain.limits)
    phases = PHASES[n_dim]
    seeds = SeedList.from_info(phases, 0.5 * domain.n_vol)
    seeds.position(domain, rtol=0.0, rng_seed=rng_seed, periodic=periodic)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=periodic)
    return domain, phases, seeds, pmesh


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _facet_normal(loop):
    """Outward-agnostic normal of a facet, scaled by its measure."""
    if loop.shape[1] == 2:
        t = loop[1] - loop[0]
        return np.array([t[1], -t[0]])
    n = np.zeros(3)
    for i in range(len(loop)):
        n += np.cross(loop[i], loop[(i + 1) % len(loop)])
    return 0.5 * n


def _cell_planes(pts, pmesh, reg):
    """Outward unit normals, offsets and measures of the facets of a cell."""
    verts = np.unique(np.concatenate([pmesh.facets[f]
                                      for f in pmesh.regions[reg]]))
    center = pts[verts].mean(axis=0)
    planes = []
    for f in pmesh.regions[reg]:
        loop = pts[pmesh.facets[f]]
        n = _facet_normal(loop)
        measure = np.linalg.norm(n)
        assert measure > 0
        n = n / measure
        p0 = loop.mean(axis=0)
        if np.dot(n, p0 - center) < 0:
            n = -n
        planes.append((n, np.dot(n, p0), measure, p0))
    return planes, verts


def _check_closed(pmesh, reg):
    """Each vertex (2D) or edge (3D) is shared by exactly two facets."""
    counts = Counter()
    for f in pmesh.regions[reg]:
        facet = pmesh.facets[f]
        if len(facet) == 2:
            counts.update(facet)
        else:
            for i in range(len(facet)):
                counts[tuple(sorted((facet[i],
                                     facet[(i + 1) % len(facet)])))] += 1
    assert all([c == 2 for c in counts.values()])
    if len(pmesh.facets[0]) > 2:
        verts = set()
        for f in pmesh.regions[reg]:
            verts.update(pmesh.facets[f])
        assert len(verts) - len(counts) + len(pmesh.regions[reg]) == 2


def _inside(planes, points, tol=1e-9):
    normals = np.array([p[0] for p in planes])
    offsets = np.array([p[1] for p in planes])
    return np.all(points @ normals.T - offsets <= tol, axis=1)


# --------------------------------------------------------------------------- #
# Polygonal / polyhedral meshes                                               #
# --------------------------------------------------------------------------- #
def test_cells_partition_the_domain(case):
    domain, phases, seeds, pmesh = case
    pts = np.array(pmesh.points)
    lims = np.array(domain.limits)
    n_dim = len(lims)
    scale = np.max(lims[:, 1] - lims[:, 0])
    assert np.all(pts >= lims[:, 0] - 1e-9)
    assert np.all(pts <= lims[:, 1] + 1e-9)
    # no duplicate points
    assert len(np.unique(np.round(pts / scale, 9), axis=0)) == len(pts)

    # facets are planar with a non-zero measure and distinct points
    for facet in pmesh.facets:
        assert len(set(facet)) == len(facet) >= n_dim
        loop = pts[facet]
        n = _facet_normal(loop)
        assert np.linalg.norm(n) > 1e-12
        n /= np.linalg.norm(n)
        assert np.abs((loop - loop[0]) @ n).max() < 1e-8 * scale

    # facet neighbors and regions agree; wall facets are on their wall
    for f, neighs in enumerate(pmesh.facet_neighbors):
        for r in neighs:
            if r >= 0:
                assert f in pmesh.regions[r]
            else:
                axis, side = divmod(-r - 1, 2)
                assert np.allclose(pts[pmesh.facets[f], axis],
                                   lims[axis][side], atol=1e-9)
    for r, region in enumerate(pmesh.regions):
        for f in region:
            assert r in pmesh.facet_neighbors[f]

    # every cell is closed and convex; volumes add up to the domain volume
    vols = np.zeros(len(pmesh.regions))
    all_planes = []
    for r in range(len(pmesh.regions)):
        _check_closed(pmesh, r)
        planes, verts = _cell_planes(pts, pmesh, r)
        all_planes.append(planes)
        vols[r] = sum([np.dot(n, p0) * m for n, _, m, p0 in planes]) / n_dim
        assert vols[r] > 0
        assert np.all(_inside(planes, pts[verts], tol=1e-8 * scale))
    assert np.isclose(vols.sum(), domain.n_vol, rtol=1e-9)
    assert np.allclose(pmesh.volumes, vols, rtol=1e-9, atol=1e-12)

    # random points of the domain lie in exactly one cell
    rng = np.random.default_rng(0)
    lengths = lims[:, 1] - lims[:, 0]
    sample = lims[:, 0] + rng.random((20000, n_dim)) * lengths
    counts = np.zeros(len(sample), dtype=int)
    for planes in all_planes:
        counts += _inside(planes, sample, tol=1e-12)
    assert np.all(counts == 1)


# --------------------------------------------------------------------------- #
# Triangular / tetrahedral meshes                                             #
# --------------------------------------------------------------------------- #
def test_elements_partition_the_cells(case):
    domain, phases, seeds, pmesh = case
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=10)
    pts = np.array(mesh.points)
    elems = np.array(mesh.elements)
    lims = np.array(domain.limits)
    n_dim = len(lims)

    # positively oriented elements that add up to the domain volume
    rel = pts[elems[:, 1:]] - pts[elems[:, :1]]
    svol = np.linalg.det(rel) / (2.0 if n_dim == 2 else 6.0)
    assert np.all(svol > 0)
    assert np.isclose(svol.sum(), domain.n_vol, rtol=1e-9)

    # every element face is shared by two elements or lies on the boundary
    faces = Counter()
    for e in elems:
        for i in range(n_dim + 1):
            faces[tuple(sorted(np.delete(e, i)))] += 1
    assert max(faces.values()) == 2
    for face, c in faces.items():
        if c == 1:
            fp = pts[list(face)]
            assert any([np.allclose(fp[:, ax], lims[ax][k], atol=1e-9)
                        for ax in range(n_dim) for k in range(2)])

    # element attributes: cells of the same amorphous phase are merged and
    # labelled with one seed number; the element volumes of each attribute
    # add up to the volume of its cells and every element lies in one of them
    attrs = np.array(mesh.element_attributes)
    conv = _amorphous_seed_numbers(pmesh, phases)
    att_of_reg = np.array([conv.get(s, s) for s in pmesh.seed_numbers])
    assert set(attrs.tolist()) == set(att_of_reg.tolist())
    ppts = np.array(pmesh.points)
    cell_vols = np.array(pmesh.volumes)
    cents = pts[elems].mean(axis=1)
    for att in np.unique(att_of_reg):
        regs = np.nonzero(att_of_reg == att)[0]
        mask = attrs == att
        assert np.isclose(svol[mask].sum(), cell_vols[regs].sum(), rtol=1e-9)
        inside = np.zeros(np.sum(mask), dtype=bool)
        for r in regs:
            planes, _ = _cell_planes(ppts, pmesh, r)
            inside |= _inside(planes, cents[mask])
        assert np.all(inside)

    # the labels: crystalline cells keep their seed number, amorphous cells
    # that share a facet (also across a periodic face) share a label, which
    # is the seed number of one of them
    seed_nums = np.array(pmesh.seed_numbers)
    phase_nums = np.array(pmesh.phase_numbers)
    amorph = np.array([phases[p]['material_type'] == 'amorphous'
                       for p in phase_nums])
    assert np.array_equal(att_of_reg[~amorph], seed_nums[~amorph])
    assert set(att_of_reg[amorph]) <= set(seed_nums[amorph])
    pairs = [tuple(n) for n in pmesh.facet_neighbors]
    for axis_pairs in (pmesh.periodic_facets or {}).values():
        pairs += [(max(pmesh.facet_neighbors[lo]),
                   max(pmesh.facet_neighbors[hi])) for lo, hi in axis_pairs]
    n_merged = 0
    for r_a, r_b in pairs:
        if min(r_a, r_b) >= 0 and amorph[r_a] and amorph[r_b]:
            assert att_of_reg[r_a] == att_of_reg[r_b]
            n_merged += 1
    assert n_merged > 0
