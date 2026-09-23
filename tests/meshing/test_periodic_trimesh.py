"""Tests for periodic triangular and raster meshes (2D)."""
import numpy as np
import pytest
import scipy.stats

import microstructpy as msp
from microstructpy.meshing import PolyMesh
from microstructpy.meshing import RasterMesh
from microstructpy.meshing import TriMesh
from microstructpy.seeding import SeedList


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _phases():
    return [{'shape': 'circle', 'size': scipy.stats.uniform(0.15, 0.15),
             'material_type': 'crystalline'},
            {'shape': 'ellipse', 'size': scipy.stats.uniform(0.2, 0.15),
             'aspect_ratio': scipy.stats.uniform(1.5, 1.5),
             'angle_deg': scipy.stats.uniform(0, 180),
             'material_type': 'amorphous'}]


@pytest.fixture(scope='module')
def periodic_case():
    domain = msp.geometry.Square(side_length=2, corner=(0, 0))
    phases = _phases()
    seeds = SeedList.from_info(phases, 0.55 * domain.area)
    seeds.position(domain, rtol=0.0, rng_seed=1, periodic=True)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=True)
    return domain, phases, seeds, pmesh


def _element_areas(mesh):
    pts = np.array(mesh.points)
    elems = np.array(mesh.elements)
    p0, p1, p2 = pts[elems[:, 0]], pts[elems[:, 1]], pts[elems[:, 2]]
    return 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) -
                  (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))


def _check_periodic_mesh(mesh, domain, per_axes):
    pts = np.array(mesh.points)
    lims = np.array(domain.limits)
    assert mesh.periodic_axes == list(per_axes)
    for axis, flag in enumerate(per_axes):
        if not flag:
            assert axis not in mesh.periodic_nodes
            continue
        lb, ub = lims[axis]
        shift = np.zeros(2)
        shift[axis] = ub - lb
        pairs = mesh.periodic_nodes[axis]
        low = set(np.nonzero(np.isclose(pts[:, axis], lb))[0])
        high = set(np.nonzero(np.isclose(pts[:, axis], ub))[0])
        # every node on a periodic face is paired, exactly
        assert len(pairs) == len(low) == len(high) > 0
        assert set([lo for lo, _ in pairs]) == low
        assert set([hi for _, hi in pairs]) == high
        for lo, hi in pairs:
            assert np.array_equal(pts[hi], pts[lo] + shift)
        # facets (edges) on the faces are paired
        kp_map = dict(pairs)
        f_pairs = dict(mesh.periodic_facets[axis])
        n_low = 0
        for f_num, facet in enumerate(mesh.facets):
            if all([kp in low for kp in facet]):
                n_low += 1
                assert f_num in f_pairs
                image = mesh.facets[f_pairs[f_num]]
                assert set(image) == set([kp_map[kp] for kp in facet])
        # the edges on a face connect its nodes in a chain
        assert n_low == len(pairs) - 1


# --------------------------------------------------------------------------- #
# Triangular meshes                                                           #
# --------------------------------------------------------------------------- #
def test_periodic_trimesh_nodes_match(periodic_case):
    domain, phases, seeds, pmesh = periodic_case
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=20)
    _check_periodic_mesh(mesh, domain, [True, True])
    areas = _element_areas(mesh)
    assert np.all(areas > 0)
    assert np.isclose(areas.sum(), domain.area)
    # element attributes are seed numbers of the mesh
    assert set(mesh.element_attributes) <= set(pmesh.seed_numbers)


def test_periodic_trimesh_edge_subdivision(periodic_case):
    domain, phases, seeds, pmesh = periodic_case
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=20,
                                 max_edge_length=0.08)
    _check_periodic_mesh(mesh, domain, [True, True])
    pts = np.array(mesh.points)
    facets = np.array(mesh.facets)
    lengths = np.linalg.norm(pts[facets[:, 0]] - pts[facets[:, 1]], axis=1)
    assert lengths.max() <= 0.08 + 1e-9
    # the boundary was subdivided (more node pairs than polymesh points
    # on the face)
    poly_pts = np.array(pmesh.points)
    n_poly = np.sum(np.isclose(poly_pts[:, 0], 0))
    assert len(mesh.periodic_nodes[0]) > n_poly


def test_periodic_trimesh_single_axis():
    domain = msp.geometry.Rectangle(length=3, width=2, corner=(0, 0))
    phases = _phases()
    seeds = SeedList.from_info(phases, 0.55 * domain.area)
    seeds.position(domain, rtol=0.0, rng_seed=2, periodic='x')
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='x')
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=20)
    _check_periodic_mesh(mesh, domain, [True, False])
    assert np.isclose(_element_areas(mesh).sum(), domain.area)


def test_non_periodic_trimesh_unchanged(periodic_case):
    domain, phases, seeds, pmesh = periodic_case
    pmesh_np = PolyMesh.from_seeds(seeds, domain)
    mesh = TriMesh.from_polymesh(pmesh_np, phases, min_angle=20)
    assert mesh.periodic_axes is None
    assert mesh.periodic_nodes is None
    assert 'Periodic' not in str(mesh)


def test_periodic_trimesh_file_round_trip(periodic_case, tmp_path):
    domain, phases, seeds, pmesh = periodic_case
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=20)
    fname = str(tmp_path / 'trimesh.txt')
    mesh.write(fname)
    loaded = TriMesh.from_file(fname)
    assert np.array_equal(np.array(loaded.points), np.array(mesh.points))
    assert np.array_equal(np.array(loaded.elements),
                          np.array(mesh.elements))
    assert loaded.periodic_axes == mesh.periodic_axes
    assert {k: [tuple(p) for p in v] for k, v in
            loaded.periodic_nodes.items()} == mesh.periodic_nodes
    assert {k: [tuple(p) for p in v] for k, v in
            loaded.periodic_facets.items()} == mesh.periodic_facets


def test_periodic_trimesh_abaqus_node_sets(periodic_case, tmp_path):
    domain, phases, seeds, pmesh = periodic_case
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=20)
    fname = str(tmp_path / 'mesh.inp')
    mesh.write(fname, 'abaqus', seeds, pmesh)
    with open(fname, 'r') as file:
        lines = file.read().splitlines()

    def nset(name):
        i = lines.index('*Nset, nset=' + name + ', unsorted') + 1
        nodes = []
        while i < len(lines) and not lines[i].startswith('*'):
            nodes.extend([int(n) for n in lines[i].split(',')])
            i += 1
        return nodes

    pts = np.array(mesh.points)
    for axis, name in enumerate('XY'):
        low = nset('Set-N-Periodic-' + name + '-Low')
        high = nset('Set-N-Periodic-' + name + '-High')
        pairs = mesh.periodic_nodes[axis]
        assert low == [lo + 1 for lo, _ in pairs]
        assert high == [hi + 1 for _, hi in pairs]
        shift = np.zeros(2)
        shift[axis] = 2
        for lo, hi in zip(low, high):
            assert np.array_equal(pts[hi - 1], pts[lo - 1] + shift)


def test_periodic_gmsh_not_supported(periodic_case):
    domain, phases, seeds, pmesh = periodic_case
    with pytest.raises(NotImplementedError):
        TriMesh.from_polymesh(pmesh, phases, mesher='gmsh', mesh_size=0.1)


# --------------------------------------------------------------------------- #
# Raster meshes                                                               #
# --------------------------------------------------------------------------- #
def test_periodic_raster_mesh(periodic_case):
    domain, phases, seeds, pmesh = periodic_case
    mesh = RasterMesh.from_polymesh(pmesh, 0.05, phases)
    pts = np.array(mesh.points)
    assert mesh.periodic_axes == [True, True]
    for axis in (0, 1):
        pairs = mesh.periodic_nodes[axis]
        assert len(pairs) == 41
        shift = np.zeros(2)
        shift[axis] = 2
        for lo, hi in pairs:
            assert np.array_equal(pts[hi], pts[lo] + shift)
    with pytest.raises(ValueError):
        RasterMesh.from_polymesh(pmesh, 0.07, phases)
