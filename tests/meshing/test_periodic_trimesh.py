"""Tests for periodic triangular, tetrahedral and raster meshes."""
import numpy as np
import pytest
import scipy.stats
from periodic_helpers import check_periodic_pairs
from periodic_helpers import wedge_seeds_2d

import microstructpy as msp
from microstructpy.meshing import PolyMesh
from microstructpy.meshing import RasterMesh
from microstructpy.meshing import TriMesh
from microstructpy.meshing import trimesh as trimesh_module
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
    """Every node on a periodic face is paired with its exact image, and
    the facets on the faces (edges in 2D, triangles in 3D) are paired."""
    assert mesh.periodic_axes == list(per_axes)
    faces = check_periodic_pairs(mesh.points, mesh.facets,
                                 mesh.periodic_nodes, mesh.periodic_facets,
                                 per_axes, domain)
    for axis, (low, high, n_low) in faces.items():
        if domain.n_dim == 2:
            # the edges on a face connect its nodes in a chain
            assert n_low == len(low) - 1
        else:
            assert n_low > 0


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


def _wedge_polymesh(angle_deg):
    """A square of side 3, periodic in x, with a facet meeting the face
    x = 3 at ``angle_deg`` (see the tests of the edge optimization)."""
    domain = msp.geometry.Square(side_length=3, corner=(0, 0))
    return PolyMesh.from_seeds(wedge_seeds_2d(angle_deg), domain,
                               periodic='x')


def _min_edge(mesh):
    pts = np.array(mesh.points)
    elems = np.array(mesh.elements)
    lengths = [np.linalg.norm(pts[elems[:, i]] - pts[elems[:, (i + 1) % 3]],
                              axis=1) for i in range(3)]
    return np.min(lengths)


def test_periodic_trimesh_no_cascade_at_wedges():
    # a corner narrower than the minimum angle makes Triangle refine it in
    # shells of small elements; the passes that match the periodic faces
    # must not deepen the shells (they did, one level per pass)
    phases = [{'shape': 'circle', 'size': 0.4}]
    for angle_deg, min_angle in ((15.0, 20), (15.0, 25)):
        pmesh = _wedge_polymesh(angle_deg)
        mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=min_angle,
                                     max_volume=0.05)
        pmesh.periodic_axes = [False, False]
        pmesh.periodic_points = {}
        pmesh.periodic_facets = {}
        plain = TriMesh.from_polymesh(pmesh, phases, min_angle=min_angle,
                                      max_volume=0.05)
        # at most one level of shells beyond the non-periodic mesh (the
        # shells of both faces are put together), with some slack
        assert _min_edge(mesh) >= 0.3 * _min_edge(plain)


def test_ghost_layer_copies_are_closed(periodic_case):
    # the copies of the cells outside the periodic faces are closed
    # polygons, including those of the cells at the corners of the domain,
    # which are moved along both axes: every point of a copy is shared by
    # at least two facets, so that Triangle does not eat into the copies
    domain, phases, seeds, pmesh = periodic_case
    pts = [list(p) for p in pmesh.points]
    kps = {i: i for i in range(len(pts))}
    facets = [list(f) for f in pmesh.facets]
    facet_nums = [f + 1 for f in range(len(facets))]
    labels = np.arange(len(pmesh.regions))
    out = trimesh_module._ghost_layer(pmesh, phases, labels, kps, pts,
                                      facets, facet_nums, [], [], np.inf)
    g_pts, g_facets, g_nums = out[0], out[1], out[2]
    assert len(g_pts) > len(pts)
    assert any([n == 0 for n in g_nums])
    degree = np.zeros(len(g_pts), dtype=int)
    for facet in g_facets:
        for kp in facet:
            degree[kp] += 1
    assert np.all(degree[len(pts):] >= 2)
    # the copies cover the corners of the domain: points outside along
    # both axes exist
    arr = np.array(g_pts)
    lims = np.array(domain.limits)
    outside = (arr < lims[:, 0] - 1e-9) | (arr > lims[:, 1] + 1e-9)
    assert np.any(np.all(outside, axis=1))


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


# --------------------------------------------------------------------------- #
# 3D                                                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def periodic_case_3d():
    domain = msp.geometry.Cube(side_length=1.5, corner=(0, 0, 0))
    phases = [{'shape': 'sphere', 'size': scipy.stats.uniform(0.35, 0.2),
               'material_type': 'crystalline'},
              {'shape': 'sphere', 'size': 0.4, 'material_type': 'amorphous'}]
    seeds = SeedList.from_info(phases, 0.5 * domain.volume)
    seeds.position(domain, rtol=0.0, rng_seed=1, periodic=True)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=True)
    return domain, phases, seeds, pmesh


def _element_volumes(mesh):
    pts = np.array(mesh.points)
    elems = np.array(mesh.elements)
    rel = pts[elems[:, 1:]] - pts[elems[:, :1]]
    return np.linalg.det(rel) / 6.0


def test_periodic_tetmesh_nodes_match(periodic_case_3d):
    domain, phases, seeds, pmesh = periodic_case_3d
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=10)
    _check_periodic_mesh(mesh, domain, [True, True, True])
    vols = _element_volumes(mesh)
    assert np.all(np.abs(vols) > 0)
    assert np.isclose(np.abs(vols).sum(), domain.volume)
    assert set(mesh.element_attributes) <= set(pmesh.seed_numbers)


def test_periodic_tetmesh_single_axis():
    domain = msp.geometry.Box(limits=[(0, 1.5), (0, 1), (0, 1)])
    phases = [{'shape': 'sphere', 'size': scipy.stats.uniform(0.3, 0.2)}]
    seeds = SeedList.from_info(phases, 0.5 * domain.volume)
    seeds.position(domain, rtol=0.0, rng_seed=3, periodic='z')
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic='z')
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=10)
    _check_periodic_mesh(mesh, domain, [False, False, True])
    assert np.isclose(np.abs(_element_volumes(mesh)).sum(), domain.volume)


def test_periodic_tetmesh_file_and_abaqus(periodic_case_3d, tmp_path):
    domain, phases, seeds, pmesh = periodic_case_3d
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=10)
    fname = str(tmp_path / 'trimesh.txt')
    mesh.write(fname)
    loaded = TriMesh.from_file(fname)
    assert np.array_equal(np.array(loaded.points), np.array(mesh.points))
    assert {k: [tuple(p) for p in v] for k, v in
            loaded.periodic_nodes.items()} == mesh.periodic_nodes
    mesh.write(str(tmp_path / 'mesh.inp'), 'abaqus', seeds, pmesh)
    with open(str(tmp_path / 'mesh.inp'), 'r') as file:
        text = file.read()
    for name in 'XYZ':
        assert '*Nset, nset=Set-N-Periodic-' + name + '-Low' in text
        assert '*Nset, nset=Set-N-Periodic-' + name + '-High' in text


def test_periodic_raster_mesh_3d(periodic_case_3d):
    domain, phases, seeds, pmesh = periodic_case_3d
    mesh = RasterMesh.from_polymesh(pmesh, 0.15, phases)
    assert mesh.periodic_axes == [True, True, True]
    pts = np.array(mesh.points)
    for axis in range(3):
        pairs = mesh.periodic_nodes[axis]
        assert len(pairs) == 11 * 11
        shift = np.zeros(3)
        shift[axis] = 1.5
        for lo, hi in pairs:
            assert np.array_equal(pts[hi], pts[lo] + shift)
