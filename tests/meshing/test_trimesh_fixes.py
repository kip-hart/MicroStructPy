"""Tests for the bug fixes in microstructpy.meshing.trimesh.

Each test corresponds to a defect that was fixed in the TriMesh and
RasterMesh classes: mesh size controls that were ignored, raster meshes
with inverted elements and wrong facets, corrupt or incomplete output
files, and robustness of the constructors and writers.
"""
from __future__ import division

import copy
import re

import meshpy.triangle
import numpy as np
import pytest
import scipy.stats
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

import microstructpy as msp
from microstructpy.meshing import trimesh as trimesh_module
from microstructpy.meshing.trimesh import RasterMesh
from microstructpy.meshing.trimesh import TriMesh

# Abaqus element face definitions (local node numbers, 0-based)
# CPS4: S1 = 1-2, S2 = 2-3, S3 = 3-4, S4 = 4-1
# C3D8: S1 = 1-2-3-4, S2 = 5-8-7-6, S3 = 1-5-6-2, S4 = 2-6-7-3,
#       S5 = 3-7-8-4, S6 = 4-8-5-1
ABAQUS_FACES = {
    4: {1: (0, 1), 2: (1, 2), 3: (2, 3), 4: (3, 0)},
    8: {1: (0, 1, 2, 3), 2: (4, 7, 6, 5), 3: (0, 4, 5, 1),
        4: (1, 5, 6, 2), 5: (2, 6, 7, 3), 6: (3, 7, 4, 0)},
}

MESH_SIZE = 0.1


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def case_2d():
    """2D microstructure with crystalline, amorphous, and void phases."""
    np.random.seed(1)
    phases = [{'shape': 'circle', 'size': scipy.stats.uniform(0.15, 0.1),
               'material_type': 'crystalline', 'fraction': 0.5},
              {'shape': 'ellipse', 'size': 0.25, 'aspect_ratio': 2,
               'angle_deg': scipy.stats.uniform(0, 180),
               'material_type': 'amorphous', 'fraction': 0.35},
              {'shape': 'circle', 'size': 0.2, 'material_type': 'void',
               'fraction': 0.15}]
    domain = msp.geometry.Square(side_length=2)
    seeds = msp.seeding.SeedList.from_info(phases, domain.area)
    seeds.position(domain)
    pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)
    return pmesh, phases, seeds


@pytest.fixture(scope='module')
def case_3d():
    """3D microstructure with crystalline, amorphous, and void phases."""
    np.random.seed(1)
    phases = [{'shape': 'sphere', 'size': scipy.stats.uniform(0.3, 0.2),
               'material_type': 'crystalline', 'fraction': 0.5},
              {'shape': 'sphere', 'size': 0.4, 'material_type': 'amorphous',
               'fraction': 0.35},
              {'shape': 'sphere', 'size': 0.35, 'material_type': 'void',
               'fraction': 0.15}]
    domain = msp.geometry.Cube(side_length=1.5)
    seeds = msp.seeding.SeedList.from_info(phases, domain.volume)
    seeds.position(domain)
    pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)
    return pmesh, phases, seeds


@pytest.fixture(scope='module')
def tri_2d(case_2d):
    pmesh, phases, _ = case_2d
    return TriMesh.from_polymesh(pmesh, phases, min_angle=20)


@pytest.fixture(scope='module')
def tri_3d(case_3d):
    pmesh, phases, _ = case_3d
    return TriMesh.from_polymesh(pmesh, phases, min_angle=10)


@pytest.fixture(scope='module')
def raster_2d(case_2d):
    pmesh, phases, _ = case_2d
    return RasterMesh.from_polymesh(pmesh, MESH_SIZE, phases)


@pytest.fixture(scope='module')
def raster_3d(case_3d):
    pmesh, phases, _ = case_3d
    return RasterMesh.from_polymesh(pmesh, MESH_SIZE, phases)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def simplex_volumes(mesh):
    """Areas of the triangles or volumes of the tetrahedra of a mesh."""
    pts = np.array(mesh.points)
    elems = np.array(mesh.elements)
    rel = pts[elems[:, 1:]] - pts[elems[:, :1]]
    if pts.shape[1] == 2:
        return 0.5 * np.abs(np.linalg.det(rel))
    return np.abs(np.linalg.det(rel)) / 6


def element_phases(mesh, seeds):
    """Phase number of each element, from the seed numbers."""
    seed_phases = np.array([seed.phase for seed in seeds])
    return seed_phases[np.array(mesh.element_attributes)]


def facet_elements(mesh):
    """Set of elements that contain all the nodes of each facet."""
    node_elems = {}
    for e_num, elem in enumerate(mesh.elements):
        for kp in elem:
            node_elems.setdefault(int(kp), set()).add(e_num)
    return [set.intersection(*[node_elems[int(kp)] for kp in facet])
            for facet in mesh.facets]


def domain_face(pts, facet):
    """Voro++ id of the domain face all the points of a facet lie on.

    Returns -1/-2 for the -x/+x faces, -3/-4 for y, -5/-6 for z, and None
    if the facet is not on the boundary of the bounding box of the points.
    """
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    f_pts = pts[list(facet)]
    for axis in range(pts.shape[1]):
        if np.allclose(f_pts[:, axis], mins[axis]):
            return -(2 * axis + 1)
        if np.allclose(f_pts[:, axis], maxs[axis]):
            return -(2 * axis + 2)
    return None


def parse_abaqus(filename):
    """Nodes, elements, element surfaces, and surface unions of a deck."""
    nodes = {}
    elems = {}
    surfaces = {}
    unions = {}
    block = None
    name = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('**'):
                continue
            if line.startswith('*'):
                key = line.split(',')[0].lower()
                if key == '*node':
                    block = 'node'
                elif key == '*element':
                    block = 'element'
                elif key == '*surface':
                    name = re.search(r'name=([^,]+)', line).group(1)
                    if 'combine=union' in line:
                        block = 'union'
                        unions[name] = []
                    else:
                        block = 'surface'
                        surfaces[name] = []
                else:
                    block = None
                continue

            vals = [v.strip() for v in line.split(',')]
            if block == 'node':
                nodes[int(vals[0])] = [float(v) for v in vals[1:]]
            elif block == 'element':
                elems[int(vals[0])] = [int(v) for v in vals[1:]]
            elif block == 'surface':
                surfaces[name].append((int(vals[0]), int(vals[1][1:])))
            elif block == 'union':
                unions[name].append(vals[0])
    return nodes, elems, surfaces, unions


def check_raster_facets(mesh, pmesh):
    """Checks on the facets of a raster mesh built from a polymesh."""
    pts = np.array(mesh.points)
    facets = np.array(mesh.facets)
    facet_atts = np.array(mesh.facet_attributes)
    elem_atts = np.array(mesh.element_attributes)
    n_kp = 2 * (pts.shape[1] - 1)

    assert facets.shape == (len(facet_atts), n_kp)
    assert len(facets) > 0

    # Attributes are polymesh facet numbers
    assert np.all(facet_atts >= 0)
    assert np.all(facet_atts < len(pmesh.facets))

    # No duplicate facets
    keys = [tuple(sorted(f)) for f in facets]
    assert len(set(keys)) == len(keys)

    poly_neighbors = np.array(pmesh.facet_neighbors)
    n_interior = 0
    n_boundary = 0
    for facet, att, elems in zip(facets, facet_atts, facet_elements(mesh)):
        neighs = poly_neighbors[att]
        on_face = domain_face(pts, facet)
        if len(elems) == 2:
            # interface: a shared face between pixels of different seeds
            # (or merged amorphous regions), approximating an interior facet
            e1, e2 = elems
            assert elem_atts[e1] != elem_atts[e2]
            assert np.min(neighs) >= 0
            n_interior += 1
        else:
            # boundary of the mesh: the domain boundary or a void
            assert len(elems) == 1
            if np.min(neighs) < 0:
                assert on_face == np.min(neighs)
                n_boundary += 1
            else:
                assert on_face is None
    assert n_interior > 0
    assert n_boundary > 0

    # Every face of the mesh boundary is a facet
    face_counts = {}
    for elem in mesh.elements:
        for local_kps in ABAQUS_FACES[len(elem)].values():
            key = tuple(sorted([int(elem[k]) for k in local_kps]))
            face_counts[key] = face_counts.get(key, 0) + 1
    boundary_faces = {k for k, n in face_counts.items() if n == 1}
    assert boundary_faces <= set(keys)


def check_raster_abaqus(mesh, pmesh, filename):
    """Checks on the Abaqus deck of a raster mesh."""
    mesh.write(filename, 'abaqus', polymesh=pmesh)
    nodes, elems, surfaces, unions = parse_abaqus(filename)

    assert len(nodes) == len(mesh.points)
    assert len(elems) == len(mesh.elements)

    facet_sets = {}
    for facet, att in zip(mesh.facets, mesh.facet_attributes):
        key = frozenset([int(kp) + 1 for kp in facet])
        facet_sets.setdefault(int(att), set()).add(key)

    n_entries = 0
    for name, entries in surfaces.items():
        att = int(name.split('-')[1])
        for elem_id, face_id in entries:
            elem = elems[elem_id]
            face = ABAQUS_FACES[len(elem)][face_id]
            face_nodes = frozenset([elem[k] for k in face])
            assert face_nodes in facet_sets[att]
            n_entries += 1
    assert n_entries == len(mesh.facets)

    assert len(unions) > 0
    for members in unions.values():
        assert len(members) > 0
        for member in members:
            assert member in surfaces


def parse_vtk_rectilinear(filename):
    """Dimensions and scalar cell data arrays of a rectilinear grid."""
    with open(filename, 'r') as file:
        text = file.read()
    dims = [int(n) for n in re.search(r'DIMENSIONS (\d+) (\d+) (\d+)',
                                      text).groups()]
    n_cells = int(re.search(r'CELL_DATA (\d+)', text).group(1))
    scalars = {}
    for block in text.split('SCALARS ')[1:]:
        name = block.split()[0]
        values = block.split('LOOKUP_TABLE default')[1].split()
        scalars[name] = [float(v) for v in values]
    return dims, n_cells, scalars


# --------------------------------------------------------------------------- #
# 1. TetGen maximum volume                                                    #
# --------------------------------------------------------------------------- #
def test_tetgen_global_max_volume(case_3d):
    pmesh, phases, _ = case_3d
    free = TriMesh.from_polymesh(pmesh, phases, min_angle=10)
    # a bound below the largest unconstrained element must refine the mesh
    max_volume = 0.5 * simplex_volumes(free).max()
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=10,
                                 max_volume=max_volume)

    vols = simplex_volumes(mesh)
    assert np.all(vols <= max_volume + 1e-9)
    assert len(mesh.elements) > len(free.elements)


def test_tetgen_per_phase_max_volume(case_3d):
    pmesh, phases, seeds = case_3d
    max_volume = 5e-4

    # Only the phase with a maximum volume is refined
    phases = copy.deepcopy(phases)
    phases[0]['max_volume'] = max_volume
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=10)
    vols = simplex_volumes(mesh)
    elem_phases = element_phases(mesh, seeds)
    assert np.all(vols[elem_phases == 0] <= max_volume + 1e-9)
    assert np.max(vols[elem_phases != 0]) > max_volume

    # A per-phase maximum larger than the global default is not capped
    phases[0]['max_volume'] = 1e-2
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=10,
                                 max_volume=max_volume)
    vols = simplex_volumes(mesh)
    elem_phases = element_phases(mesh, seeds)
    assert np.all(vols[elem_phases != 0] <= max_volume + 1e-9)
    assert np.max(vols[elem_phases == 0]) > max_volume
    assert np.all(vols[elem_phases == 0] <= 1e-2 + 1e-9)


# --------------------------------------------------------------------------- #
# 2. Triangle maximum area                                                    #
# --------------------------------------------------------------------------- #
def test_triangle_per_phase_max_volume_exceeds_global(case_2d):
    pmesh, phases, seeds = case_2d
    phases = copy.deepcopy(phases)
    phases[0]['max_volume'] = 1e-2
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=20,
                                 max_volume=1e-3)

    areas = simplex_volumes(mesh)
    elem_phases = element_phases(mesh, seeds)
    assert np.max(areas[elem_phases == 0]) > 1e-3
    assert np.all(areas[elem_phases == 0] <= 1e-2 + 1e-9)
    assert np.all(areas[elem_phases != 0] <= 1e-3 + 1e-9)


def test_triangle_infinite_max_volume_not_passed(case_2d, monkeypatch):
    pmesh, phases, _ = case_2d
    captured = {}
    orig_build = meshpy.triangle.build

    def build(info, **kwargs):
        captured['info'] = info
        captured['kwargs'] = kwargs
        return orig_build(info, **kwargs)

    monkeypatch.setattr(meshpy.triangle, 'build', build)
    mesh = TriMesh.from_polymesh(pmesh, phases, min_angle=20)

    # An infinite area would be formatted as the switch 'ainf'
    assert captured['kwargs']['max_volume'] is None
    assert captured['kwargs']['volume_constraints']

    # Without an area constraint the mesh is the min_angle-only mesh
    ref_mesh = orig_build(captured['info'], attributes=True,
                          volume_constraints=False, max_volume=None,
                          min_angle=20, generate_faces=True)
    n_ref = len(ref_mesh.elements)
    assert abs(len(mesh.elements) - n_ref) <= 0.05 * n_ref


# --------------------------------------------------------------------------- #
# 3. Raster meshes                                                            #
# --------------------------------------------------------------------------- #
def test_raster_2d_elements_counter_clockwise(raster_2d):
    pts = np.array(raster_2d.points)
    elems = np.array(raster_2d.elements)
    assert elems.shape[1] == 4

    x = pts[elems, 0]
    y = pts[elems, 1]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    signed_areas = 0.5 * np.sum(x * y_next - x_next * y, axis=1)
    assert np.all(signed_areas > 0)
    assert np.allclose(signed_areas, MESH_SIZE ** 2)


def test_raster_3d_elements_right_handed(raster_3d):
    pts = np.array(raster_3d.points)
    elems = np.array(raster_3d.elements)
    assert elems.shape[1] == 8

    # nodes 1-4 on the bottom face, 5-8 on the top face
    z = pts[elems, 2]
    assert np.allclose(z[:, :4], z[:, :1])
    assert np.allclose(z[:, 4:], z[:, :1] + MESH_SIZE)

    # scalar triple product of the edges at node 1
    p1 = pts[elems[:, 0]]
    v12 = pts[elems[:, 1]] - p1
    v14 = pts[elems[:, 3]] - p1
    v15 = pts[elems[:, 4]] - p1
    triple = np.einsum('ij,ij->i', np.cross(v12, v14), v15)
    assert np.all(triple > 0)
    assert np.allclose(triple, MESH_SIZE ** 3)


def test_raster_2d_facets(raster_2d, case_2d):
    check_raster_facets(raster_2d, case_2d[0])


def test_raster_3d_facets(raster_3d, case_3d):
    check_raster_facets(raster_3d, case_3d[0])


def test_raster_default_phases(case_2d):
    pmesh = case_2d[0]
    mesh = RasterMesh.from_polymesh(pmesh, MESH_SIZE)
    assert len(mesh.elements) == 400
    assert np.all(np.array(mesh.element_attributes) >= 0)
    assert len(mesh.facets) > 0


def test_raster_vtk_2d(raster_2d, case_2d, tmp_path):
    filename = str(tmp_path / 'raster_2d.vtk')
    raster_2d.write(filename, 'vtk', seeds=case_2d[2])

    dims, n_cells, scalars = parse_vtk_rectilinear(filename)
    assert dims == [21, 21, 1]
    assert n_cells == 400
    assert len(scalars['element_attributes']) == n_cells
    assert len(scalars['phase_numbers']) == n_cells

    # Void pixels are marked with -1, the others have their attributes
    atts = np.array(scalars['element_attributes'])
    assert np.sum(atts >= 0) == len(raster_2d.elements)
    assert set(atts[atts >= 0]) == set(raster_2d.element_attributes)


def test_raster_vtk_3d_with_void(raster_3d, case_3d, tmp_path):
    filename = str(tmp_path / 'raster_3d.vtk')
    raster_3d.write(filename, 'vtk', seeds=case_3d[2])

    dims, n_cells, scalars = parse_vtk_rectilinear(filename)
    assert dims == [16, 16, 16]
    assert n_cells == 15 ** 3
    assert len(scalars['element_attributes']) == n_cells
    assert len(scalars['phase_numbers']) == n_cells

    atts = np.array(scalars['element_attributes'])
    assert np.sum(atts < 0) > 0  # voids
    assert np.sum(atts >= 0) == len(raster_3d.elements)


def test_raster_abaqus_2d(raster_2d, case_2d, tmp_path):
    check_raster_abaqus(raster_2d, case_2d[0], str(tmp_path / 'r2d.inp'))


def test_raster_abaqus_3d(raster_3d, case_3d, tmp_path):
    check_raster_abaqus(raster_3d, case_3d[0], str(tmp_path / 'r3d.inp'))


def test_raster_plot_3d_fresh_figure(raster_3d):
    n_att = int(np.max(raster_3d.element_attributes)) + 1
    facecolors = np.array(['C' + str(i % 10) for i in range(n_att)])
    plt.close('all')
    fig = plt.figure()
    try:
        raster_3d.plot(index_by='attribute', facecolors=facecolors)
        assert len(fig.axes) == 1
        assert fig.axes[0].name == '3d'
    finally:
        plt.close('all')


# --------------------------------------------------------------------------- #
# 4. Triangle/TetGen file format                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('n_dim', [2, 3])
def test_tet_tri_files(n_dim, tri_2d, tri_3d, tmp_path):
    mesh = {2: tri_2d, 3: tri_3d}[n_dim]
    basename = str(tmp_path / 'mesh')
    mesh.write(basename, 'tet/tri')

    # edge/face file
    ext = {2: '.edge', 3: '.face'}[n_dim]
    with open(basename + ext, 'r') as file:
        lines = file.read().strip().split('\n')
    header = lines[0].split()
    assert len(header) == 2
    n_facets, n_markers = [int(n) for n in header]
    assert n_facets == len(mesh.facets)
    assert n_markers == 1
    assert len(lines) == n_facets + 1
    for i, (line, facet) in enumerate(zip(lines[1:], mesh.facets)):
        vals = [int(v) for v in line.split()]
        assert len(vals) == n_dim + 2
        assert vals[0] == i
        assert vals[1:-1] == [int(kp) for kp in facet]
        assert vals[-1] in (0, 1)

    # element file
    with open(basename + '.ele', 'r') as file:
        lines = file.read().strip().split('\n')
    n_elems, n_kp, n_atts = [int(n) for n in lines[0].split()]
    assert (n_elems, n_kp, n_atts) == (len(mesh.elements), n_dim + 1, 1)
    assert len(lines) == n_elems + 1
    for i, (line, elem) in enumerate(zip(lines[1:], mesh.elements)):
        vals = line.split()
        assert int(vals[0]) == i
        assert [int(v) for v in vals[1:1 + n_kp]] == [int(k) for k in elem]

    # node file
    with open(basename + '.node', 'r') as file:
        lines = file.read().strip().split('\n')
    assert [int(n) for n in lines[0].split()] == [len(mesh.points), n_dim,
                                                  0, 1]
    assert len(lines) == len(mesh.points) + 1


# --------------------------------------------------------------------------- #
# 5. Abaqus exterior surfaces                                                 #
# --------------------------------------------------------------------------- #
def test_abaqus_exterior_unions_defined(tri_2d, case_2d, tmp_path):
    pmesh, _, seeds = case_2d
    filename = str(tmp_path / 'tri_2d.inp')
    tri_2d.write(filename, 'abaqus', seeds=seeds, polymesh=pmesh)

    _, elems, surfaces, unions = parse_abaqus(filename)
    assert len(elems) == len(tri_2d.elements)
    assert len(unions) == 4  # one per side of the square
    for members in unions.values():
        assert len(members) > 0
        for member in members:
            assert member in surfaces

    # Facets of the polymesh on the boundary of voids have no surface
    poly_neighbors = np.array(pmesh.facet_neighbors)
    n_boundary = np.sum(np.any(poly_neighbors < 0, axis=1))
    n_union = sum([len(members) for members in unions.values()])
    assert n_union < n_boundary


# --------------------------------------------------------------------------- #
# 6. Text format precision                                                    #
# --------------------------------------------------------------------------- #
def test_txt_round_trip_exact(tri_2d, tmp_path):
    filename = str(tmp_path / 'tri_2d.txt')
    tri_2d.write(filename, 'txt')
    mesh = TriMesh.from_file(filename)

    assert np.array_equal(np.array(mesh.points), np.array(tri_2d.points))
    assert np.array_equal(np.array(mesh.elements),
                          np.array(tri_2d.elements))
    assert np.array_equal(np.array(mesh.element_attributes),
                          np.array(tri_2d.element_attributes))
    assert np.array_equal(np.array(mesh.facets), np.array(tri_2d.facets))
    assert np.array_equal(np.array(mesh.facet_attributes),
                          np.array(tri_2d.facet_attributes))


# --------------------------------------------------------------------------- #
# 7. Robustness                                                               #
# --------------------------------------------------------------------------- #
def test_str_without_optional_attributes(tri_2d, tmp_path):
    mesh = TriMesh(tri_2d.points, tri_2d.elements)
    mesh_str = str(mesh)
    assert 'Element Attributes' not in mesh_str
    assert 'Facet' not in mesh_str
    assert mesh_str.endswith(', '.join([str(k) for k in mesh.elements[-1]]))

    filename = str(tmp_path / 'no_atts.txt')
    mesh.write(filename)
    read_mesh = TriMesh.from_file(filename)
    assert np.array_equal(np.array(read_mesh.points),
                          np.array(mesh.points))
    assert np.array_equal(np.array(read_mesh.elements),
                          np.array(mesh.elements))

    # Facets without facet attributes
    mesh = TriMesh(tri_2d.points, tri_2d.elements, facets=tri_2d.facets)
    mesh_str = str(mesh)
    assert 'Facets: ' + str(len(tri_2d.facets)) in mesh_str
    assert 'Facet Attributes' not in mesh_str
    mesh.write(filename)
    read_mesh = TriMesh.from_file(filename)
    assert np.array_equal(np.array(read_mesh.facets),
                          np.array(mesh.facets))


def test_abaqus_without_polymesh(tri_2d, tmp_path):
    filename = str(tmp_path / 'no_polymesh.inp')
    tri_2d.write(filename, 'abaqus')

    _, elems, surfaces, unions = parse_abaqus(filename)
    assert len(elems) == len(tri_2d.elements)
    assert len(surfaces) == len(np.unique(tri_2d.facet_attributes))
    assert len(unions) == 0


def test_unknown_mesher_raises(case_2d):
    pmesh, phases, _ = case_2d
    with pytest.raises(ValueError) as excinfo:
        TriMesh.from_polymesh(pmesh, phases, mesher='bogus')
    assert 'Triangle/TetGen' in str(excinfo.value)
    assert 'gmsh' in str(excinfo.value)

    # the comparison is case-insensitive
    mesh = TriMesh.from_polymesh(pmesh, phases, mesher=' TRIANGLE ',
                                 min_angle=20)
    assert len(mesh.elements) > 0


def test_gmsh_default_phases(case_2d):
    pmesh = case_2d[0]
    mesh = TriMesh.from_polymesh(pmesh, mesher='gmsh')
    assert len(mesh.elements) > 0
    assert len(mesh.element_attributes) == len(mesh.elements)
    assert set(mesh.element_attributes) <= set(pmesh.seed_numbers)


def test_gmsh_no_unused_points(case_2d):
    pmesh, phases, _ = case_2d
    mesh = TriMesh.from_polymesh(pmesh, phases, mesher='gmsh')

    n_pts = len(mesh.points)
    used = np.unique(np.array(mesh.elements))
    assert np.array_equal(used, np.arange(n_pts))
    assert np.all(np.array(mesh.facets) < n_pts)
    assert len(mesh.facets) > 0


def test_sort_facets_disjoint_loops_raises():
    loop = [[0, 1], [2, 0], [1, 2]]
    assert trimesh_module._sort_facets(loop) == [[0, 1], [1, 2], [2, 0]]

    two_loops = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]
    with pytest.raises(ValueError) as excinfo:
        trimesh_module._sort_facets(two_loops)
    assert 'loop' in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 8. Plot keyword arguments as numpy arrays                                   #
# --------------------------------------------------------------------------- #
def test_plot_numpy_facecolors_per_attribute(tri_2d):
    atts = np.array(tri_2d.element_attributes)
    n_att = int(np.max(atts)) + 1
    facecolors = np.array(['C' + str(i % 10) for i in range(n_att)])

    plt.close('all')
    plt.figure()
    try:
        ax = plt.gca()
        tri_2d.plot(index_by='attribute', facecolors=facecolors)
        pc = ax.collections[-1]
        fc = pc.get_facecolor()
        assert fc.shape == (len(tri_2d.elements), 4)
        expected = mcolors.to_rgba_array(facecolors[atts])
        assert np.allclose(fc, expected)
    finally:
        plt.close('all')


# --------------------------------------------------------------------------- #
# 10. Element type checks in the writers                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('fmt', ['abaqus', 'vtk', 'tet/tri'])
def test_write_rejects_non_simplex_elements(fmt, raster_2d, tmp_path):
    mesh = TriMesh(raster_2d.points, raster_2d.elements,
                   raster_2d.element_attributes, raster_2d.facets,
                   raster_2d.facet_attributes)
    with pytest.raises(ValueError) as excinfo:
        mesh.write(str(tmp_path / 'quads'), fmt)
    assert '4 nodes' in str(excinfo.value)

    # the text format does not depend on the element type
    mesh.write(str(tmp_path / 'quads.txt'), 'txt')
    read_mesh = TriMesh.from_file(str(tmp_path / 'quads.txt'))
    assert len(read_mesh.elements) == len(mesh.elements)
