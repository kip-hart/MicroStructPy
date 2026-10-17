"""Quality and size control of periodic meshes.

The quality and size settings of the mesh (min_angle, max_volume, the
max_volume of each phase, max_edge_length) must act on periodic meshes as
on non-periodic ones, and the nodes on opposite faces must still match.
"""
import numpy as np
import pytest
import scipy.stats

import microstructpy as msp
from microstructpy.meshing import PolyMesh
from microstructpy.meshing import TriMesh
from microstructpy.seeding import SeedList

PHASES_2D = [{'shape': 'circle', 'size': scipy.stats.uniform(0.15, 0.15),
              'material_type': 'crystalline', 'max_volume': 1e-3},
             {'shape': 'ellipse', 'size': scipy.stats.uniform(0.2, 0.15),
              'aspect_ratio': scipy.stats.uniform(1.5, 1.5),
              'angle_deg': scipy.stats.uniform(0, 180),
              'material_type': 'amorphous'}]
PHASES_3D = [{'shape': 'sphere', 'size': scipy.stats.uniform(0.3, 0.2),
              'material_type': 'crystalline', 'max_volume': 5e-4},
             {'shape': 'sphere', 'size': 0.4, 'material_type': 'amorphous'}]


@pytest.fixture(scope='module')
def case_2d():
    domain = msp.geometry.Square(side_length=2, corner=(0, 0))
    seeds = SeedList.from_info(PHASES_2D, 0.55 * domain.area)
    seeds.position(domain, rtol=0.0, rng_seed=3, periodic=True)
    return domain, PolyMesh.from_seeds(seeds, domain, periodic=True)


@pytest.fixture(scope='module')
def case_3d():
    domain = msp.geometry.Cube(side_length=1.5, corner=(0, 0, 0))
    seeds = SeedList.from_info(PHASES_3D, 0.5 * domain.volume)
    seeds.position(domain, rtol=0.0, rng_seed=1, periodic=True)
    return domain, PolyMesh.from_seeds(seeds, domain, periodic=True)


def _min_angles_2d(pts, elems):
    p = pts[elems]
    angs = []
    for k in range(3):
        a = p[:, k] - p[:, (k + 1) % 3]
        b = p[:, k] - p[:, (k + 2) % 3]
        cos = np.einsum('ij,ij->i', a, b)
        cos /= np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        angs.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.min(angs, axis=0)


def _min_dihedrals(pts, tets):
    p = pts[tets]
    faces = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
    normals = []
    for f in faces:
        n = np.cross(p[:, f[1]] - p[:, f[0]], p[:, f[2]] - p[:, f[0]])
        normals.append(n / np.linalg.norm(n, axis=1)[:, None])
    dih = []
    for i in range(4):
        for j in range(i + 1, 4):
            cos = np.einsum('ij,ij->i', normals[i], normals[j])
            dih.append(180 - np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.min(dih, axis=0)


def _corner_angles_2d(pmesh):
    """Interior angles of the cells of a 2D polymesh."""
    pts = np.array(pmesh.points)
    angs = []
    for region in pmesh.regions:
        kps = sorted(set([kp for f in region for kp in pmesh.facets[f]]))
        cen = pts[kps].mean(axis=0)
        order = np.argsort(np.arctan2(pts[kps][:, 1] - cen[1],
                                      pts[kps][:, 0] - cen[0]))
        loop = pts[np.array(kps)[order]]
        for i in range(len(loop)):
            a = loop[i - 1] - loop[i]
            b = loop[(i + 1) % len(loop)] - loop[i]
            cos = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
            angs.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.array(angs)


def _check_pairs(mesh, domain):
    pts = np.array(mesh.points)
    lims = np.array(domain.limits)
    for axis, (lb, ub) in enumerate(lims):
        pairs = mesh.periodic_nodes[axis]
        n_low = np.sum(np.isclose(pts[:, axis], lb))
        n_high = np.sum(np.isclose(pts[:, axis], ub))
        assert len(pairs) == n_low == n_high > 0
        shift = np.zeros(len(lims))
        shift[axis] = ub - lb
        for lo, hi in pairs:
            assert np.array_equal(pts[hi], pts[lo] + shift)


def _attribute_phases(mesh, pmesh):
    phase_of_seed = {}
    for seed_num, phase_num in zip(pmesh.seed_numbers, pmesh.phase_numbers):
        phase_of_seed[seed_num] = phase_num
    return np.array([phase_of_seed[a] for a in mesh.element_attributes])


# --------------------------------------------------------------------------- #
# 2D                                                                          #
# --------------------------------------------------------------------------- #
def test_2d_min_angle(case_2d):
    domain, pmesh = case_2d
    mesh = TriMesh.from_polymesh(pmesh, PHASES_2D, min_angle=25)
    pts, elems = np.array(mesh.points), np.array(mesh.elements)
    angs = _min_angles_2d(pts, elems)
    # Triangle does not improve the angles of the cells themselves
    n_small_corners = np.sum(_corner_angles_2d(pmesh) < 25)
    assert np.sum(angs < 25 - 1e-6) <= n_small_corners
    assert angs.min() >= 10
    _check_pairs(mesh, domain)


def test_2d_max_volume(case_2d):
    domain, pmesh = case_2d
    mesh = TriMesh.from_polymesh(pmesh, PHASES_2D, min_angle=25,
                                 max_volume=4e-3)
    pts, elems = np.array(mesh.points), np.array(mesh.elements)
    areas = np.linalg.det(pts[elems[:, 1:]] - pts[elems[:, :1]]) / 2.0
    assert np.all(areas > 0)
    assert areas.max() <= 4e-3 * (1 + 1e-9)
    # the maximum volume of the first phase
    phases = _attribute_phases(mesh, pmesh)
    assert areas[phases == 0].max() <= 1e-3 * (1 + 1e-9)
    assert areas[phases == 1].max() > 1e-3
    assert np.isclose(areas.sum(), domain.area)
    _check_pairs(mesh, domain)


def test_2d_max_edge_length(case_2d):
    domain, pmesh = case_2d
    mesh = TriMesh.from_polymesh(pmesh, PHASES_2D, min_angle=25,
                                 max_edge_length=0.06)
    pts = np.array(mesh.points)
    facets = np.array(mesh.facets)
    lengths = np.linalg.norm(pts[facets[:, 0]] - pts[facets[:, 1]], axis=1)
    assert lengths.max() <= 0.06 * (1 + 1e-9)
    _check_pairs(mesh, domain)


# --------------------------------------------------------------------------- #
# 3D                                                                          #
# --------------------------------------------------------------------------- #
def test_3d_min_dihedral(case_3d):
    domain, pmesh = case_3d
    mesh = TriMesh.from_polymesh(pmesh, PHASES_3D, min_angle=15)
    pts, elems = np.array(mesh.points), np.array(mesh.elements)
    dih = _min_dihedrals(pts, elems)
    assert np.percentile(dih, 5) >= 15
    assert np.mean(dih < 10) <= 0.02
    _check_pairs(mesh, domain)


def test_3d_max_volume(case_3d):
    domain, pmesh = case_3d
    mesh = TriMesh.from_polymesh(pmesh, PHASES_3D, min_angle=15,
                                 max_volume=2e-3)
    pts, elems = np.array(mesh.points), np.array(mesh.elements)
    vols = np.linalg.det(pts[elems[:, 1:]] - pts[elems[:, :1]]) / 6.0
    assert np.all(vols > 0)
    assert vols.max() <= 2e-3 * (1 + 1e-9)
    phases = _attribute_phases(mesh, pmesh)
    assert vols[phases == 0].max() <= 5e-4 * (1 + 1e-9)
    assert vols[phases == 1].max() > 5e-4
    assert np.isclose(vols.sum(), domain.volume)
    _check_pairs(mesh, domain)


def test_3d_max_edge_length(case_3d):
    domain, pmesh = case_3d
    mesh = TriMesh.from_polymesh(pmesh, PHASES_3D, min_angle=15,
                                 max_edge_length=0.1)
    pts = np.array(mesh.points)
    lims = np.array(domain.limits)
    facets = np.array(mesh.facets)
    on_wall = np.zeros(len(facets), dtype=bool)
    for axis in range(3):
        for value in lims[axis]:
            on_wall |= np.all(np.isclose(pts[facets][:, :, axis], value),
                              axis=1)
    tris = facets[on_wall]
    edges = np.concatenate([np.linalg.norm(pts[tris[:, i]] -
                                           pts[tris[:, (i + 1) % 3]], axis=1)
                            for i in range(3)])
    assert edges.max() <= 0.1 * 1.1
    _check_pairs(mesh, domain)
