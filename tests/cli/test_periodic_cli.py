"""End-to-end test of a periodic microstructure through the CLI."""
import os

import numpy as np
import pytest

import microstructpy as msp
from microstructpy import cli
from microstructpy.meshing import PolyMesh
from microstructpy.meshing import TriMesh
from microstructpy.seeding import SeedList

PERIODIC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<input>
    <material>
        <name> Matrix </name>
        <shape> circle </shape>
        <size>
            <dist_type> uniform </dist_type>
            <loc> 0.25 </loc>
            <scale> 0.15 </scale>
        </size>
        <fraction> 2 </fraction>
    </material>
    <material>
        <name> Inclusions </name>
        <shape> ellipse </shape>
        <size> 0.4 </size>
        <aspect_ratio> 2 </aspect_ratio>
        <angle_deg>
            <dist_type> uniform </dist_type>
            <loc> 0 </loc>
            <scale> 180 </scale>
        </angle_deg>
        <fraction> 1 </fraction>
    </material>

    <domain>
        <shape> square </shape>
        <side_length> 3 </side_length>
        <corner> 0, 0 </corner>
        <periodic> {periodic} </periodic>
    </domain>

    <settings>
        <directory> {directory} </directory>
        <verbose> False </verbose>
        <mesh_min_angle> 20 </mesh_min_angle>
        <mesh_max_edge_length> 0.1 </mesh_max_edge_length>
        <verify> True </verify>
        {extra}
    </settings>
</input>
"""


def _run(tmp_path, periodic, extra=''):
    out_dir = tmp_path / 'out'
    xml = tmp_path / 'input.xml'
    xml.write_text(PERIODIC_XML.format(periodic=periodic,
                                       directory=str(out_dir), extra=extra))
    cli.run_file(str(xml))
    return out_dir


def test_periodic_input_read():
    in_data = cli.dict_convert({'domain': {'shape': 'square',
                                           'periodic': ' xy '}})
    assert in_data['domain']['periodic'].strip() == 'xy'


def test_periodic_run(tmp_path):
    out_dir = _run(tmp_path, 'xy')
    for name in ('seeds.txt', 'polymesh.txt', 'trimesh.txt', 'seeds.png',
                 'polymesh.png', 'trimesh.png'):
        assert os.path.exists(str(out_dir / name))

    pmesh = PolyMesh.from_file(str(out_dir / 'polymesh.txt'))
    tmesh = TriMesh.from_file(str(out_dir / 'trimesh.txt'))
    assert pmesh.periodic_axes == [True, True]
    assert tmesh.periodic_axes == [True, True]
    assert np.isclose(sum(pmesh.volumes), 9.0)

    pts = np.array(tmesh.points)
    for axis in (0, 1):
        shift = np.zeros(2)
        shift[axis] = 3
        pairs = tmesh.periodic_nodes[axis]
        assert len(pairs) == np.sum(np.isclose(pts[:, axis], 0))
        for lo, hi in pairs:
            assert np.array_equal(pts[hi], pts[lo] + shift)

    # verification ran (grains split by the faces are unwrapped)
    assert os.path.exists(str(out_dir / 'verification' / 'mles.txt'))


def test_periodic_run_single_axis(tmp_path):
    out_dir = _run(tmp_path, 'y')
    tmesh = TriMesh.from_file(str(out_dir / 'trimesh.txt'))
    assert tmesh.periodic_axes == [False, True]
    assert list(tmesh.periodic_nodes) == [1]


def test_periodic_margin_edge_opt(tmp_path):
    # the margin is passed to the edge optimization: no piece thinner than
    # the margin (half of mesh_max_edge_length) is left at the faces,
    # unless the optimizer could not fix it, and the mesh is periodic
    extra = '<edge_opt> True </edge_opt>\n<edge_opt_n_iter> 3 '
    extra += '</edge_opt_n_iter>\n<periodic_margin> auto </periodic_margin>'
    out_dir = _run(tmp_path, 'xy', extra)
    pmesh = PolyMesh.from_file(str(out_dir / 'polymesh.txt'))
    tmesh = TriMesh.from_file(str(out_dir / 'trimesh.txt'))
    assert pmesh.periodic_axes == [True, True]
    assert np.isclose(sum(pmesh.volumes), 9.0)
    pts = np.array(tmesh.points)
    for axis in (0, 1):
        for lo, hi in tmesh.periodic_nodes[axis]:
            shift = np.zeros(2)
            shift[axis] = 3
            assert np.array_equal(pts[hi], pts[lo] + shift)
    # the seeds written are the optimized ones: they reproduce the mesh
    seeds = SeedList.from_file(str(out_dir / 'seeds.txt'))
    domain = msp.geometry.Square(side_length=3, corner=(0, 0))
    pmesh_re = PolyMesh.from_seeds(seeds, domain, periodic=True)
    assert np.allclose(np.sort(pmesh_re.volumes), np.sort(pmesh.volumes),
                       rtol=0, atol=1e-9)


def test_periodic_margin_setting():
    inf = float('inf')
    h_2d = np.sqrt(4 * 0.004 / np.sqrt(3))
    assert np.isclose(cli._periodic_margin('auto', 2, 0.004, inf), 0.5 * h_2d)
    assert np.isclose(cli._periodic_margin('auto', 2, 0.004, 0.05), 0.025)
    h_3d = (6 * np.sqrt(2) * 0.02) ** (1.0 / 3)
    assert np.isclose(cli._periodic_margin('auto', 3, 0.02, inf), 0.5 * h_3d)
    assert cli._periodic_margin('auto', 3, inf, inf) == 0
    assert cli._periodic_margin(0.03, 2, 0.004, inf) == 0.03
    assert cli._periodic_margin('none', 2, 0.004, inf) == 0
    with pytest.raises(ValueError):
        cli._periodic_margin('big', 2, 1, 1)
