"""Tests for the CLI and input parsing fixes (NOTES.md 5.4, table D)"""

import contextlib
import copy
import os
import shutil
import signal
import sys

import numpy as np
import pytest

from microstructpy import _misc
from microstructpy import cli
from microstructpy import geometry
from microstructpy import seeding
from microstructpy.meshing import PolyMesh
from microstructpy.meshing import TriMesh
from microstructpy.seeding import Seed

PKG_EXAMPLES = os.path.join(os.path.dirname(cli.__file__), 'examples')

HIST_CSV = '0.5, 1\n1, 2, 2.5\n'
CDF_CSV = '1, 0\n2, 0.5\n2.5, 1\n'

MATERIAL_XML = """
    <material>
        <name> {name} </name>
        <shape> circle </shape>
        <size> {size} </size>
    </material>
"""

CDF_MATERIAL_XML = """
    <material>
        <name> {name} </name>
        <shape> circle </shape>
        <size>
            <dist_type> cdf </dist_type>
            <filename> {filename} </filename>
        </size>
    </material>
"""

DOMAIN_XML = """
    <domain>
        <shape> square </shape>
        <side_length> 1 </side_length>
    </domain>
"""


def _tiny_case():
    """A 2D case with a handful of circles, runs in a fraction of a second"""
    phases = [{'shape': 'circle', 'size': 0.4}]
    domain = geometry.factory('square', side_length=1)
    return phases, domain


def _box_polymesh(seed_numbers, phase_numbers):
    """A 2x1x1 box split into two unit cubes, both touching the boundary

    Point index = 4 x + 2 y + z, for x in {0, 1, 2} and y, z in {0, 1}.
    """
    points = [[x, y, z] for x in range(3) for y in range(2)
              for z in range(2)]
    facets = [[0, 2, 3, 1],      # 0: x = 0
              [0, 4, 5, 1],      # 1: y = 0, cube 0
              [2, 6, 7, 3],      # 2: y = 1, cube 0
              [0, 4, 6, 2],      # 3: z = 0, cube 0
              [1, 5, 7, 3],      # 4: z = 1, cube 0
              [4, 6, 7, 5],      # 5: x = 1, shared
              [4, 8, 9, 5],      # 6: y = 0, cube 1
              [6, 10, 11, 7],    # 7: y = 1, cube 1
              [4, 8, 10, 6],     # 8: z = 0, cube 1
              [5, 9, 11, 7],     # 9: z = 1, cube 1
              [8, 10, 11, 9]]    # 10: x = 2
    regions = [[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 9, 10]]
    facet_neighbors = [[0, -1], [0, -3], [0, -4], [0, -5], [0, -6], [0, 1],
                       [1, -3], [1, -4], [1, -5], [1, -6], [1, -2]]
    volumes = [1.0, 1.0]
    return PolyMesh(points, facets, regions, seed_numbers, phase_numbers,
                    facet_neighbors, volumes)


def _cube_trimesh():
    """Tetrahedral mesh of the first unit cube of :func:`_box_polymesh`

    The element attributes are the seed number (0) and the facet attributes
    are the numbers of the polymesh facets the triangles belong to.
    """
    def ind(x, y, z):
        return 4 * x + 2 * y + z

    points = [[x, y, z] for x in range(2) for y in range(2)
              for z in range(2)]
    o, e = ind(0, 0, 0), ind(1, 1, 1)
    path = [ind(1, 0, 0), ind(1, 1, 0), ind(0, 1, 0), ind(0, 1, 1),
            ind(0, 0, 1), ind(1, 0, 1)]
    elements = [[o, path[i], path[(i + 1) % 6], e] for i in range(6)]

    quads = {0: [0, 2, 3, 1], 1: [0, 4, 5, 1], 2: [2, 6, 7, 3],
             3: [0, 4, 6, 2], 4: [1, 5, 7, 3], 5: [4, 6, 7, 5]}
    facets = []
    facet_attributes = []
    for f_num, quad in quads.items():
        facets.append(quad[:3])
        facets.append([quad[0], quad[2], quad[3]])
        facet_attributes.extend([f_num, f_num])
    return TriMesh(points, elements, [0] * len(elements), facets,
                   facet_attributes)


@contextlib.contextmanager
def _time_limit(seconds):
    """Fail if the block runs longer than the given time (POSIX only)

    An alarm signal interrupts an endless loop in pure Python code, which
    is how the visibility walk used to fail.
    """
    if not hasattr(signal, 'SIGALRM'):
        yield
        return

    def handler(signum, frame):
        raise RuntimeError('the function did not terminate')

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


# --------------------------------------------------------------------------- #
# D1: <dist_type> cdf </dist_type> bins are masses, not densities            #
# --------------------------------------------------------------------------- #
def test_cdf_matches_equivalent_histogram(tmp_path):
    hist_file = tmp_path / 'hist.csv'
    hist_file.write_text(HIST_CSV)
    cdf_file = tmp_path / 'cdf.csv'
    cdf_file.write_text(CDF_CSV)

    d_hist = cli._dist_convert({'dist_type': 'histogram',
                                'filename': str(hist_file)})
    d_cdf = cli._dist_convert({'dist_type': ' cdf ',
                               'filename': str(cdf_file)})
    for dist in (d_hist, d_cdf):
        assert np.isclose(dist.cdf(2), 0.5)
        assert np.isclose(dist.mean(), 1.875)


def test_cdf_file_is_reproduced(tmp_path):
    basename = 'aphanitic_cdf.csv'
    src = os.path.join(PKG_EXAMPLES, basename)
    if not os.path.exists(src):
        src = os.path.join(PKG_EXAMPLES, basename)
    if not os.path.exists(src):
        pytest.skip(basename + ' not available')
    dst = tmp_path / basename
    shutil.copy(src, str(dst))

    xs, cdf_exp = np.loadtxt(str(dst), delimiter=',').T
    dist = cli._dist_convert({'dist_type': 'cdf', 'filename': str(dst)})
    cdf_act = dist.cdf(xs)

    # The first value can be non-zero in the file while the histogram
    # starts at 0 there; the remaining mismatch is the normalization.
    assert np.max(np.abs(cdf_act - cdf_exp)) < 2e-2
    assert np.abs(dist.mean() - np.trapz(1 - cdf_exp, xs) - xs[0]) < 1e-2


def test_pdf_is_alias_of_histogram(tmp_path):
    hist_file = tmp_path / 'hist.csv'
    hist_file.write_text(HIST_CSV)

    dist = cli._dist_convert({'dist_type': ' PDF ',
                              'filename': str(hist_file)})
    assert np.isclose(dist.cdf(2), 0.5)
    assert np.isclose(dist.mean(), 1.875)


# --------------------------------------------------------------------------- #
# D2: plot_tri visibility walk with a void region on the boundary            #
# --------------------------------------------------------------------------- #
def test_visible_regions_void_on_boundary():
    pmesh = _box_polymesh([0, 1], [0, 1])
    phases = [{'material_type': 'solid'}, {'material_type': 'void'}]

    with _time_limit(10):
        vis = cli._visible_regions(pmesh, phases)
    assert vis == {0}

    # the shared facet shows the solid region, the void's exterior nothing
    invis = set(range(-6, 0))
    assert cli._visible_neighbor([0, 1], vis, invis) == 0
    assert cli._visible_neighbor([1, 0], vis, invis) == 0
    assert cli._visible_neighbor([1, -2], vis, invis) is None
    assert cli._visible_neighbor([0, -1], vis, invis) == 0


def test_visible_regions_all_solid_and_2d():
    pmesh = _box_polymesh([0, 1], [0, 0])
    assert cli._visible_regions(pmesh, [{}]) == {0, 1}

    points = [[0, 0], [1, 0], [1, 1], [0, 1]]
    facets = [[0, 1], [1, 2], [2, 3], [3, 0]]
    pmesh_2d = PolyMesh(points, facets, [[0, 1, 2, 3]], [0], [0],
                        [[0, -3], [0, -2], [0, -4], [0, -1]], [1.0])
    assert cli._visible_regions(pmesh_2d, [{}]) == {0}


def test_plot_tri_void_on_boundary(tmp_path):
    pmesh = _box_polymesh([0, 1], [0, 1])
    tmesh = _cube_trimesh()
    seeds = seeding.SeedList([Seed.factory('sphere', phase=0, r=0.5),
                              Seed.factory('sphere', phase=1, r=0.5)])
    phases = [{'name': 'Solid', 'material_type': 'solid', 'color': 'C0'},
              {'name': 'Hole', 'material_type': 'void', 'color': 'C1'}]

    plot_file = tmp_path / 'trimesh.png'
    with _time_limit(30):
        cli.plot_tri(tmesh, phases, seeds, pmesh, [str(plot_file)])
    assert plot_file.exists()


# --------------------------------------------------------------------------- #
# D3: relative filenames inside repeated <material> tags                     #
# --------------------------------------------------------------------------- #
def test_relative_filename_in_repeated_materials(tmp_path, monkeypatch):
    (tmp_path / 'sizes_cdf.csv').write_text(CDF_CSV)
    mats = ''.join([CDF_MATERIAL_XML.format(name=n, filename='sizes_cdf.csv')
                    for n in ('A', 'B')])
    xml_file = tmp_path / 'input.xml'
    xml_file.write_text('<input>' + mats + DOMAIN_XML + '</input>')

    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(str(elsewhere))

    in_data = cli.read_input(str(xml_file))
    phases = in_data['material']
    assert len(phases) == 2
    for phase in phases:
        assert np.isclose(phase['size'].cdf(2), 0.5)


# --------------------------------------------------------------------------- #
# D8, D9: from_str booleans, infinities                                      #
# --------------------------------------------------------------------------- #
def test_from_str_words_containing_booleans():
    assert _misc.from_str('true_cdf.csv') == 'true_cdf.csv'
    assert _misc.from_str('Falsework') == 'Falsework'
    assert _misc.from_str('/data/true_north/cdf.csv') == \
        '/data/true_north/cdf.csv'


def test_from_str_booleans():
    assert _misc.from_str('True') is True
    assert _misc.from_str('true') is True
    assert _misc.from_str('FALSE') is False
    assert _misc.from_str(' false ') is False
    assert _misc.from_str('(true, false)') == (True, False)
    assert _misc.from_str('[True, FALSE]') == [True, False]


def test_from_str_special_floats():
    assert _misc.from_str('inf') == float('inf')
    assert _misc.from_str(' inf ') == float('inf')
    assert _misc.from_str('-inf') == -float('inf')
    assert np.isnan(_misc.from_str('nan'))
    assert _misc.from_str('1e-3') == 1e-3
    assert isinstance(_misc.from_str('2'), int)


def test_dict_convert_inf_setting():
    settings = cli.dict_convert({'settings': {'mesh_max_volume': ' inf '}})
    val = settings['settings']['mesh_max_volume']
    assert isinstance(val, float)
    assert val == float('inf')


# --------------------------------------------------------------------------- #
# D10: run() must not modify the caller's dictionaries                       #
# --------------------------------------------------------------------------- #
def test_run_does_not_mutate_arguments(tmp_path):
    phases, domain = _tiny_case()
    rng_seeds = {'position': 1, 'size': 2}
    filetypes = {'seeds': ['txt'], 'seeds_plot': [], 'poly_plot': [],
                 'tri_plot': []}
    rng_exp = copy.deepcopy(rng_seeds)
    ft_exp = copy.deepcopy(filetypes)

    for dirname, restart in (('a', True), ('b', False)):
        cli.run(phases, domain, restart=restart,
                directory=str(tmp_path / dirname), filetypes=filetypes,
                rng_seeds=rng_seeds, verify=False)
        assert rng_seeds == rng_exp
        assert filetypes == ft_exp

    seeds_a = (tmp_path / 'a' / 'seeds.txt').read_text()
    seeds_b = (tmp_path / 'b' / 'seeds.txt').read_text()
    assert seeds_a == seeds_b
    assert not (tmp_path / 'a' / 'seeds.png').exists()


def test_run_default_dicts_are_fresh(tmp_path):
    phases, domain = _tiny_case()
    for dirname in ('a', 'b'):
        cli.run(phases, domain, restart=False,
                directory=str(tmp_path / dirname),
                filetypes={'seeds': 'txt', 'seeds_plot': [],
                           'poly_plot': [], 'tri_plot': []})
    seeds_a = (tmp_path / 'a' / 'seeds.txt').read_text()
    seeds_b = (tmp_path / 'b' / 'seeds.txt').read_text()
    assert seeds_a == seeds_b


# --------------------------------------------------------------------------- #
# D11: <include> plus own <material>                                         #
# --------------------------------------------------------------------------- #
def test_include_and_own_materials(tmp_path):
    inc_mats = ''.join([MATERIAL_XML.format(name=n, size=0.1)
                        for n in ('A', 'B')])
    (tmp_path / 'materials.xml').write_text('<input>' + inc_mats + '</input>')

    own_mat = MATERIAL_XML.format(name='C', size=0.3)
    xml_file = tmp_path / 'input.xml'
    xml_file.write_text('<input>\n<include> materials.xml </include>\n' +
                        own_mat + DOMAIN_XML + '</input>')

    in_data = cli.read_input(str(xml_file))
    names = [p['name'] for p in in_data['material']]
    assert names == ['A', 'B', 'C']
    assert [p['size'] for p in in_data['material']] == [0.1, 0.1, 0.3]


def test_two_includes_with_materials(tmp_path):
    for fname, name in (('a.xml', 'A'), ('b.xml', 'B')):
        mat = MATERIAL_XML.format(name=name, size=0.1)
        (tmp_path / fname).write_text('<input>' + mat + '</input>')
    xml_file = tmp_path / 'input.xml'
    xml_file.write_text('<input><include> a.xml </include>' +
                        '<include> b.xml </include>' + DOMAIN_XML +
                        '</input>')

    in_data = cli.read_input(str(xml_file))
    assert [p['name'] for p in in_data['material']] == ['A', 'B']


def test_include_scalar_override(tmp_path):
    (tmp_path / 'base.xml').write_text('<material><shape> circle </shape>'
                                       '<size> 0.1 </size></material>')
    xml_file = tmp_path / 'input.xml'
    xml_file.write_text('<input><material><include> base.xml </include>'
                        '<size> 0.3 </size></material>' + DOMAIN_XML +
                        '</input>')
    in_data = cli.read_input(str(xml_file))
    assert in_data['material']['shape'] == 'circle'
    assert in_data['material']['size'] == 0.3


def test_empty_tag_is_allowed(tmp_path):
    xml_file = tmp_path / 'input.xml'
    xml_file.write_text('<input><settings><seeds_kwargs> </seeds_kwargs>'
                        '<verbose> False </verbose></settings></input>')
    file_dict = cli.input2dict(str(xml_file))
    in_data = cli.dict_convert(file_dict['input'], str(tmp_path))
    assert in_data['settings']['seeds_kwargs'] == {}
    assert in_data['settings']['verbose'] is False


# --------------------------------------------------------------------------- #
# D12: coloring by number with a single material or seed                     #
# --------------------------------------------------------------------------- #
def test_color_by_number_single_item():
    color = cli._phase_color_by(0, [{'color': 'r'}],
                                color_by='material number')
    assert len(color) == 4

    seeds = seeding.SeedList([Seed.factory('circle', phase=0, r=1)])
    for color_by in ('seed number', 'material number'):
        colors = cli._seed_colors(seeds, [{}], color_by=color_by)
        assert len(colors) == 1
        assert len(colors[0]) == 4

    # 3D and 2D polygon meshes with a single seed and phase
    pmesh = _box_polymesh([0, 0], [0, 0])
    for color_by in ('seed number', 'material number'):
        colors = cli._poly_colors(pmesh, [{}], color_by, 'viridis', 3)
        assert len(colors) == 1

    points = [[0, 0], [1, 0], [1, 1], [0, 1]]
    facets = [[0, 1], [1, 2], [2, 3], [3, 0]]
    pmesh_2d = PolyMesh(points, facets, [[0, 1, 2, 3]], [0], [0],
                        [[0, -3], [0, -2], [0, -4], [0, -1]], [1.0])
    for color_by in ('seed number', 'material number'):
        colors = cli._poly_colors(pmesh_2d, [{}], color_by, 'viridis', 2)
        assert len(colors) == 1


# --------------------------------------------------------------------------- #
# Minor: main() exit code, <tri> types, mesher case, edgecolor              #
# --------------------------------------------------------------------------- #
def test_main_exits_on_missing_input_file(tmp_path, monkeypatch, capsys):
    pattern = str(tmp_path / 'missing*.xml')
    monkeypatch.setattr(sys, 'argv', ['microstructpy', pattern])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code != 0
    assert 'missing' in capsys.readouterr().err


def test_unsupported_tri_type_error(tmp_path):
    phases, domain = _tiny_case()
    with pytest.raises(ValueError, match='nonsense'):
        cli.run(phases, domain, restart=False, directory=str(tmp_path),
                filetypes={'tri': 'nonsense'})


def test_mesher_name_case_insensitive(tmp_path, monkeypatch):
    calls = []

    class FakeRaster(object):
        @classmethod
        def from_polymesh(cls, *args, **kwargs):
            calls.append('raster')
            return cls()

    class FakeTri(object):
        @classmethod
        def from_polymesh(cls, *args, **kwargs):
            calls.append('tri')
            return cls()

    monkeypatch.setattr(cli, 'RasterMesh', FakeRaster)
    monkeypatch.setattr(cli, 'TriMesh', FakeTri)

    phases, domain = _tiny_case()
    filetypes = {'seeds_plot': [], 'poly_plot': [], 'tri_plot': []}
    cli.run(phases, domain, restart=False, directory=str(tmp_path),
            filetypes=filetypes, mesher=' Raster ')
    assert calls == ['raster']


def test_plot_poly_pops_singular_edgecolor(tmp_path, monkeypatch):
    points = [[0, 0], [1, 0], [1, 1], [0, 1]]
    facets = [[0, 1], [1, 2], [2, 3], [3, 0]]
    pmesh = PolyMesh(points, facets, [[0, 1, 2, 3]], [0], [0],
                     [[0, -3], [0, -2], [0, -4], [0, -1]], [1.0])
    received = {}

    def fake_plot_facets(self, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(PolyMesh, 'plot_facets', fake_plot_facets)
    monkeypatch.setattr(PolyMesh, 'plot', lambda self, **kwargs: None)

    plot_file = str(tmp_path / 'polymesh.png')
    cli.plot_poly(pmesh, [{}], [plot_file], edgecolor='none')
    assert 'edgecolor' not in received
    assert 'edgecolors' not in received
    assert received['color'] == ['none'] * 4
