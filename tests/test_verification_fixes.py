"""Tests for the verification fixes (NOTES.md 5.4, table D, items D4-D7)"""

import numpy as np
import scipy.stats

from microstructpy import seeding
from microstructpy import verification
from microstructpy.seeding import Seed


def _ellipse_seeds(angles, rng=None, angle_kw='angle_rad'):
    """Ellipse seeds with the given angles, optionally perturbed"""
    seeds = []
    for angle in angles:
        size = 1.0
        if rng is not None:
            size += 0.02 * rng.randn()
            angle += 0.02 * rng.randn()
        kwargs = {'size': size, 'aspect_ratio': 2, angle_kw: angle}
        seeds.append(Seed.factory('ellipse', phase=0, **kwargs))
    return seeding.SeedList(seeds)


def _rectangle_seeds(n, rng=None):
    seeds = []
    for _ in range(n):
        lengths = np.array([0.5, 0.25])
        if rng is not None:
            lengths *= 1 + 0.05 * rng.randn(2)
        seeds.append(Seed.factory('rectangle', phase=0,
                                  side_lengths=tuple(lengths)))
    return seeding.SeedList(seeds)


# --------------------------------------------------------------------------- #
# D4: angle_rad distributions are kept, phases are not modified              #
# --------------------------------------------------------------------------- #
def test_error_stats_keeps_angle_rad_distribution():
    dist = scipy.stats.uniform(loc=-0.5, scale=1.0)
    rng = np.random.RandomState(0)
    angles = dist.rvs(size=60, random_state=rng)
    seeds = _ellipse_seeds(angles)
    fit_seeds = _ellipse_seeds(angles, rng)
    phases = [{'shape': 'ellipse', 'size': 1, 'aspect_ratio': 2,
               'angle_rad': dist}]

    errs = verification.error_stats(fit_seeds, seeds, phases)

    assert phases[0]['angle_rad'] is dist
    assert errs[0]['angle_rad']['ks_statistic'] < 0.5
    assert errs[0]['angle_rad']['mae'] < 0.1


def test_error_stats_random_angle_rad():
    rng = np.random.RandomState(1)
    angles = 2 * np.pi * rng.rand(60)
    seeds = _ellipse_seeds(angles)
    phases = [{'shape': 'ellipse', 'size': 1, 'aspect_ratio': 2,
               'angle_rad': 'random'}]

    errs = verification.error_stats(seeds, seeds, phases)
    assert phases[0]['angle_rad'] == 'random'
    assert errs[0]['angle_rad']['ks_statistic'] < 0.5


# --------------------------------------------------------------------------- #
# D5: <orientation> random </orientation>                                    #
# --------------------------------------------------------------------------- #
def test_error_stats_random_orientation(tmp_path):
    rng = np.random.RandomState(2)
    angles = 360 * rng.rand(60)
    seeds = _ellipse_seeds(angles, angle_kw='angle_deg')
    fit_seeds = _ellipse_seeds(angles, rng, angle_kw='angle_deg')
    phases = [{'shape': 'ellipse', 'size': 1, 'aspect_ratio': 2,
               'orientation': 'random'}]

    errs = verification.error_stats(fit_seeds, seeds, phases)
    assert phases[0]['orientation'] == 'random'
    stats = errs[0]['orientation']
    assert stats['ks_statistic'] < 0.5
    assert stats['mae'] < 5

    fname = tmp_path / 'err_stats.txt'
    verification.write_error_stats(errs, phases, str(fname))
    assert 'orientation' in fname.read_text()


def test_error_stats_matrix_orientation():
    angles = np.full(20, 30.0)
    seeds = _ellipse_seeds(angles, angle_kw='angle_deg')
    fit_seeds = _ellipse_seeds(angles, np.random.RandomState(3),
                               angle_kw='angle_deg')
    ct, st = np.cos(np.radians(30)), np.sin(np.radians(30))
    phases = [{'shape': 'ellipse', 'size': 1, 'aspect_ratio': 2,
               'orientation': np.array([[ct, -st], [st, ct]])}]

    errs = verification.error_stats(fit_seeds, seeds, phases)
    assert errs[0]['orientation']['mae'] < 5


def test_error_stats_orientation_skipped_in_3d():
    seeds = seeding.SeedList([Seed.factory('ellipsoid', phase=0, a=1,
                                           b=0.5, c=0.5) for _ in range(5)])
    phases = [{'shape': 'ellipsoid', 'a': 1, 'b': 0.5, 'c': 0.5,
               'orientation': 'random'}]
    errs = verification.error_stats(seeds, seeds, phases)
    assert errs[0]['orientation'] == {}


# --------------------------------------------------------------------------- #
# D6: vector-valued parameters are handled per component                     #
# --------------------------------------------------------------------------- #
def test_mle_dist_per_component():
    values = np.array([[0.5, 0.25], [0.52, 0.24], [0.48, 0.26], [0.5, 0.25]])
    dists = [scipy.stats.norm(0.5, 0.1), scipy.stats.norm(0.25, 0.1)]

    mles = verification._mle_dist(values, dists)
    assert len(mles) == 2
    assert abs(mles[0].mean() - 0.5) < 0.03
    assert abs(mles[1].mean() - 0.25) < 0.03

    # tuples of constants give the component means
    means = verification._mle_dist(values, (0.5, 0.25))
    assert np.allclose(means, [0.5, 0.25])


def test_safe_rvs_tuple():
    samples = verification._safe_rvs((0.4, 0.2), 5)
    assert samples.shape == (5, 2)
    assert np.allclose(samples[:, 0], 0.4)
    assert np.allclose(samples[:, 1], 0.2)

    dists = [scipy.stats.uniform(0, 1), 3]
    samples = verification._safe_rvs(dists, 7)
    assert samples.shape == (7, 2)
    assert np.allclose(samples[:, 1], 3)


def test_write_mle_phases_vector(tmp_path):
    inp_phases = [{'name': 'Bricks', 'side_lengths': (0.5, 0.25)}]
    out_phases = [{'name': 'Bricks', 'side_lengths': [0.51, 0.24]}]
    fname = tmp_path / 'mles.txt'
    verification.write_mle_phases(inp_phases, out_phases, str(fname))

    text = fname.read_text()
    assert 'side_lengths[0]' in text
    assert 'side_lengths[1]' in text
    assert '0.51' in text


def test_verification_rectangle_side_lengths(tmp_path):
    rng = np.random.RandomState(4)
    seeds = _rectangle_seeds(30)
    fit_seeds = _rectangle_seeds(30, rng)
    phases = [{'shape': 'rectangle', 'side_lengths': (0.5, 0.25)}]

    errs = verification.error_stats(fit_seeds, seeds, phases)
    assert len(errs[0]['side_lengths']) == 2
    for j, length in enumerate((0.5, 0.25)):
        assert errs[0]['side_lengths'][j]['mae'] < 0.1 * length
        assert 'ks_statistic' in errs[0]['side_lengths'][j]

    err_file = tmp_path / 'err_stats.txt'
    verification.write_error_stats(errs, phases, str(err_file))
    text = err_file.read_text()
    assert 'side_lengths[0]' in text
    assert 'side_lengths[1]' in text

    mles = verification.mle_phases(fit_seeds, phases)
    lengths = np.array([s.geometry.side_lengths for s in fit_seeds])
    assert np.allclose(mles[0]['side_lengths'], lengths.mean(axis=0))
    verification.write_mle_phases(phases, mles, str(tmp_path / 'mles.txt'))

    verification.plot_distributions(fit_seeds, phases, str(tmp_path), 'png')
    assert (tmp_path / 'side_lengths_pdf.png').exists()
    assert (tmp_path / 'side_lengths_cdf.png').exists()


def test_verification_distributed_axes(tmp_path):
    dists = [scipy.stats.uniform(0.8, 0.4), scipy.stats.uniform(0.4, 0.2)]
    rng = np.random.RandomState(5)
    seeds = seeding.SeedList([
        Seed.factory('ellipse', phase=0, axes=[d.rvs(random_state=rng)
                                               for d in dists])
        for _ in range(40)])
    phases = [{'shape': 'ellipse', 'axes': dists}]

    errs = verification.error_stats(seeds, seeds, phases)
    assert len(errs[0]['axes']) == 2
    assert errs[0]['axes'][1]['ks_statistic'] < 0.5

    mles = verification.mle_phases(seeds, phases)
    assert abs(mles[0]['axes'][0].mean() - 1.0) < 0.1
    assert abs(mles[0]['axes'][1].mean() - 0.5) < 0.1

    verification.plot_distributions(seeds, phases, str(tmp_path), 'png')
    assert (tmp_path / 'axes_pdf.png').exists()


# --------------------------------------------------------------------------- #
# D7: extra phase fields are ignored                                         #
# --------------------------------------------------------------------------- #
def test_extra_phase_fields_ignored(tmp_path):
    dist = scipy.stats.uniform(loc=0.5, scale=0.5)
    rng = np.random.RandomState(6)
    sizes = dist.rvs(size=30, random_state=rng)
    seeds = seeding.SeedList([Seed.factory('circle', phase=0, size=s)
                              for s in sizes])
    phases = [{'shape': 'circle', 'size': dist, 'notes': 'x',
               'max_volume': 0.1, 'color': 'C2'}]

    verification.plot_distributions(seeds, phases, str(tmp_path), 'png')
    names = [p.name for p in tmp_path.iterdir()]
    assert 'size_pdf.png' in names
    assert 'size_cdf.png' in names
    assert not any(n.startswith(('notes', 'max_volume')) for n in names)

    mles = verification.mle_phases(seeds, phases)
    assert set(mles[0].keys()) == {'size'}
    verification.write_mle_phases(phases, mles, str(tmp_path / 'mles.txt'))

    errs = verification.error_stats(seeds, seeds, phases)
    assert set(errs[0].keys()) == {'size'}
    assert errs[0]['size']['ks_statistic'] < 0.5


# --------------------------------------------------------------------------- #
# Minor: axis labels and table headings                                      #
# --------------------------------------------------------------------------- #
def test_axis_labels():
    assert verification._axis_label('radius') == 'Radius'
    assert verification._axis_label('angle_rad') == 'Angle (radians)'
    assert verification._axis_label('angle_deg') == 'Angle (degrees)'
    assert verification._axis_label('orientation') == \
        'Orientation (degrees)'
    assert verification._axis_label('side_lengths') == 'Side Lengths'


def test_mle_headings_strip_suffix():
    hdr1, hdr2 = verification._mle_hdr(['i', 'name', 'kw', 'nu_inp',
                                        'nu_out', 's_inp', 's_out'])
    assert hdr1 == ['', '', '', 'Input', 'Output', 'Input', 'Output']
    assert hdr2 == ['#', 'Name', 'Parameter', 'nu', 'nu', 's', 's']
