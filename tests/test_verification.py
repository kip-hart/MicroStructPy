from __future__ import division

import numpy as np
import pytest
import scipy.stats

from microstructpy import meshing
from microstructpy import seeding
from microstructpy import verification


def test_volume_fractions():
    a = (0, 0)
    b = (1, 0)
    c = (1, 1)
    d = (0, 1)
    e = (2, 0)

    pts = [a, b, c, d, e]
    facets = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 2)]
    regions = [(0, 1, 2, 3), (4, 5, 1)]
    phase_nums = [0, 1]

    pmesh = meshing.PolyMesh(pts, facets, regions, phase_numbers=phase_nums)
    exp_vols = [2 / 3, 1 / 3]
    act_vols = verification.volume_fractions(pmesh, 2)
    assert np.isclose(act_vols, exp_vols).all()


@pytest.fixture
def volume_fractions():
    vol_fracs = [1 / 4, 3 / 4]
    phases = [{'volume': .25}, {'volume': .75}]
    yield vol_fracs, phases



def test_write_volume_fractions(tmpdir, volume_fractions):
    filename = tmpdir.join('volume_fraction.txt')
    vol_fracs, phases = volume_fractions
    verification.write_volume_fractions(vol_fracs, phases, str(filename))
    assert filename.check()


def test_plot_volume_fractions(tmpdir, volume_fractions):
    filename = tmpdir.join('volume_fraction.png')
    vol_fracs, phases = volume_fractions
    verification.plot_volume_fractions(vol_fracs, phases, str(filename))
    assert filename.check()


def test_plot_distributions(tmpdir):
    s1 = seeding.Seed.factory('circle', phase=0)
    s2 = seeding.Seed.factory('circle', phase=1)
    seeds = seeding.SeedList([s1, s2, s1, s2])
    phases = [{'r': 1}, {'r': scipy.stats.uniform(loc=0, scale=2)}]

    verification.plot_distributions(seeds, phases, dirname=str(tmpdir),
                                    ext='png')
    assert tmpdir.join('r_pdf.png').check()
    assert tmpdir.join('r_cdf.png').check()


def test__r2():
    x = np.arange(10)
    assert np.isclose(verification._r2(x, x), 1)