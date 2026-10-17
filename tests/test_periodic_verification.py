"""Verification of periodic microstructures: split grains are unwrapped."""
import numpy as np
import scipy.stats

import microstructpy as msp
from microstructpy import _misc
from microstructpy import verification
from microstructpy.meshing import PolyMesh
from microstructpy.meshing import TriMesh
from microstructpy.seeding import SeedList


def test_unwrap_points():
    dom_lims = [(0.0, 2.0), (0.0, 3.0)]
    pts = [[1.9, 0.1], [0.1, 2.9], [1.0, 1.0]]
    out = _misc.unwrap_points(pts, [0.1, 0.2], [True, True], dom_lims)
    assert np.allclose(out, [[-0.1, 0.1], [0.1, -0.1], [1.0, 1.0]])
    out = _misc.unwrap_points(pts, [0.1, 0.2], [True, False], dom_lims)
    assert np.allclose(out, [[-0.1, 0.1], [0.1, 2.9], [1.0, 1.0]])


def test_split_grains_fit_after_unwrapping():
    phases = [{'shape': 'circle', 'size': scipy.stats.uniform(0.2, 0.2),
               'material_type': 'crystalline'}]
    domain = msp.geometry.Square(side_length=3, corner=(0, 0))
    seeds = SeedList.from_info(phases, domain.area)
    seeds.position(domain, rng_seed=1, periodic=True)
    pmesh = PolyMesh.from_seeds(seeds, domain, periodic=True)
    tmesh = TriMesh.from_polymesh(pmesh, phases, min_angle=20)

    fit = verification.seeds_of_best_fit(seeds, phases, pmesh, tmesh)
    n_pieces = np.bincount(pmesh.seed_numbers, minlength=len(seeds))
    split = n_pieces > 1
    assert split.sum() > 0
    assert all([s.geometry is not None for s in fit])

    r_in = np.array([s.geometry.r for s in seeds])
    r_fit = np.array([s.geometry.r for s in fit])
    rel_err = np.abs(r_fit - r_in) / r_in
    # the fits of the split grains are as good as the others, and their
    # centers stay near the seeds (not at the average of two images)
    assert rel_err[split].mean() < rel_err[~split].mean() + 0.05
    # without unwrapping, the center of a split grain would lie between
    # its pieces, about half a domain length away from the seed
    for seed, fit_seed, is_split in zip(seeds, fit, split):
        if not is_split:
            continue
        d_cen = np.array(fit_seed.geometry.center) - np.array(seed.position)
        assert np.linalg.norm(d_cen) < seed.geometry.r
