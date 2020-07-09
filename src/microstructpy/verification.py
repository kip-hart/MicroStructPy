"""Verification

This module contains functions related to mesh verification.

"""

# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #

from __future__ import division
from __future__ import print_function

import copy
import os

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import microstructpy.geometry
from microstructpy import _misc
from microstructpy import seeding

__author__ = 'Kenneth (Kip) Hart'

styles_vec = ['o', '^', 's', 'D', 'v', '*']

hist_class = scipy.stats._continuous_distns.rv_histogram
discr_rvs = scipy.stats._discrete_distns._distn_names
conti_rvs = scipy.stats._continuous_distns._distn_names

ori_deg_kws = ['orientation', 'angle', 'angle_deg']
ori_rad_kws = ['angle_rad']


# --------------------------------------------------------------------------- #
#                                                                             #
# Volume Fractions                                                            #
#                                                                             #
# --------------------------------------------------------------------------- #
def volume_fractions(poly_mesh, n_phases):
    """Verify volume fractions

    This function computes the volume fractions of each phase in the output
    mesh. It does so by summing the volumes of the cells in the polygonal
    mesh.

    Args:
        poly_mesh (PolyMesh): The polygonal/polyhedral mesh.
        n_phases (int): Number of phases.

    Returns:
        numpy.ndarray: Volume fractions of each phase in the poly mesh.

    """
    region_phases = np.array(poly_mesh.phase_numbers, dtype='int')
    region_volumes = np.array(poly_mesh.volumes, dtype='float')
    vol_list = np.full(n_phases, 0, dtype='float')
    for i in range(n_phases):
        mask = np.isclose(region_phases, i)
        vol_list[i] = np.sum(region_volumes[mask])

    v_total = np.sum(vol_list)
    if v_total == 0:
        return vol_list
    return vol_list / v_total


def write_volume_fractions(vol_fracs, phases, filename='volume_fractions.txt'):
    """Write volume fractions to a file

    Write the volume fractions verification out to a file.
    The output columns are:

        1. Phase number
        2. Phase name
        3. Input relative volume (average, if distributed)
        4. Output relative volume
        5. Input volume fraction (average, if distributed)
        6. Output volume fraction

    The first three lines of the output file are headings.

    Args:
        vol_fracs (list or numpy.ndarray): Volume fractions of the output mesh.
        phases (list): List of phase dictionaries.
        filename (str): *(optional)* Name of file to write.
            Defaults to ``volume_fractions.txt``.

    Returns:
        none, prints formatted volume fraction verification table to file

    """
    # Sample input volume distributions
    n_inp = 5000
    vol_trials = np.full((n_inp, len(phases)), -1, dtype='float')
    for i, phase in enumerate(phases):
        mask = vol_trials[:, i] < 0
        while np.any(mask):
            new_vals = _safe_rvs(phase.get('fraction', 1), np.sum(mask))
            vol_trials[mask, i] = new_vals
            mask = vol_trials[:, i] < 0

    # Determine input and output relative volumes
    vols_exp = np.array([_safe_mean(p.get('fraction', 1)) for p in phases])
    vols_act = np.sum(vols_exp) * np.array(vol_fracs)

    # Determine input volume fractions
    vol_total = np.sum(vol_trials, axis=1)

    k = 1 / vol_total
    frac_trials = vol_trials * k.reshape(-1, 1)
    vf_exp = np.mean(frac_trials, axis=0)

    # Create table
    hdr1 = ['',
            '',
            'Input (Avg.)',
            'Output',
            'Input (Avg.)',
            'Output']
    hdr2 = ['#',
            'Name',
            'Relative Volume',
            'Relative Volume',
            'Volume Fraction',
            'Volume Fraction']

    rows = [hdr1, hdr2]
    for i, phase in enumerate(phases):
        name = phase.get('name', 'Material ' + str(i + 1))
        row = [i,
               name,
               vols_exp[i],
               vols_act[i],
               vf_exp[i],
               vol_fracs[i]]
        rows.append(row)

    # Write table
    file_dir = os.path.dirname(filename)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(filename, 'w') as file:
        file.write(_fixed_width_table(rows, [0, 2, 3, 4, 5]))


def plot_volume_fractions(vol_fracs, phases, filename='volume_fractions.png'):
    """Plot volume fraction verification

    This function creates a bar chart comparing the input and output volume
    fractions. If the input volume fraction is distributed, the top of the
    bar will be a curve representing the CDF of the distribution.

    Args:
        vol_fracs (list or numpy.ndarray): Output volume fractions.
        phases (list): List of phase dictionaries
        filename (str or list): *(optional)* Filename(s) to save the plot.
            Defaults to ``volume_fractions.png``.

    Returns:
        none, writes plot to file.
    """
    # Sample input volume distributions
    n_inp = 5000
    vol_trials = np.full((n_inp, len(phases)), -1, dtype='float')
    for i, phase in enumerate(phases):
        mask = vol_trials[:, i] < 0
        while np.any(mask):
            new_vals = _safe_rvs(phase.get('fraction', 1), np.sum(mask))
            vol_trials[mask, i] = new_vals
            mask = vol_trials[:, i] < 0

    # Determine input volume fractions
    vol_total = np.sum(vol_trials, axis=1)
    k = 1 / vol_total
    frac_trials = vol_trials * k.reshape(-1, 1)

    # Initialize plot
    plt.clf()
    total_w = 0.8
    bar_w = 0.5 * total_w
    n_plt = 201
    q_plt = np.linspace(0, 1, n_plt)
    x_cens = np.arange(len(phases))

    # Plot input volume fraction bars
    for i, phase in enumerate(phases):
        x_plt = x_cens[i] - bar_w + bar_w * q_plt
        y_plt = np.quantile(frac_trials[:, i], q_plt)

        x_plt = np.concatenate((x_plt, [x_plt[-1], x_plt[0]]))
        y_plt = np.concatenate((y_plt, [0, 0]))

        if i == 0:
            plt.fill(x_plt, y_plt, 'white', edgecolor='k', label='Input',
                     zorder=3)
        else:
            plt.fill(x_plt, y_plt, 'white', edgecolor='k', zorder=3)

    # Plot output volume fraction bars
    plt.bar(x_cens + 0.5 * bar_w, vol_fracs, width=bar_w, label='Actual',
            color='gray', edgecolor='black', zorder=4)

    # Format plot
    xticks = range(len(phases))
    xlbls = [p.get('name', 'Material ' + str(i + 1)) for i, p in
             enumerate(phases)]
    plt.xticks(xticks, xlbls)
    plt.gca().set_ylim(bottom=0)
    plt.grid(axis='y')

    plt.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.155),
               fancybox=True)
    plt.ylabel('Volume Fraction')
    plt.title('Volume Fraction Verification')

    if not isinstance(filename, list):
        filename = [filename]

    for fname in filename:
        write_dir = os.path.dirname(fname)
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        plt.savefig(fname)


# --------------------------------------------------------------------------- #
#                                                                             #
# Seeds of Best Fit                                                           #
#                                                                             #
# --------------------------------------------------------------------------- #
def seeds_of_best_fit(seeds, phases, pmesh, tmesh):
    """Calculate seed geometries of best fit

    This function computes the seeds of best fit for the resultant polygonal
    and triangular meshes. It calls the the ``best_fit`` function of each
    seed's geometry, then copies the other seed attributes to create a new
    :class:`.SeedList`.

    The points on the faces of the grains are used to determine a fit geometry.
    Points on the exterior of the domain are not used since they would alter
    the shape of the best fit seed.

    Args:
        seeds (SeedList): List of seed geometries.
        phases (list): List of material phases. See :ref:`phase_dict_guide`
            for more information on formatting.
        pmesh (PolyMesh): Resultant polygonal/polyhedral mesh.
        tmesh (TriMesh): Resultant triangular/tetrahedral mesh.

    Returns:
        SeedList: List of seeds of best fit.

    .. note:

        If the geometry of best fit for the seed is ill-conditioned (e.g.
        negative axes for an ellipse), the seed geometry is set to None.
    """
    poly_pts = np.array(pmesh.points, dtype='float')
    poly_neighs = np.array(pmesh.facet_neighbors, dtype='int')
    tri_parents = np.array(tmesh.facet_attributes, dtype='int')
    tri_facets = np.array(tmesh.facets, dtype='int')
    tri_pts = np.array(tmesh.points, dtype='float')
    seed_nums = np.array(pmesh.seed_numbers, dtype='int')

    poly_neighs[poly_neighs < 0] = -1
    tri_facet_neigh_cells = poly_neighs[tri_parents]

    tri_facet_neigh_seeds = seed_nums[tri_facet_neigh_cells]
    tri_facet_neigh_seeds[tri_facet_neigh_cells < 0] = -1
    poly_facet_neigh_seeds = seed_nums[poly_neighs]
    poly_facet_neigh_seeds[poly_neighs < 0] = -1

    tri_facet_is_ext = np.min(tri_facet_neigh_seeds, axis=-1) < 0
    poly_facet_is_ext = np.min(poly_facet_neigh_seeds, axis=-1) < 0

    n_dim = seeds[0].geometry.n_dim
    fit_seeds = []
    for i, seed in enumerate(seeds):
        p = seed.phase
        mat_type = phases[p].get('material_type', 'crystalline')
        if mat_type in _misc.kw_solid:
            mask = np.any(tri_facet_neigh_seeds == i, axis=-1)
            mask &= ~tri_facet_is_ext
            seed_facets = tri_facets[mask]
            kps = np.unique(seed_facets)

            if len(kps) <= n_dim:
                mask = np.any(tri_facet_neigh_seeds == i, axis=-1)
                seed_facets = tri_facets[mask]
                kps = np.unique(seed_facets)

            seed_pts = tri_pts[kps]

        if (mat_type not in _misc.kw_solid) or (len(kps) <= n_dim):
            mask = np.any(poly_facet_neigh_seeds == i, axis=-1)
            mask &= ~poly_facet_is_ext
            seed_facets = [f for f, m in zip(pmesh.facets, mask) if m]
            kps = np.unique([kp for f in seed_facets for kp in f])

            if len(kps) <= n_dim:
                mask = np.any(poly_facet_neigh_seeds == i, axis=-1)
                seed_facets = [f for f, m in zip(pmesh.facets, mask) if m]
                kps = np.unique([kp for f in seed_facets for kp in f])
            seed_pts = poly_pts[kps.astype('int')]

        try:
            fit_geom = seed.geometry.best_fit(seed_pts)
        except ValueError:
            fit_geom = None
        fit_seed = copy.deepcopy(seed)

        # Check for ill-conditioned seeds of best fit
        if _ill_conditioned(fit_geom):
            fit_seed.geometry = None
        else:
            fit_seed.geometry = fit_geom

        fit_seeds.append(fit_seed)
    return seeding.SeedList(fit_seeds)


def _ill_conditioned(geom):
    if isinstance(geom, microstructpy.geometry.Ellipse):
        return min(geom.axes) <= 0
    return False


# --------------------------------------------------------------------------- #
#                                                                             #
# Compare to Phases                                                           #
#                                                                             #
# --------------------------------------------------------------------------- #
def plot_distributions(seeds, phases, dirname='.', ext='png', poly_mesh=None,
                       verif_mask=None):
    """Plot comparison between input and output distributions

    This function takes seeds and compares them against the input phases.
    A polygon mesh can be included for cases where grains are given an
    area or volume distribution, rather than size/shape/etc.

    This function creates both PDF and CDF plots.

    Args:
        seeds (SeedList): List of seeds to compare.
        phases (list): List of phase dictionaries.
        dirname (str): *(optional)* Plot output directory.
            Defaults to ``.``.
        ext (str or list): *(optional)* File extension(s) of the output plots.
            Defaults to ``'png'``.
        poly_mesh (PolyMesh): *(optional)* Polygonal mesh, useful for phases
            with an area or volume distribution.

    Returns:
        none, creates plot files.
    """
    if verif_mask is None:
        verif_mask = np.full(len(seeds), True)

    # Get geometry of seeds
    comp_phases = _phase_values(seeds, phases, poly_mesh, verif_mask)

    # Determine all geometry keywords used
    kws = set([kw for phase in comp_phases for kw in phase.keys()])
    kws -= set(_misc.gen_kws)

    # Create Plots
    n_dim = len(seeds[0].position)
    for kw in sorted(kws):
        if (kw in _misc.ori_kws) and (n_dim == 3):
            continue

        # PDF plot
        plt.clf()
        line_colors = []
        line_labels = []
        ymax = 0
        for i, phase in enumerate(phases):
            if (kw not in comp_phases[i]) or (kw not in phase):
                continue

            # Get input PDF
            ymax_inp_pdf = _plot_inp_pdf(kw, i, phase)
            ymax = max(ymax, ymax_inp_pdf)

            # Get output PDF
            comp_phase = comp_phases[i]
            ymax_out_pdf, lcs, lls = _plot_out_pdf(kw, i, phase, comp_phase)
            ymax = max(ymax, ymax_out_pdf)
            line_colors.extend(lcs)
            line_labels.extend(lls)

        # Format PDF plot
        color_legend = plt.legend(line_colors, line_labels, loc=4)

        style_labels = ['Input', 'Actual']
        dashed_line = Line2D([0], [0], color='k', ls=':')
        solid_line = Line2D([0], [0], color='k', ls='-')
        style_lines = [dashed_line, solid_line]
        style_labels = ['Input', 'Actual']
        style_legend = plt.legend(style_lines, style_labels, loc=2)

        plt.gca().add_artist(style_legend)
        plt.gca().add_artist(color_legend)

        plt.grid(True)
        xlbl = ' '.join([s.capitalize() for s in kw.split('_')])
        xlbl = xlbl.replace('Rad', '(radians)').replace('Deg', '(degrees)')
        xlbl = xlbl.replace('Orientation', 'Orientation (degrees)')
        plt.xlabel(xlbl)
        plt.ylabel('Probability Density Function')

        plt.ylim([0, 1.1 * ymax])

        plt.title('PDF Comparison')

        # Save PDF plot
        if not isinstance(ext, list):
            ext = [ext]

        basename = kw + '_pdf.'
        filepath = os.path.join(dirname, basename)
        for extension in ext:
            filename = filepath + extension
            plt.savefig(filename)
        plt.close()

        # CDF Plot
        plt.clf()
        line_colors = []
        line_labels = []
        for i, phase in enumerate(phases):
            if (kw not in comp_phases[i]) or (kw not in phase):
                continue

            # Get input CDF
            _plot_inp_cdf(kw, i, phase)

            # Get output PDF
            comp_phase = comp_phases[i]
            lcs, lls = _plot_out_cdf(kw, i, phase, comp_phase)
            line_colors.extend(lcs)
            line_labels.extend(lls)

        # Format CDF plot
        color_legend = plt.legend(line_colors, line_labels, loc=4)
        style_legend = plt.legend(style_lines, style_labels, loc=2)

        plt.gca().add_artist(style_legend)
        plt.gca().add_artist(color_legend)

        plt.grid(True)
        xlbl = ' '.join([s.capitalize() for s in kw.split('_')])
        xlbl = xlbl.replace('Rad', '(radians)').replace('Deg', '(degrees)')
        xlbl = xlbl.replace('Orientation', 'Orientation (degrees)')
        plt.xlabel(xlbl)
        plt.ylabel('Cumulative Distribution Function')

        plt.ylim([0, 1])

        plt.title('CDF Comparison')

        # Save PDF plot
        if not isinstance(ext, list):
            ext = [ext]

        basename = kw + '_cdf.'
        filepath = os.path.join(dirname, basename)
        for extension in ext:
            filename = filepath + extension
            plt.savefig(filename)
        plt.close()


def _plot_inp_pdf(kw, i, phase):
    ymax = 0
    inp_dist = phase[kw]
    color = phase.get('color', 'C' + str(i % 10))
    if kw in ori_deg_kws and phase[kw] == 'random':
        x_plt = [0, 360]
        y_plt = [1 / 360, 1 / 360]
        plt.plot(x_plt, y_plt, color=color, ls=':')
        ymax = 1 / 360

    elif kw in ori_rad_kws and phase[kw] == 'random':
        x_plt = [0, 2 * np.pi]
        y_plt = [0.5 / np.pi, 0.5 / np.pi]
        plt.plot(x_plt, y_plt, color=color, ls=':')
        ymax = y_plt[0]

    elif phase[kw] == 'random':
        e_str = 'Cannot create PDF for random setting'
        e_str += ' of keyword <' + str(kw) + '>'
        raise NotImplementedError(e_str)

    elif kw == 'orientation':
        ct = inp_dist[0][0]
        st = inp_dist[1][0]
        inp_deg = np.rad2deg(np.arctan2(st, ct))
        plt.plot([inp_deg, inp_deg], [0, 1e12], color=color, ls=':')

    elif isinstance(inp_dist, list):
        for j, dist in enumerate(inp_dist):
            try:
                lb = dist.ppf(1e-3)
                ub = dist.ppf(1 - 1e-3)
                x_plt = np.linspace(lb, ub, 51)
                y_plt = dist.pdf(x_plt)
                ymax = max(ymax, np.max(y_plt))
            except AttributeError:
                x_plt = [dist, dist]
                y_plt = [0, 1e12]
            m = styles_vec[j % len(styles_vec)]
            plt.plot(x_plt, y_plt, color=color, ls=':', marker=m)

    else:
        try:
            lb = inp_dist.ppf(1e-3)
            ub = inp_dist.ppf(1 - 1e-3)
            x_plt = np.linspace(lb, ub, 51)
            y_plt = inp_dist.pdf(x_plt)
            ymax = np.max(y_plt)
        except AttributeError:
            x_plt = [inp_dist, inp_dist]
            y_plt = [0, 1e12]

        plt.plot(x_plt, y_plt, color=color, ls=':')
    return ymax


def _plot_inp_cdf(kw, i, phase):
    quants = np.linspace(0, 1, 501)[1:-1]
    quants_lr = np.linspace(0, 1, 21)[1:-1]

    inp_dist = phase[kw]
    color = phase.get('color', 'C' + str(i % 10))
    if kw in ori_deg_kws and phase[kw] == 'random':
        x_plt = [0, 360]
        y_plt = [0, 1]
        plt.plot(x_plt, y_plt, color=color, ls=':')
    elif kw in ori_rad_kws and phase[kw] == 'random':
        x_plt = [0, 2 * np.pi]
        y_plt = [0, 1]
        plt.plot(x_plt, y_plt, color=color, ls=':')

    elif phase[kw] == 'random':
        e_str = 'Cannot create CDF for random setting'
        e_str += ' of keyword <' + str(kw) + '>'
        raise NotImplementedError(e_str)

    elif kw == 'orientation':
        ct = inp_dist[0][0]
        st = inp_dist[1][0]
        inp_deg = np.rad2deg(np.arctan2(st, ct))
        plt.plot([inp_deg, inp_deg], [0, 1], color=color, ls=':')

    elif isinstance(inp_dist, list):
        for j, dist in enumerate(inp_dist):
            try:
                x_plt = dist.ppf(quants)
                y_plt = quants

                x_plt_lr = dist.ppf(quants_lr)
                y_plt_lr = quants_lr
            except AttributeError:
                x_plt = np.full(11, dist)
                y_plt = np.linspace(0, 1, 11)

                x_plt_lr = x_plt
                y_plt_lr = y_plt
            m = styles_vec[j % len(styles_vec)]
            plt.plot(x_plt, y_plt, color=color, ls=':')
            plt.plot(x_plt_lr, y_plt_lr, color=color, marker=m)

    else:
        try:
            x_plt = inp_dist.ppf(quants)
            y_plt = quants
        except AttributeError:
            x_plt = [inp_dist, inp_dist]
            y_plt = [0, 1]

        plt.plot(x_plt, y_plt, color=color, ls=':')


def _plot_out_pdf(kw, i, phase, comp_phase):
    line_colors = []
    line_labels = []
    ymax = 0

    inp_dist = phase[kw]
    comp_vals = np.array([v for v in comp_phase[kw] if v is not None])
    color = phase.get('color', 'C' + str(i % 10))
    name = str(phase.get('name', 'Material ' + str(i + 1)))
    if kw == 'orientation':
        comp_ct = comp_vals[:, 0, 0]
        comp_st = comp_vals[:, 1, 0]
        comp_vals = np.rad2deg(np.arctan2(comp_st, comp_ct))

        ys, xbs, _ = plt.hist(comp_vals, density=True, histtype='step',
                              color=color)
        ymax = max(ys)

        line_colors.append(Line2D([0], [0], color=color))
        line_labels.append(name)

    elif isinstance(inp_dist, list):
        for j, vals in enumerate(comp_vals):
            ys, xbs, _ = plt.hist(vals, density=True, histtype='step',
                                  color=color)
            ymax = max(ymax, np.max(ys))
            xs = 0.5 * (xbs[:-1] + xbs[1:])

            m = styles_vec[j % len(styles_vec)]
            plt.plot(xs, ys, color=color, marker=m)

            line_colors.append(Line2D([0], [0], color=color, marker=m))
            line_labels.append(name + '[' + str(j) + ']')

    else:
        ys, xbs, _ = plt.hist(comp_vals, bins=20, density=True,
                              histtype='step', color=color)
        ymax = max(ymax, np.max(ys))

        line_colors.append(Line2D([0], [0], color=color))
        line_labels.append(name)

    return ymax, line_colors, line_labels


def _plot_out_cdf(kw, i, phase, comp_phase):
    quants = np.linspace(0, 1, 501)

    line_colors = []
    line_labels = []

    inp_dist = phase[kw]
    comp_vals = np.array([v for v in comp_phase[kw] if v is not None])
    color = phase.get('color', 'C' + str(i % 10))
    name = str(phase.get('name', 'Material ' + str(i + 1)))
    if kw == 'orientation':
        comp_ct = comp_vals[:, 0, 0]
        comp_st = comp_vals[:, 1, 0]
        comp_vals = np.rad2deg(np.arctan2(comp_st, comp_ct))

        x_plt = np.quantile(comp_vals, quants)
        y_plt = quants

        plt.plot(x_plt, y_plt, color=color)
        line_colors.append(Line2D([0], [0], color=color))
        line_labels.append(name)

    elif isinstance(inp_dist, list):
        for j, vals in enumerate(comp_vals):

            x_plt = np.quantile(vals, quants)
            y_plt = quants

            m = styles_vec[j % len(styles_vec)]
            plt.plot(x_plt, y_plt, color=color, marker=m)

            line_colors.append(Line2D([0], [0], color=color, marker=m))
            line_labels.append(name + '[' + str(j) + ']')

    else:
        x_plt = np.quantile(comp_vals, quants)
        y_plt = quants

        plt.plot(x_plt, y_plt, color=color)
        line_colors.append(Line2D([0], [0], color=color))
        line_labels.append(name)

    return line_colors, line_labels


# --------------------------------------------------------------------------- #
#                                                                             #
# Maximum Likelihood Estimators for Phases                                    #
#                                                                             #
# --------------------------------------------------------------------------- #
def mle_phases(seeds, phases, poly_mesh=None, verif_mask=None):
    """Get maximum likelihood estimators (MLEs) for phases

    This function finds distributions in the list of phases and computes the
    MLE parameters for those distributions. The returned value is a list of
    phases with the same length and dictionary keywords, except the
    distributions are replaced with MLE distributions (based on the seeds).
    Constant values are replaced with the mean of the seed values.

    Note that the directional statistics are not used - so the results for
    orientation angles and matrices are unreliable.

    Also note that SciPy currently does not support MLEs for discrete random
    variables. Any discrete distributions will be given a histogram output.

    .. note::

        Directional statistics are not used and as such the results for
        orientation angles and matrices are unreliable. The only exception
        is normally distributed orientation angles.

    Args:
        seeds (SeedList): List of seeds.
        phases (list): List of input phase dictionaries.
        poly_mesh (PolyMesh): *(optional)* Polygonal/polyhedral mesh.
        verif_mask (list or numpy.ndarray): *(optional)* Mask for which
            seeds to include in the MLE parameter calculation. Default is
            True for all seeds.
    """
    circ_highs = {'angle': 360, 'angle_deg': 360, 'angle_rad': 2 * np.pi}

    if verif_mask is None:
        verif_mask = np.full(len(seeds), True)

    # Get geometry of seeds
    comp_phases = _phase_values(seeds, phases, poly_mesh, verif_mask)

    # Get MLEs
    param_phases = []
    for comp_phase, phase in zip(comp_phases, phases):
        param_phase = {}
        for kw in comp_phase:
            comp_vals = np.array([v for v in comp_phase[kw] if v is not None])
            inp_dist = phase[kw]
            can_circ = 'angle' in kw and hasattr(inp_dist, 'dist')
            if can_circ:
                can_circ &= inp_dist.dist.name == 'norm'

            if can_circ:
                high = circ_highs[kw]
                circ_mean = scipy.stats.circmean(comp_vals, high=high)
                circ_scl = scipy.stats.circstd(comp_vals, high=high)
                dist = scipy.stats.norm(loc=circ_mean, scale=circ_scl)
            else:
                dist = _mle_dist(comp_vals, inp_dist)
            param_phase[kw] = dist
        param_phases.append(param_phase)
    return param_phases


def write_mle_phases(inp_phases, out_phases, filename='mles.txt'):
    """Write MLE parameters in a table

    This function writes out a text file containing the input parameters and
    maximum likelihood estimators (MLEs) for the outputs.

    Args:
        inp_phases (list): List of input phase dictionaries.
        out_phases (list): List of output phase dictionaries.
        filename (str): *(optional)* Filename of the output table.
            Defaults to ``mles.txt``.

    Returns:
        none, writes file.

    """
    # Create table rows as dictionaries
    rows_dict = []

    assert len(inp_phases) == len(out_phases)
    for i in range(len(inp_phases)):
        inp_phase = inp_phases[i]
        out_phase = out_phases[i]

        name = inp_phase.get('name', 'Material ' + str(i + 1))

        kws = set(out_phase.keys()) - set(_misc.gen_kws)
        for kw in sorted(kws):
            inp_dist = inp_phase[kw]
            out_dist = out_phase[kw]

            if kw == 'orientation':
                row_dict = {'i': i, 'name': name, 'kw': kw}
                rows_dict.append(row_dict)
                continue
            if isinstance(inp_dist, list):
                for j in range(len(inp_dist)):
                    row_dict = {'i': i, 'name': name, 'kw': kw + '[' + j + ']'}
                    inp_dict = _dist_dict(inp_dist[j])
                    out_dict = _dist_dict(out_dist[j])
                    for key in inp_dict:
                        row_dict[key + '_inp'] = inp_dict[key]
                        row_dict[key + '_out'] = out_dict[key]
                    rows_dict.append(row_dict)
            else:
                row_dict = {'i': i, 'name': name, 'kw': kw}
                inp_dict = _dist_dict(inp_dist)
                out_dict = _dist_dict(out_dist)
                for key in inp_dict:
                    row_dict[key + '_inp'] = inp_dict[key]
                    row_dict[key + '_out'] = out_dict[key]
                rows_dict.append(row_dict)

    # Determine headings / order of dictionaries
    hdg_kws = set([kw for row in rows_dict for kw in row.keys()])
    init_kws = ['i', 'name', 'kw']
    loc_scale_kws = ['loc_inp', 'loc_out', 'scale_inp', 'scale_out']

    all_kws = [kw for kw in init_kws]
    for kw in sorted(list(hdg_kws)):
        if kw in init_kws or kw in loc_scale_kws:
            continue
        all_kws.append(kw)

    for kw in loc_scale_kws:
        if kw in hdg_kws:
            all_kws.append(kw)

    # create list rows
    rows_list = [[d.get(kw, '') for kw in all_kws] for d in rows_dict]

    # create headers
    hdr = _mle_hdr(all_kws)
    table = hdr + rows_list

    rjust = [i for i, kw in enumerate(all_kws) if kw not in ['name', 'kw']]

    # Save table
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(filename, 'w') as file:
        file.write(_fixed_width_table(table, rjust))


def _mle_hdr(all_kws):
    hdr1 = []
    hdr2 = []
    for kw in all_kws:
        if kw == 'i':
            h1 = ''
            h2 = '#'
        elif kw == 'name':
            h1 = ''
            h2 = 'Name'
        elif kw == 'kw':
            h1 = ''
            h2 = 'Parameter'
        elif kw.endswith('_inp'):
            h1 = 'Input'
            h2 = kw.rstrip('_inp')
        elif kw.endswith('_out'):
            h1 = 'Output'
            h2 = kw.rstrip('_out')
        else:
            raise ValueError('Cannot creating heading for keyword ' + str(kw))
        hdr1.append(h1)
        hdr2.append(h2)
    return [hdr1, hdr2]


# --------------------------------------------------------------------------- #
#                                                                             #
# Phase Error Statistics                                                      #
#                                                                             #
# --------------------------------------------------------------------------- #
def error_stats(fit_seeds, seeds, phases, poly_mesh=None, verif_mask=None):
    """Error statistics for seeds

    This function creates a dictionary of error statistics for each of the
    input distributions in the phases.

    Args:
        fit_seeds (SeedList): List of seeds of best fit.
        seeds (SeedList): List of seeds.
        phases (list): List of input phase dictionaries.
        poly_mesh (PolyMesh): *(optional)* Polygonal/polyhedral mesh.
        verif_mask (list or numpy.ndarray): *(optional)* Mask for seeds to
            be included in the analysis. Defaults to all True.

    Returns:
        list: List with the same size and dictionary keywords as phases,
        but with error statistics dictionaries in each entry.

    """

    if verif_mask is None:
        verif_mask = np.full(len(seeds), True)

    # Organize the geometry values
    init_phases = _phase_values(seeds, phases, verif_mask=verif_mask)
    outp_phases = _phase_values(fit_seeds, phases, poly_mesh, verif_mask)

    err_phases = []
    for i in range(len(phases)):
        i_phase = init_phases[i]
        o_phase = outp_phases[i]
        phase = phases[i]
        for kw in phase:
            if kw in ('angle', 'angle_deg') and phase[kw] == 'random':
                phase[kw] = scipy.stats.uniform(loc=0, scale=360)
            if kw == 'angle_rad':
                phase[kw] = scipy.stats.uniform(loc=0, scale=2 * np.pi)

        err_io = {kw: _kw_errs(i_phase[kw], o_phase[kw]) for kw in i_phase}
        err_po = {kw: _kw_stats(phase[kw], o_phase[kw]) for kw in o_phase}

        err_phase = {}
        for kw in i_phase:
            if kw == 'orientation':
                err_phase[kw] = {}
                continue
            val = err_io[kw].copy()
            val.update(err_po[kw])
            err_phase[kw] = val

        err_phases.append(err_phase)
    return err_phases


def _kw_errs(y_exp, y_act):
    if np.array(y_exp).ndim > 1:
        return [_kw_errs(*tup) for tup in zip(y_exp, y_act)]

    errs = {}

    mask = np.array([y_a is not None for y_a in y_act])
    if not np.any(mask):
        return errs

    y_expect = np.array(y_exp)[mask]
    y_actual = np.array([y_a for y_a in y_act if y_a is not None])

    r = y_actual - y_expect

    # MAE
    mae = np.mean(np.abs(r))
    errs['mae'] = mae

    # MSE
    mse = np.mean(r * r)
    errs['mse'] = mse

    # RMSE
    rmse = np.sqrt(mse)
    errs['rmse'] = rmse

    # R^2
    coeff_det = _r2(y_actual, y_expect)
    errs['R^2'] = coeff_det

    # Max Error
    inf_norm = np.max(np.abs(r))
    errs['inf_norm'] = inf_norm

    return errs


def _r2(y_act, y_exp):
    r = y_act - y_exp
    mse = np.mean(r * r)
    y_bar = np.mean(y_act)
    r_ybar = y_act - y_bar
    mse_baseline = np.mean(r_ybar * r_ybar)

    coeff_det = 1 - (mse / mse_baseline)
    return coeff_det


def _kw_stats(dist_exp, y_act):
    if isinstance(dist_exp, list):
        return [_kw_stats(*tup) for tup in zip(dist_exp, y_act)]

    stats = {}
    y_actual = np.array([y_a for y_a in y_act if y_a is not None])
    if len(y_actual) == 0:
        return stats

    y_pred = _safe_rvs(dist_exp, 5000)

    # Wasserstein Distance
    wass = scipy.stats.wasserstein_distance(y_actual, y_pred)
    stats['wasserstein_distance'] = wass

    # Energy Distance
    e_dist = scipy.stats.energy_distance(y_actual, y_pred)
    stats['energy_distance'] = e_dist

    # K-S Test
    if hasattr(dist_exp, 'cdf'):
        cdf_func = dist_exp.cdf
    else:
        def cdf_func(x):
            return (x > dist_exp).astype('float')

    ks_stat, ks_p = scipy.stats.kstest(y_actual, cdf_func)
    stats['ks_statistic'] = ks_stat
    stats['ks_p_value'] = ks_p

    # T-test
    t_stat, t_p = scipy.stats.ttest_ind(y_actual, y_pred)
    stats['t_stat'] = t_stat
    stats['t_p_value'] = t_p

    return stats


def write_error_stats(errs, phases, filename='error_stats.txt'):
    """Write error statistics to file

    This function takes previously computed error statistics and writes them
    to a human-readable text file.

    Args:
        errs (list): List of error statistics for each input phase parameter.
            Organized the same as ``phases``.
        phases (list): List of input phases. See :ref:`phase_dict_guide` for
            more details.
        filename (str): *(optional)* The name of the file to contain the
            error statistics. Defaults to ``error_stats.txt``.
    """
    # Create table rows as dictionaries
    rows_dict = []

    assert len(errs) == len(phases)
    for i in range(len(errs)):
        err_dict = errs[i]
        phase = phases[i]

        name = phase.get('name', 'Material ' + str(i + 1))

        kws = set(err_dict.keys()) - set(_misc.gen_kws)
        for kw in kws:
            err_metrics = err_dict[kw]
            inp_dist = phase[kw]
            if isinstance(inp_dist, list):
                for j in range(len(inp_dist)):
                    row_dict = {'i': i, 'name': name, 'kw': kw + '[' + j + ']'}
                    row_dict.update(err_metrics[j])
                    rows_dict.append(row_dict)
            else:
                row_dict = {'i': i, 'name': name, 'kw': kw}
                row_dict.update(err_metrics)
                rows_dict.append(row_dict)

    # Determine headings / order of dictionaries
    hdg_kws = set([kw for row in rows_dict for kw in row.keys()])
    init_kws = ['i', 'name', 'kw']

    all_kws = [kw for kw in init_kws]
    for kw in sorted(list(hdg_kws)):
        if kw not in init_kws:
            all_kws.append(kw)

    # create list rows
    rows_list = [[d.get(kw, '') for kw in all_kws] for d in rows_dict]

    # create headers
    hdr1 = []
    hdr2 = []

    hd1 = {'i': '', 'name': '', 'kw': '',
           'energy_distance': 'Energy',
           'inf_norm': 'Maximum',
           'ks_p_value': 'K-S',
           'ks_statistic': 'K-S',
           'mae': 'Mean Average',
           'mse': 'Mean Squared',
           'rmse': 'Root Mean',
           't_p_value': 't-test',
           't_stat': 't-test',
           'wasserstein_distance': 'Wasserstein'}

    hd2 = {'i': '#', 'name': 'Name', 'kw': 'Parameter',
           'energy_distance': 'Distance',
           'inf_norm': 'Absolute Error',
           'ks_p_value': 'p-value',
           'ks_statistic': 'Statistic',
           'mae': 'Error',
           'mse': 'Error',
           'rmse': 'Squared Error',
           't_p_value': 'p-value',
           't_stat': 'Statistic',
           'wasserstein_distance': 'Distance'}
    for kw in all_kws:
        hdr1.append(hd1.get(kw, ''))
        hdr2.append(hd2.get(kw, kw))

    table = [hdr1, hdr2] + rows_list

    rjust = [i for i, kw in enumerate(all_kws) if kw not in ['name', 'kw']]

    # Save table
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(filename, 'w') as file:
        file.write(_fixed_width_table(table, rjust))


# --------------------------------------------------------------------------- #
#                                                                             #
# Private Functions                                                           #
#                                                                             #
# --------------------------------------------------------------------------- #
def _safe_mean(x):
    try:
        mu = x.mean()
    except AttributeError:
        mu = x
    return mu


def _safe_rvs(x, size=1):
    if isinstance(x, list):
        return np.array([_safe_rvs(xi, size) for xi in x]).T

    try:
        samples = x.rvs(size=size)
    except AttributeError:
        samples = np.full(size, x)
    return samples


def _phase_values(seeds, phases, poly_mesh=None, verif_mask=None):
    """Takes the properties of the seeds and organizes them like the phases

    """
    if verif_mask is None:
        verif_mask = np.full(len(seeds), True)

    if poly_mesh is not None:
        cell_vols = np.array(poly_mesh.volumes)
        vols = []
        for i in range(len(seeds)):
            mask = np.isclose(poly_mesh.seed_numbers, i)
            vols.append(np.sum(cell_vols[mask]))

    comp_phases = []
    for i, phase in enumerate(phases):
        comp_phase = {}
        phase_seeds = [s for f, s in zip(verif_mask, seeds)
                       if s.phase == i and f]
        for kw in phase:
            if 'rot_seq' in kw:
                continue
            if kw in _misc.gen_kws:
                continue

            try:
                if kw in ('area', 'volume') and (poly_mesh is not None):
                    vals = [v for v, s, f in zip(vols, seeds, verif_mask)
                            if s.phase == i and f]
                else:
                    vals = [_getattr(s.geometry, kw) for s in phase_seeds]

                comp_phase[kw] = vals
            except AttributeError:
                pass

        comp_phases.append(comp_phase)
    return comp_phases


def _getattr(inst, kw):
    try:
        a = getattr(inst, kw)
    except AttributeError:
        a = None
    return a


def _mle_dist(values, dist):
    if isinstance(dist, list):
        return [_mle_dist(*tup) for tup in zip(values, dist)]

    if not (hasattr(dist, 'dist') or isinstance(dist, hist_class)):
        return np.mean(values)

    if isinstance(dist, hist_class) or dist.dist.name in discr_rvs:
        hist = np.histogram(values, bins='auto', density=True)
        return scipy.stats.rv_histogram(hist)

    args = dist.dist.fit(values)
    kwargs = {'loc': args[-2], 'scale': args[-1]}

    name = dist.dist.name
    return scipy.stats.__dict__[name](*args[:-2], **kwargs)


def _dist_dict(dist):
    if not (hasattr(dist, 'dist') or isinstance(dist, hist_class)):
        return {'loc': dist}

    if isinstance(dist, hist_class) or dist.dist.name in discr_rvs:
        return {}

    args = dist.args
    kwds = dist.kwds
    shape_vals, loc, scale = dist.dist._parse_args(*args, **kwds)

    param_dict = {'loc': loc, 'scale': scale}
    if len(shape_vals) == 0:
        return param_dict

    shape_kwds = dist.dist.shapes.split(',')
    for val, kwd in zip(shape_vals, shape_kwds):
        param_dict[kwd.strip().lower()] = val
    return param_dict


def _fixed_width_table(rows, rjust_cols=[]):
    # heading row
    hr = None
    for i, row in enumerate(rows):
        if all([isinstance(col, str) for col in row]):
            hr = i

    # Convert rows to strings
    str_rows = []
    for row in rows:
        str_row = []
        for col in row:
            if isinstance(col, float):
                if col == 0 or np.abs(np.log10(np.abs(col))) < 3:
                    col_str = '{:f}'.format(col)
                else:
                    col_str = '{:e}'.format(col)
            else:
                col_str = str(col)
            str_row.append(col_str)
        str_rows.append(str_row)

    # Determine column widths
    col_widths = [max([len(s) for s in col]) for col in zip(*str_rows)]

    # Add divider
    if hr is not None:
        str_rows.insert(hr + 1, [w * '-' for w in col_widths])

    # Create table
    table = ''
    for i, row in enumerate(str_rows):
        row_strings = []
        for j, col in enumerate(row):
            w = col_widths[j]
            if j in rjust_cols and i > hr + 1:
                row_strings.append(col.rjust(w))
            else:
                row_strings.append(col.ljust(w))
        table += ' '.join(row_strings) + '\n'
    return table
