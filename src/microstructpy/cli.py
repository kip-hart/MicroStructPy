"""Command Line Interface.

This module contains the command line interface (CLI) for MicroStructPy.
The CLI primarily reads XML input files and creates a microstructure according
to those inputs. It can also run demo input files.

"""


from __future__ import division
from __future__ import print_function

import argparse
import collections
import glob
import os
import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import xmltodict
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D

from microstructpy import _misc
from microstructpy import geometry
from microstructpy import seeding
from microstructpy import verification
from microstructpy.meshing import PolyMesh
from microstructpy.meshing import TriMesh
from microstructpy.meshing.trimesh import facet_check

__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# Main Function                                                               #
#                                                                             #
# --------------------------------------------------------------------------- #
def main():
    """CLI calling function"""
    parser_descr = 'Create a microstructure and unstructured grid using the'
    parser_descr += ' MicroStructPy package. This is the command line'
    parser_descr += ' interface (CLI) for the package. It reads either demo'
    parser_descr += ' input files or user-created input files and generates:'
    parser_descr += ' 1 A list of seed geometries,'
    parser_descr += ' 2 A polygon/polyhedron mesh, and'
    parser_descr += ' 3. A triangle/tetrahedron mesh.'
    parser_descr += 'Additional outputs, such as plots, can be specified in'
    parser_descr += ' the input file. Please read the input file section of'
    parser_descr += ' the MicroStructPy documentation for more details.'
    parser = argparse.ArgumentParser(description=parser_descr)

    user_descr = 'user-created XML input files, separated by spaces'
    parser.add_argument('user_files', metavar='u', type=str, nargs='*',
                        default=[], help=user_descr)

    demo_descr = 'demonstration names, separated by spaces '
    demo_descr += ' (see documentation for list of available names)'
    parser.add_argument('--demo', dest='demo_files', metavar='d', type=str,
                        nargs='*', default=[], help=demo_descr)

    args = parser.parse_args()

    # run user-generated files
    user_files = [f for fnames in args.user_files for f in glob.glob(fnames)]
    for filename in set(user_files):
        run_file(filename)

    # run demo files
    cli_path = os.path.dirname(__file__)
    demo_path = os.path.normpath(os.path.join(cli_path, 'examples'))
    demo_files = args.demo_files

    if len(demo_files) == 1 and demo_files[0].lower().strip() == 'all':
        xml_files = os.path.normpath(os.path.join(demo_path, '*.xml'))
        xml_demos = [os.path.basename(f) for f in glob.glob(xml_files)]

        py_files = os.path.normpath(os.path.join(demo_path, '*.py'))
        py_demos = [os.path.basename(f) for f in glob.glob(py_files)]

        demo_files = xml_demos + py_demos

    for demo_basename in demo_files:
        ex_file = os.path.join(demo_path, demo_basename)
        ext = demo_basename.split('.')[-1]
        if os.path.exists(ex_file) and (ext in ['py', 'xml']):
            print('Running ' + repr(demo_basename))

            local_filename = os.path.join('./', demo_basename)
            needed_files = _misc.demo_needs.get(demo_basename, [])

            for basename in [demo_basename] + needed_files:
                filename = os.path.join(demo_path, basename)
                try:
                    shutil.move(filename, './')
                except shutil.Error:
                    pass
                else:
                    shutil.copy(basename, filename)

            if ext == 'py':
                subprocess.call(['python', local_filename])
            elif ext == 'xml':
                run_file(local_filename)
            else:
                pass

        else:
            p_str = 'Skipping file ' + repr(demo_basename) + ', since '
            p_str += 'it is not a valid example filename.'
            print(p_str)


def run_file(filename):
    """Run an input file

    This function reads an input file and runs it through the standard
    workflow.

    Args:
        filename (str): The name of an XML input file.

    """
    in_data = read_input(filename)
    phases = in_data['material']
    domain = in_data['domain']
    kwargs = in_data['settings']
    run(phases, domain, **kwargs)


def read_input(filename):
    """Convert input file to dictionary

    This function reads an input file and parses it into a dictionary.

    Args:
        filename (str): The name of an XML input file.

    Returns:
        collections.OrderedDict: Dictionary of run inputs.

    """
    # Read in the file
    file_path = os.path.dirname(filename)

    with open(filename, 'r') as file:
        file_dict = xmltodict.parse(file.read())

    assert 'input' in file_dict, 'Root <input> not found in input file.'
    in_data = dict_convert(file_dict['input'], file_path)

    k_args = ('material', 'domain')
    for arg in k_args:
        assert arg in in_data

    # Get domain
    domain_data = in_data['domain']
    domain_shape = domain_data['shape']
    domain_kwargs = {k: v for k, v in domain_data.items() if k != 'shape'}
    domain = geometry.factory(domain_shape, **domain_kwargs)
    in_data['domain'] = domain

    # Default settings
    kwargs = in_data.get('settings', {})
    run_dir = kwargs.get('directory', '.')
    if not os.path.isabs(run_dir):
        rel_path = os.path.join(file_path, run_dir)
        kwargs['directory'] = os.path.expanduser(os.path.normpath(rel_path))
    in_data['settings'] = kwargs
    return in_data


def run(phases, domain, verbose=False, restart=True, directory='.',
        filetypes={}, rng_seeds={}, plot_axes=True, rtol='fit',
        mesh_max_volume=float('inf'), mesh_min_angle=0,
        mesh_max_edge_length=float('inf'), verify=False, color_by='material',
        colormap='viridis', seeds_kwargs={}, poly_kwargs={}, tri_kwargs={}):
    r"""Run MicroStructPy

    This is the primary run function for the package. It performs these steps:

        * Create a list of un-positioned seeds
        * Position seeds in domain
        * Create a polygon mesh from the seeds
        * Create a triangle mesh from the polygon mesh
        * (optional) Perform mesh verification

    Args:
        phases (list or dict): Single phase dictionary or list of multiple
            phase dictionaries. See :ref:`phase_dict_guide` for more details.
        domain (from :mod:`microstructpy.geometry`): The geometry of the
            domain.
        verbose (bool): *(optional)* Option to run in verbose mode.
            Prints status updates to the terminal. Defaults to False.
        restart (bool): *(optional)* Option to run in restart mode.
            Saves caches at the end of each step and reads caches to restart
            the analysis. Defaults to True.
        directory (str): *(optional)* File path where outputs will be saved.
            This path can either be relative to the current directory,
            or an absolute path. Defaults to the current working directory.
        filetypes (dict): *(optional)* Filetypes for the output files.
            A dictionary containing many of the possible file types is::

                filetypes = {'seeds': 'txt',
                             'seeds_plot': ['eps',
                                            'pdf',
                                            'png',
                                            'svg'],
                             'poly': ['txt', 'ply', 'vtk'],
                             'poly_plot': 'png',
                             'tri': ['txt', 'abaqus', 'vtk'],
                             'tri_plot': ['png', 'pdf'],
                             'verify_plot': 'pdf'
                             }

            If an entry is not included in the dictionary, then that output
            is not saved. Default is an empty dictionary. If *restart* is
            True, then 'txt' is added to the 'seeds', 'poly', and 'tri' fields.
        rng_seeds (dict): *(optional)* The random number generator (RNG) seeds.
            The dictionary values should all be non-negative integers.
            Specifically, RNG seeds should be convertible to NumPy `uint32`_.
            An example dictionary is::

                rng_seeds = {'fraction': 0,
                             'phase': 134092,
                             'position': 1,
                             'size': 95,
                             'aspect_ratio': 2,
                             'orienation': 2
                             }

            If a seed is not specified, the default value is 0.
        rtol (float or str): *(optional)* The relative overlap tolerance
            between seeds. This parameter should be between 0 and 1.
            The condition for two circles to overlap is:

            .. math::

                || x_2 - x_1 || + \text{rtol} \min(r_1, r_2) < r_1 + r_2

            The default value is ``'fit'``, which uses the mean and variance
            of the size distribution to estimate a value for rtol.
        mesh_max_volume (float): *(optional)* The maximum volume (area in 2D)
            of a mesh cell in the triangular mesh. Default is infinity,
            which turns off the maximum volume quality setting.
            Value should be stritly positive.
        mesh_min_angle (float): *(optional)* The minimum interior angle,
            in degrees,  of a cell in the triangular mesh. For 3D meshes,
            this is the dihedral angle between faces of the tetrahedron.
            Defaults to 0, which turns off the angle quality constraint.
            Value should be in the range 0-60.
        mesh_max_edge_length (float): *(optional)* The maximum edge length of
            elements along grain boundaries. Currently only supported in 2D.
        plot_axes (bool): *(optional)* Option to show the axes in output plots.
            When False, the plots are saved without axes and very tight
            borders. Defaults to True.
        verify (bool): *(optional)* Option to verify the output mesh against
            the input phases. Defaults to False.
        color_by (str): *(optional)* {'material' | 'seed number' |
            'material number'} Option to choose how the polygons/polyhedra
            are colored. Defaults to 'material'.
        colormap (str): *(optional)* Name of the matplotlib colormap to color
            the seeds. Ignored if `color_by='material'`. Defaults to 'viridis',
            the standard matplotlib colormap.
            See `Choosing Colormaps in Matplotlib`_ for more details.
        seed_kwargs (dict): additional keyword arguments that will be passed to
            :meth:`.SeedList.plot`.
        poly_kwargs (dict): Additional keyword arguments that will be passed to
            :meth:`.PolyMesh.plot_facets` in 2D and
            :meth:`.PolyMesh.plot` in 3D.
        tri_kwargs (dict): Additional keyword arguments that will be passed to
            :meth:`.TriMesh.plot`.

    .. _`Specifying Colors`: https://matplotlib.org/users/colors.html
    .. _`Choosing Colormaps in Matplotlib`: https://matplotlib.org/tutorials/colors/colormaps.html
    .. _`uint32`: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    """  # NOQA: E501
    # ----------------------------------------------------------------------- #
    # Condition Inputs                                                        #
    # ----------------------------------------------------------------------- #
    # Phases
    # ------
    if type(phases) is not list:
        phases = [phases]

    # Settings
    # --------
    # filetypes
    if restart:
        for kw in ('seeds', 'poly', 'tri'):
            if kw not in filetypes:
                filetypes[kw] = 'txt'
            elif isinstance(filetypes[kw], list):
                if 'txt' not in filetypes[kw]:
                    filetypes[kw].append('txt')
            else:
                filetypes[kw] = [filetypes[kw], 'txt']

    if verbose:
        print('Running MicroStructPy in verbose mode.')

    # ----------------------------------------------------------------------- #
    # Create Directory                                                        #
    # ----------------------------------------------------------------------- #
    if not os.path.exists(directory):
        if verbose:
            print('Creating run directory, ' + directory)
        os.makedirs(directory)

    # ----------------------------------------------------------------------- #
    # Create Seeds                                                            #
    # ----------------------------------------------------------------------- #
    seed_basename = 'seeds.txt'
    seed_filename = os.path.join(directory, seed_basename)
    if restart and os.path.exists(seed_filename):
        # Read seeds from file
        if verbose:
            print('Reading seeds from file.')
            print('Seed file: ' + seed_filename)
        seeds = seeding.SeedList.from_file(seed_filename)
        seeds_created = False
    else:
        seeds_created = True
        # Initialize seeds
        if verbose:
            print('Creating un-positioned list of seeds.')

        seeds = _unpositioned_seeds(phases, domain, rng_seeds)

        if verbose:
            print('There are ' + str(len(seeds)) + ' seeds.')
            print('Positioning seeds in domain.')

        kw = 'position'
        rng_seed = rng_seeds.get(kw, 0)
        pos_dists = {i: p[kw] for i, p in enumerate(phases) if kw in p}
        seeds.position(domain, pos_dists, rng_seed, rtol=rtol, verbose=verbose)

        # Write seeds
        if 'seeds' in filetypes:
            seeds.write(seed_filename)

    # ----------------------------------------------------------------------- #
    # Plot Seeds                                                              #
    # ----------------------------------------------------------------------- #
    plot_types = filetypes.get('seeds_plot', ['png'])
    if not plot_types:
        plot_types = []
    elif type(plot_types) is not list:
        plot_types = [plot_types]

    plot_files = []
    for ext in plot_types:
        fname = os.path.join(directory, 'seeds.' + str(ext))
        if seeds_created or not os.path.exists(fname):
            plot_files.append(fname)

    if plot_files and verbose:
        print('Plotting seeds.')

    plot_seeds(seeds, phases, domain, plot_files, plot_axes, color_by,
               colormap, **seeds_kwargs)

    # ----------------------------------------------------------------------- #
    # Create Polygon Mesh                                                     #
    # ----------------------------------------------------------------------- #
    poly_basename = 'polymesh.txt'
    poly_filename = os.path.join(directory, poly_basename)
    if restart and os.path.exists(poly_filename) and not seeds_created:
        # Read polygon mesh from file
        if verbose:
            print('Reading polygon mesh from file.')
            print('Polygon mesh filename: ' + poly_filename)

        pmesh = PolyMesh.from_file(poly_filename)
        poly_created = False
    else:
        poly_created = True

        # Create polygon mesh from seeds
        if verbose:
            print('Creating polygon mesh.')

        pmesh = PolyMesh.from_seeds(seeds, domain)

    # Write polymesh
    poly_types = filetypes.get('poly', [])
    if type(poly_types) != list:
        poly_types = [poly_types]

    for poly_type in poly_types:
        fname = poly_filename.replace('.txt', '.' + poly_type)
        pmesh.write(fname, poly_type)

    # ----------------------------------------------------------------------- #
    # Plot Polygon Mesh                                                       #
    # ----------------------------------------------------------------------- #
    plot_types = filetypes.get('poly_plot', ['png'])
    if not plot_types:
        plot_types = []
    elif type(plot_types) is not list:
        plot_types = [plot_types]

    plot_files = []
    for ext in plot_types:
        fname = os.path.join(directory, 'polymesh.' + str(ext))
        if poly_created or not os.path.exists(fname):
            plot_files.append(fname)

    if plot_files and verbose:
        print('Plotting polygon mesh.')

    plot_poly(pmesh, phases, plot_files, plot_axes, color_by, colormap,
              **poly_kwargs)

    # ----------------------------------------------------------------------- #
    # Create Triangular Mesh                                                  #
    # ----------------------------------------------------------------------- #
    tri_basename = 'trimesh.txt'
    tri_filename = os.path.join(directory, tri_basename)
    exts = {'abaqus': '.inp', 'txt': '.txt', 'str': '.txt', 'tet/tri': '',
            'vtk': '.vtk'}

    if restart and os.path.exists(tri_filename) and not poly_created:
        # Read triangle mesh
        if verbose:
            print('Reading triangular mesh.')
            print('Triangular mesh filename: ' + tri_filename)

        tmesh = TriMesh.from_file(tri_filename)
        tri_created = False
    else:
        tri_created = True
        # Create triangular mesh
        if verbose:
            print('Creating triangular mesh.')

        tmesh = TriMesh.from_polymesh(pmesh, phases, mesh_min_angle,
                                      mesh_max_volume, mesh_max_edge_length)

    # Write triangular mesh
    tri_types = filetypes.get('tri', [])
    if type(tri_types) != list:
        tri_types = [tri_types]

    for tri_type in tri_types:
        fname = tri_filename.replace('.txt', exts[tri_type])
        tmesh.write(fname, tri_type, seeds, pmesh)

    # ----------------------------------------------------------------------- #
    # Plot Triangular Mesh                                                    #
    # ----------------------------------------------------------------------- #
    plot_types = filetypes.get('tri_plot', ['png'])
    if not plot_types:
        plot_types = []
    elif type(plot_types) is not list:
        plot_types = [plot_types]

    plot_files = []
    for ext in plot_types:
        fname = os.path.join(directory, 'trimesh.' + str(ext))
        if tri_created or not os.path.exists(fname):
            plot_files.append(fname)

    if plot_files and verbose:
        print('Plotting triangular mesh.')

    plot_tri(tmesh, phases, seeds, pmesh, plot_files, plot_axes, color_by,
             colormap, **tri_kwargs)

    # ----------------------------------------------------------------------- #
    # Perform Verification                                                    #
    # ----------------------------------------------------------------------- #
    if not verify:
        if verbose:
            print('Finished running MicroStructPy.')
        return

    if verbose:
        print('Performing verification.')

    # Set, create verification results directory
    ver_dir = os.path.join(directory, 'verification')
    if not os.path.exists(ver_dir):
        os.makedirs(ver_dir)

    # Set plot types
    plottypes = filetypes.get('verify_plot', ['png'])
    if type(plottypes) is not list:
        plottypes = [plottypes]

    # Verify volume fractions
    vol_fracs = verification.volume_fractions(pmesh, len(phases))

    vf_fname = os.path.join(ver_dir, 'volume_fractions.txt')
    verification.write_volume_fractions(vol_fracs, phases, vf_fname)

    fnames = [os.path.join(ver_dir, 'volume_fractions.' + ext) for
              ext in plottypes]
    verification.plot_volume_fractions(vol_fracs, phases, fnames)

    # Screen out certain seeds on the boundaries
    n_dim = domain.n_dim
    default_shape = {2: 'circle', 3: 'sphere'}[n_dim]
    verif_mask = np.full(len(seeds), True)
    for neighbors in pmesh.facet_neighbors:
        if min(neighbors) < 0:
            cell_num = max(neighbors)
            seed_num = pmesh.seed_numbers[cell_num]
            phase_num = pmesh.phase_numbers[cell_num]
            phase = phases[phase_num]
            phase_geom = phase.get('shape', default_shape)
            if phase_geom in ['square', 'rectangle']:
                verif_mask[seed_num] = False

    # Determine seeds of best fit
    fit_seeds = verification.seeds_of_best_fit(seeds, phases, pmesh, tmesh)

    filename = os.path.join(ver_dir, 'fit_seeds.txt')
    fit_seeds.write(filename)

    # Create distribution plots
    verification.plot_distributions(fit_seeds, phases, ver_dir, plottypes,
                                    pmesh, verif_mask)

    # Write Maximum Likelihood Estimates
    mle_phases = verification.mle_phases(fit_seeds, phases, pmesh, verif_mask)

    mle_fname = os.path.join(ver_dir, 'mles.txt')
    verification.write_mle_phases(phases, mle_phases, mle_fname)

    # Write Error Statistics
    error_stats = verification.error_stats(fit_seeds, seeds, phases, pmesh,
                                           verif_mask)

    err_fname = os.path.join(ver_dir, 'err_stats.txt')
    verification.write_error_stats(error_stats, phases, err_fname)

    if verbose:
        print('Finished running MicrostructPy.')


# --------------------------------------------------------------------------- #
#                                                                             #
# Created Unpositioned List of Seeds                                          #
#                                                                             #
# --------------------------------------------------------------------------- #
def _unpositioned_seeds(phases, domain, rng_seeds={}):
    if domain.n_dim == 2:
        dom_vol = domain.area
    else:
        dom_vol = domain.volume
    return seeding.SeedList.from_info(phases, dom_vol, rng_seeds)


# --------------------------------------------------------------------------- #
#                                                                             #
# Plot Seeds                                                                  #
#                                                                             #
# --------------------------------------------------------------------------- #
def plot_seeds(seeds, phases, domain, plot_files=[], plot_axes=True,
               color_by='material', colormap='viridis', **edge_kwargs):
    """Plot seeds

    This function creates formatted plots of a :class:`.SeedList`.

    Args:
        seeds (SeedList): Seed list to plot.
        phases (list): List of phase dictionaries. See :ref:`phase_dict_guide`
            for more details.
        domain (from :mod:`microstructpy.geometry`): Domain geometry.
        plot_files (list): *(optional)* List of files to save the output plot.
            Defaults to saving the plot to ``polymesh.png``.
        plot_axes (bool): *(optional)* Flag to turn the axes on or off.
            True shows the axes, False removes them. Defaults to True.
        color_by (str): *(optional)* {'material' | 'seed number' |
            'material number'} Option to choose how the polygons/polyhedra
            are colored. Defaults to 'material'.
        colormap (str): *(optional)* Name of the matplotlib colormap to color
            the seeds. Ignored if `color_by='material'`. Defaults to 'viridis',
            the standard matplotlib colormap.
            See `Choosing Colormaps in Matplotlib`_ for more details.
        **edge_kwargs: additional keyword arguments that will be passed to
            :meth:`.SeedList.plot`.

    """
    if not plot_files:
        return

    phase_names = []
    given_names = False
    for i, phase in enumerate(phases):
        if 'name' in phase:
            given_names = True
        name = phase.get('name', 'Material ' + str(i + 1))
        phase_names.append(name)

    seed_colors = _seed_colors(seeds, phases, color_by, colormap)
    n_dim = seeds[0].geometry.n_dim

    # Set up axes
    plt.clf()
    plt.close('all')
    fig = plt.figure()
    ax = fig.gca(projection={2: None, 3: Axes3D.name}[n_dim], label='seeds')

    if not plot_axes:
        if n_dim == 2:
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax._axis3don = False
    fig.add_axes(ax)

    # Plot seeds
    edge_kwargs.setdefault('edgecolors', {2: 'k', 3: 'none'}[n_dim])
    seeds.plot(facecolors=seed_colors, **edge_kwargs)

    # Add legend
    custom_seeds = [None for _ in phases]
    for seed in seeds:
        phase_num = seed.phase
        if custom_seeds[phase_num] is None:
            c = _phase_color(phase_num, phases)
            lbl = phase_names[phase_num]
            phase_patch = patches.Patch(fc=c, ec='k', label=lbl)
            custom_seeds[phase_num] = phase_patch

    if given_names and color_by == 'material':
        handles = [h for h in custom_seeds if h is not None]
        ax.legend(handles=handles, loc=4)

    # Set limits
    lims = domain.limits
    if n_dim == 2:
        plt.axis('square')
        plt.xlim(lims[0])
        plt.ylim(lims[1])

    else:
        for x in lims[0]:
            for y in lims[1]:
                for z in lims[2]:
                    ax.plot([x], [y], [z])
        ax.set_aspect('equal')

    # Save plot
    for fname in plot_files:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    plt.close('all')


def _seed_colors(seeds, phases, color_by='material', colormap='viridis'):
    if color_by == 'material':
        return [_phase_color(s.phase, phases) for s in seeds]
    elif color_by == 'seed number':
        n = len(seeds)
        return [_cm_color(i / (n - 1), colormap) for i in range(n)]
    elif color_by == 'material number':
        n = len(phases)
        return [_cm_color(s.phase / (n - 1), colormap) for s in seeds]


def _phase_color(i, phases):
    return phases[i].get('color', 'C' + str(i % 10))


def _cm_color(f, colormap='viridis'):
    return plt.get_cmap(colormap)(f)


# --------------------------------------------------------------------------- #
#                                                                             #
# Plot Polygon                                                                #
#                                                                             #
# --------------------------------------------------------------------------- #
def plot_poly(pmesh, phases, plot_files=['polymesh.png'], plot_axes=True,
              color_by='material', colormap='viridis', **edge_kwargs):
    """Plot polygonal/polyhedral mesh

    This function creates formatted plots of a :class:`.PolyMesh`.

    Args:
        pmesh (PolyMesh): Polygonal mesh to plot.
        phases (list): List of phase dictionaries. See :ref:`phase_dict_guide`
            for more details.
        plot_files (list): *(optional)* List of files to save the output plot.
            Defaults to saving the plot to ``polymesh.png``.
        plot_axes (bool): *(optional)* Flag to turn the axes on or off.
            True shows the axes, False removes them. Defaults to True.
        color_by (str): *(optional)* {'material' | 'seed number' |
            'material number'} Option to choose how the polygons/polyhedra
            are colored. Defaults to 'material'.
        colormap (str): *(optional)* Name of the matplotlib colormap to color
            the seeds. Ignored if `color_by='material'`. Defaults to 'viridis',
            the standard matplotlib colormap.
            See `Choosing Colormaps in Matplotlib`_ for more details.
        **edge_kwargs: Additional keyword arguments that will be passed to
            :meth:`.PolyMesh.plot_facets` in 2D and
            :meth:`.PolyMesh.plot` in 3D.

    """
    if not plot_files:
        return

    n_dim = len(pmesh.points[0])

    phase_colors = []
    phase_names = []
    given_names = False
    for i, phase in enumerate(phases):
        color = phase.get('color', 'C' + str(i % 10))
        if 'name' in phase:
            given_names = True
        name = phase.get('name', 'Material ' + str(i + 1))
        phase_colors.append(color)
        phase_names.append(name)

    # Set up axes
    plt.clf()
    plt.close('all')
    fig = plt.figure()
    ax = fig.gca(projection={2: None, 3: Axes3D.name}[n_dim], label='poly')

    if not plot_axes:
        if n_dim == 2:
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax._axis3don = False

    # Plot polygons
    fcs = _poly_colors(pmesh, phases, color_by, colormap, n_dim)
    if n_dim == 2:
        pmesh.plot(facecolors=fcs)

        edge_color = edge_kwargs.pop('edgecolors', (0, 0, 0, 1))
        facet_colors = []
        for neigh_pair in pmesh.facet_neighbors:
            if facet_check(neigh_pair, pmesh, phases):
                facet_colors.append(edge_color)  # black
            else:
                facet_colors.append('none')

        edge_kwargs.setdefault('capstyle', 'round')
        pmesh.plot_facets(color=facet_colors, **edge_kwargs)
    else:
        edge_kwargs.setdefault('edgecolors', 'k')
        pmesh.plot(facecolors=fcs, **edge_kwargs)

    # add legend
    if given_names:
        custom_seeds = [None for _ in phases]
        for phase_num in pmesh.phase_numbers:
            if custom_seeds[phase_num] is None:
                c = phase_colors[phase_num]
                lbl = phase_names[phase_num]
                phase_patch = patches.Patch(fc=c, ec='k', label=lbl)
                custom_seeds[phase_num] = phase_patch
        handles = [h for h in custom_seeds if h is not None]
        ax.legend(handles=handles, loc=4)

    # format axes
    lims = np.array([np.min(pmesh.points, 0), np.max(pmesh.points, 0)]).T
    if n_dim == 2:
        plt.axis('square')
        plt.xlim(lims[0])
        plt.ylim(lims[1])

    else:
        for x in lims[0]:
            for y in lims[1]:
                for z in lims[2]:
                    ax.plot([x], [y], [z])
        ax.set_aspect('equal')

    # save plot
    for fname in plot_files:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close('all')


def _poly_colors(pmesh, phases, color_by, colormap, n_dim):
    if n_dim == 2:
        if color_by == 'material':
            return [_phase_color(n, phases) for n in pmesh.phase_numbers]
        elif color_by == 'seed number':
            n = max(pmesh.seed_numbers) + 1
            return [_cm_color(s / (n - 1), colormap) for s in
                    pmesh.seed_numbers]
        elif color_by == 'material number':
            n = len(phases)
            return [_cm_color(p / (n - 1), colormap) for p in
                    pmesh.phase_numbers]
    else:
        poly_fcs = []
        for n_pair in pmesh.facet_neighbors:
            if min(n_pair) < 0:
                n_int = max(n_pair)
                if color_by == 'material':
                    phase_num = pmesh.phase_numbers[n_int]
                    color = _phase_color(phase_num, phases)
                elif color_by == 'seed number':
                    n_seed = max(pmesh.seed_numbers) + 1
                    seed_num = pmesh.seed_numbers[n_int]
                    color = _cm_color(seed_num / (n_seed - 1), colormap)
                elif color_by == 'material number':
                    n_phases = len(phases)
                    phase_num = pmesh.phase_numbers[n_int]
                    color = _cm_color(phase_num / (n_phases - 1), colormap)
            else:
                color = 'none'
            poly_fcs.append(color)
        return poly_fcs


# --------------------------------------------------------------------------- #
#                                                                             #
# Plot Triangular Mesh                                                        #
#                                                                             #
# --------------------------------------------------------------------------- #
def plot_tri(tmesh, phases, seeds, pmesh, plot_files=[], plot_axes=True,
             color_by='material', colormap='viridis', **edge_kwargs):
    """Plot seeds

    This function creates formatted plots of a :class:`.TriMesh`.

    Args:
        tmesh (TriMesh): Triangular mesh to plot.
        phases (list): List of phase dictionaries. See :ref:`phase_dict_guide`
            for more details.
        seeds (SeedList): List of seed geometries.
        pmesh (PolyMesh): Polygonal mesh from which ``tmesh`` was generated.
        plot_files (list): *(optional)* List of files to save the output plot.
            Defaults to saving the plot to ``polymesh.png``.
        plot_axes (bool): *(optional)* Flag to turn the axes on or off.
            True shows the axes, False removes them. Defaults to True.
        color_by (str): *(optional)* {'material' | 'seed number' |
            'material number'} Option to choose how the polygons/polyhedra
            are colored. Defaults to 'material'.
        colormap (str): *(optional)* Name of the matplotlib colormap to color
            the seeds. Ignored if `color_by='material'`. Defaults to 'viridis',
            the standard matplotlib colormap.
            See `Choosing Colormaps in Matplotlib`_ for more details.
        **edge_kwargs: Additional keyword arguments that will be passed to
            :meth:`.TriMesh.plot`.

    """
    if not plot_files:
        return

    n_dim = len(tmesh.points[0])

    phase_colors = []
    phase_names = []
    given_names = []
    for i, phase in enumerate(phases):
        color = phase.get('color', 'C' + str(i % 10))
        name = phase.get('name', 'Material ' + str(i + 1))
        phase_colors.append(color)
        phase_names.append(name)
        given_names.append('name' in phase)

    # Set up axes
    plt.clf()
    plt.close('all')
    fig = plt.figure()
    ax = fig.gca(projection={2: None, 3: Axes3D.name}[n_dim], label='tri')

    if not plot_axes:
        if n_dim == 2:
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        else:
            ax._axis3don = False
    fig.add_axes(ax)

    # determine triangle element colors
    fcs = _tri_colors(tmesh, seeds, pmesh, phases, color_by, colormap, n_dim)
    phase_nums = range(len(phases))

    # plot triangle mesh
    edge_kwargs.setdefault('linewidths', {2: 0.5, 3: 0.1}[n_dim])
    edge_kwargs.setdefault('edgecolors', 'k')
    tmesh.plot(facecolors=fcs, **edge_kwargs)

    # add legend
    if any([given_names[phase_num] for phase_num in phase_nums]):
        custom_seeds = [None for _ in phases]
        for seed_num in tmesh.element_attributes:
            phase_num = seeds[seed_num].phase
            if custom_seeds[phase_num] is None:
                c = phase_colors[phase_num]
                lbl = phase_names[phase_num]
                phase_patch = patches.Patch(fc=c, ec='k', label=lbl)
                custom_seeds[phase_num] = phase_patch
        handles = [h for h in custom_seeds if h is not None]
        ax.legend(handles=handles, loc=4)

    # format axes
    lims = np.array([np.min(tmesh.points, 0), np.max(tmesh.points, 0)]).T
    if n_dim == 2:
        plt.axis('square')
        plt.xlim(lims[0])
        plt.ylim(lims[1])

    else:
        for x in lims[0]:
            for y in lims[1]:
                for z in lims[2]:
                    ax.plot([x], [y], [z])
        ax.set_aspect('equal')

    # save plot
    for fname in plot_files:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    plt.close('all')


def _tri_colors(tmesh, seeds, pmesh, phases, color_by, colormap, n_dim):
    seed_colors = _seed_colors(seeds, phases, color_by, colormap)

    if n_dim == 2:
        return [seed_colors[n] for n in tmesh.element_attributes]

    else:
        facet_neighbors = np.array(pmesh.facet_neighbors)
        facet_is_ext = np.any(facet_neighbors < 0, axis=1)
        facet_is_cand = np.copy(facet_is_ext)
        facet_is_analyzed = np.copy(facet_is_ext)

        while np.any(facet_is_cand):
            # Find new candidate facets
            for neighs in facet_neighbors[facet_is_cand]:
                for cell_num in neighs:
                    if cell_num < 0:
                        continue
                    s_num = pmesh.seed_numbers[cell_num]
                    p_num = seeds[s_num].phase
                    p_type = phases[p_num].get('material_type', 'solid')
                    if p_type in _misc.kw_void:
                        facet_is_cand[pmesh.regions[cell_num]] = True

            # Add candidates to list of external facets
            facet_is_ext = facet_is_ext | facet_is_cand

            # Remove previously analyzed facets from candidate list
            facet_is_cand = facet_is_cand & ~facet_is_analyzed

            # Update list of analyzed facets
            facet_is_analyzed = facet_is_analyzed | facet_is_cand

        facet_neighbors[facet_is_ext]

        elem_fcs = []
        for facet_num in tmesh.facet_attributes:
            facet_color = 'none'
            if facet_is_ext[facet_num]:
                cell_nums = facet_neighbors[facet_num]
                for cell_num in cell_nums:
                    if cell_num < 0:
                        continue
                    seed_num = pmesh.seed_numbers[cell_num]
                    phase_num = seeds[seed_num].phase

                    phase = phases[phase_num]
                    phase_type = phase.get('material_type', 'solid')
                    if phase_type not in _misc.kw_void:
                        facet_color = seed_colors[seed_num]
            elem_fcs.append(facet_color)
        return elem_fcs


# --------------------------------------------------------------------------- #
#                                                                             #
# Recursively Convert Dictionary Contents                                     #
#                                                                             #
# --------------------------------------------------------------------------- #
def dict_convert(dictionary, filepath='.'):
    """Convert dictionary from xmltodict_

    The xmltodict_ ``parse`` method creates dictionaries with values that
    are all strings, rather than strings, floats, ints, etc.
    This function recursively searches the dictionary for string values and
    attempts to convert the dictionary values.

    If a dictionary contains the key ``dist_type``, it is assumed that
    the corresponding name is a :mod:`scipy.stats` statistical distribution
    and the remaining keys are inputs for that distribution,
    with two exceptions.
    First, if the value of ``dist_type`` is ``cdf``, then the remaining key
    should be ``filename`` and its value should be the path to a CSV file,
    where each row contains the (x, CDF) points along the CDF curve.
    Second, if the value of ``dist_type`` is ``histogram``, then the remaining
    key should also be ``filename`` and its value should be the path to a CSV
    file.
    For the histogram, the first row of this CDF should be the *n+1* bin
    locations and the second row should be the *n* bin heights.

    Additionally, if a key in the dictionary contains ``filename`` or
    ``directory`` and the value associated with that key is a relative path,
    then the filepath is converted from a relative to an absolute path using
    the ``filepath`` input as the reference point.
    This behavior can be switched off by setting ``filepath=False``.

    Args:
        dictionary (list, dict, or collections.OrderedDict): Dictionary or
            dictionaries to be converted.
        filepath (str): *(optional)* Reference path to resolve relative paths.

    Returns:
        list or collections.OrderedDict: A copy of the input where the string
        values have been converted. If only one dict is passed into the
        function, then an instance of :class:`collections.OrderedDict` is
        returned.

    .. _xmltodict: https://github.com/martinblech/xmltodict
    """

    if type(dictionary) is list:
        return [dict_convert(d) for d in dictionary]

    new_dict = collections.OrderedDict()
    for key in dictionary:
        val = dictionary[key]
        if type(val) in (dict, collections.OrderedDict):
            new_val = dict_convert(val, filepath)

            # Exception for scipy.stats distributions
            if 'dist_type' in val:
                new_val = _dist_convert(new_val)

        elif type(val) is list:
            if type(val[0]) is str:
                new_val = [_misc.from_str(v) for v in val]
            else:
                new_val = dict_convert(val, filepath)

        elif val is None:
            new_val = {}

        # Exception for filepaths
        elif any([s in key.lower() for s in ('filename', 'directory')]):
            if not os.path.isabs(val) and filepath:
                new_val = os.path.abspath(os.path.join(filepath, val))
            else:
                new_val = val

        elif type(val) is str:
            new_val = _misc.from_str(val)

        else:
            err_str = 'Cannot parse type for: ' + str(type(val))
            raise ValueError(err_str)

        new_dict[key] = new_val
    return new_dict


def _dist_convert(dist_dict):
    """Convert distribution dictionary to distribution"""

    dist_type = dist_dict['dist_type'].strip().lower()
    params = {k: v for k, v in dist_dict.items() if k != 'dist_type'}

    if dist_type == 'cdf':
        cdf_filename = params['filename']
        with open(cdf_filename, 'r') as file:
            cdf = [[float(s) for s in line.split(',')] for line in file]

        bin_bnds = [x for x, _ in cdf]
        bin_cnts = [cdf[i + 1][1] - cdf[i][1] for i in range(len(cdf) - 1)]
        return scipy.stats.rv_histogram(tuple([bin_cnts, bin_bnds]))

    elif dist_type == 'histogram':
        hist_filename = params['filename']
        with open(hist_filename, 'r') as file:
            hist = [[float(s) for s in line.split(',')] for line in file]
        return scipy.stats.rv_histogram(tuple(hist))

    else:
        return scipy.stats.__dict__[dist_type](**params)


if __name__ == '__main__':
    main()
