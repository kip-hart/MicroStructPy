"""Command Line Interface.

This module contains the command line interface (CLI) for MicroStructPy.
The CLI primarily reads XML input files and creates a microstructure according
to those inputs. It can also run demo input files.

"""


from __future__ import division
from __future__ import print_function

import argparse
import ast
import collections
import glob
import os
import shutil
import subprocess

import numpy as np
import scipy.stats
import xmltodict
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from microstructpy import _misc
from microstructpy import geometry
from microstructpy import seeding
from microstructpy import verification
from microstructpy.meshing import PolyMesh
from microstructpy.meshing import RasterMesh
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
    file_dict = input2dict(filename)

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


def input2dict(filename, root_tag='input'):
    """Read input file into a dictionary

    This function reads an input file and creates a dictionary of strings
    contained within the file.

    Args:
        filename: Name of the input file.

    Returns:
        collections.OrderedDict: Dictionary of input strings.

    """

    # Read in the file
    with open(filename, 'r') as file:
        file_dict = xmltodict.parse(file.read())
    tag = list(file_dict.keys())[0]
    assert tag == root_tag, repr(tag) + ' != ' + repr(root_tag)

    return _include_expand(file_dict, filename, root_tag)


def _include_expand(inp, filename, key):
    if isinstance(inp,  str):
        return inp
    if isinstance(inp, list):
        return [_include_expand(inp_i, filename, key) for inp_i in inp]

    file_path = os.path.dirname(filename)
    exp_dict = collections.OrderedDict()
    for inp_key, inp_val in inp.items():
        if inp_key == 'include':
            includes = inp_val
            if not isinstance(includes, list):
                includes = [includes]
            for inc_filename in includes:
                inc_fname = os.path.expanduser(inc_filename)
                if os.path.isabs(inc_fname):
                    fname = inc_fname
                else:
                    fname = os.path.join(file_path, inc_fname)
                inc_dict = input2dict(fname, key)
                exp_dict.update(inc_dict[key])
        else:
            exp_dict[inp_key] = _include_expand(inp_val, filename, inp_key)
    return exp_dict


def run(phases, domain, verbose=False, restart=True, directory='.',
        filetypes={}, rng_seeds={}, plot_axes=True, rtol='fit', edge_opt=False,
        edge_opt_n_iter=100, mesher='Triangle/TetGen',
        mesh_max_volume=float('inf'), mesh_min_angle=0,
        mesh_max_edge_length=float('inf'), mesh_size=float('inf'),
        verify=False, color_by='material', colormap='viridis',
        seeds_kwargs={}, poly_kwargs={}, tri_kwargs={}):
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
        edge_opt (bool): *(optional)* This option will maximize the minimum
            edge length in the PolyMesh. The seeds associated with the
            shortest edge are displaced randomly to find improvement and
            this process iterates until `n_iter` attempts have been made
            for a given edge. Defaults to False.
        edge_opt_n_iter (int): *(optional)* Maximum number of iterations per
            edge during optimization. Ignored if `edge_opt` set to False.
            Defaults to 100.
        mesher (str): {'raster' | 'Triangle/TetGen' | 'Triangle'  | 'TetGen' |
            'gmsh'}
            specify the mesh generator. Default is 'Triangle/TetGen'.
        mesh_max_volume (float): *(optional)* The maximum volume (area in 2D)
            of a mesh cell in the triangular mesh. Default is infinity,
            which turns off the maximum volume quality setting.
            Value should be strictly positive.
        mesh_min_angle (float): *(optional)* The minimum interior angle,
            in degrees,  of a cell in the triangular mesh. For 3D meshes,
            this is the dihedral angle between faces of the tetrahedron.
            Defaults to 0, which turns off the angle quality constraint.
            Value should be in the range 0-60.
        mesh_max_edge_length (float): *(optional)* The maximum edge length of
            elements along grain boundaries. Currently only supported in 2D.
        mesh_size (float): The target size of the mesh elements. This
            option is used with gmsh. Default is infinity, whihch turns off
            this control.
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
    seeds_types = filetypes.get('seeds', [])
    if type(seeds_types) != list:
        seeds_types = [seeds_types]
    for seeds_type in seeds_types:
        fname = seed_filename.rstrip('.txt') + '.' + seeds_type
        if seeds_created or not os.path.exists(fname):
            seeds.write(fname, format=seeds_type)

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

    if plot_files:
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

        pmesh = PolyMesh.from_seeds(seeds, domain, edge_opt, edge_opt_n_iter,
                                    verbose)

    # Write polymesh
    poly_types = filetypes.get('poly', [])
    if type(poly_types) != list:
        poly_types = [poly_types]

    for poly_type in poly_types:
        fname = poly_filename.replace('.txt', '.' + poly_type)
        if poly_created or not os.path.exists(fname):
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

    if plot_files:
        plot_poly(pmesh, phases, plot_files, plot_axes, color_by, colormap,
                  **poly_kwargs)

    # ----------------------------------------------------------------------- #
    # Create Triangular Mesh                                                  #
    # ----------------------------------------------------------------------- #
    raster = mesher == 'raster'
    if raster:
        tri_basename = 'rastermesh.txt'
    else:
        tri_basename = 'trimesh.txt'
    tri_filename = os.path.join(directory, tri_basename)
    exts = {'abaqus': '.inp', 'txt': '.txt', 'str': '.txt', 'tet/tri': '',
            'vtk': '.vtk'}

    if restart and os.path.exists(tri_filename) and not poly_created:
        # Read triangle mesh
        if verbose:
            if raster:
                print('Reading raster mesh.')
            else:
                print('Reading triangular mesh.')
            print('Mesh filename: ' + tri_filename)

        tmesh = TriMesh.from_file(tri_filename)
        tri_created = False
    else:
        tri_created = True
        # Create triangular mesh
        if verbose:
            if raster:
                print('Creating raster mesh.')
            else:
                print('Creating triangular mesh.')

        if raster:
            tmesh = RasterMesh.from_polymesh(pmesh, mesh_size, phases)
        else:
            tmesh = TriMesh.from_polymesh(pmesh, phases, mesher,
                                          mesh_min_angle, mesh_max_volume,
                                          mesh_max_edge_length, mesh_size)

    # Write triangular mesh
    tri_types = filetypes.get('tri', [])
    if type(tri_types) != list:
        tri_types = [tri_types]

    for tri_type in tri_types:
        fname = tri_filename.replace('.txt', exts[tri_type])
        if tri_created or not os.path.exists(fname):
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
        if raster:
            bname = 'rastermesh'
        else:
            bname = 'trimesh'
        fname = os.path.join(directory, bname + '.' + str(ext))
        if tri_created or not os.path.exists(fname):
            plot_files.append(fname)

    if plot_files and verbose:
        if raster:
            print('Plotting raster mesh.')
        else:
            print('Plotting triangular mesh.')

    if plot_files:
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
            Defaults to saving the plot to ``seeds.png``.
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
    print('plot files seeds', plot_files)
    if not plot_files:
        plot_files = ['seeds.png']

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

    # Plot seeds
    edge_kwargs.setdefault('edgecolors', {2: 'k', 3: 'none'}[n_dim])
    if given_names and color_by == 'material':
        seeds.plot(material=phase_names, facecolors=seed_colors, loc=4,
                   **edge_kwargs)
    else:
        seeds.plot(facecolors=seed_colors, **edge_kwargs)

    # Crop to Domain
    d_lims = domain.limits
    if n_dim == 2:
        plt.axis('square')
        plt.xlim(d_lims[0])
        plt.ylim(d_lims[1])
    elif n_dim == 3:
        plt.gca().set_xlim(d_lims[0])
        plt.gca().set_ylim(d_lims[1])
        plt.gca().set_zlim(d_lims[2])

        _misc.axisEqual3D(plt.gca())

    # Save plot
    for fname in plot_files:
        if n_dim == 3:
            fig.subplots_adjust(**_misc.plt_3d_adj)
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)
        else:
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


def _phase_color_by(i, phases, color_by='material', colormap='viridis'):
    if color_by == 'material':
        return phases[i].get('color', 'C' + str(i % 10))
    elif color_by == 'material number':
        n = len(phases)
        return _cm_color(i / (n - 1), colormap)


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
        plot_files = ['polymesh.png']

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
        if given_names and color_by == 'material':
            pmesh.plot(facecolors=fcs, material=phase_names)
        else:
            pmesh.plot(facecolors=fcs)

        edge_color = edge_kwargs.pop('edgecolors', (0, 0, 0, 1))
        facet_colors = []
        for neigh_pair in pmesh.facet_neighbors:
            if facet_check(neigh_pair, pmesh, phases):
                facet_colors.append(edge_color)  # black
            else:
                facet_colors.append('none')

        edge_kwargs.setdefault('capstyle', 'round')
        pmesh.plot_facets(color=facet_colors, index_by='facet', **edge_kwargs)
    else:
        edge_kwargs.setdefault('edgecolors', 'k')
        if given_names and color_by == 'material':
            pmesh.plot(facecolors=fcs, index_by='seed', material=phase_names,
                       **edge_kwargs)
        else:
            pmesh.plot(facecolors=fcs, index_by='seed', **edge_kwargs)

    # save plot
    for fname in plot_files:
        if n_dim == 3:
            fig.subplots_adjust(**_misc.plt_3d_adj)
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)
        else:
            plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close('all')


def _poly_colors(pmesh, phases, color_by, colormap, n_dim):
    if n_dim == 2:
        if color_by == 'material':
            r_colors = [_phase_color(n, phases) for n in pmesh.phase_numbers]
        elif color_by == 'seed number':
            n = max(pmesh.seed_numbers) + 1
            r_colors = [_cm_color(s / (n - 1), colormap) for s in
                        pmesh.seed_numbers]
        elif color_by == 'material number':
            n = len(phases)
            r_colors = [_cm_color(p / (n - 1), colormap) for p in
                        pmesh.phase_numbers]
        n_seeds = max(pmesh.seed_numbers) + 1
        s_colors = ['none' for i in range(n_seeds)]
        for seed_num, r_c in zip(pmesh.seed_numbers, r_colors):
            s_colors[seed_num] = r_c
        return s_colors
    else:
        s2p = {s: p for s, p in zip(pmesh.seed_numbers, pmesh.phase_numbers)}
        n = max(s2p.keys()) + 1
        colors = []
        for s in range(n):
            if color_by == 'material':
                phase_num = s2p[s]
                color = _phase_color(phase_num, phases)
            elif color_by == 'seed number':
                color = _cm_color(s / (n - 1), colormap)
            elif color_by == 'material number':
                n_phases = len(phases)
                color = _cm_color(s2p[s] / (n_phases - 1), colormap)
            else:
                color = 'none'
            colors.append(color)
        return colors


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
            Defaults to saving the plot to ``trimesh.png``.
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
        plot_files = ['trimesh.png']

    n_dim = len(tmesh.points[0])

    phase_colors = []
    phase_names = []
    given_names = False
    for i, phase in enumerate(phases):
        color = phase.get('color', 'C' + str(i % 10))
        name = phase.get('name', 'Material ' + str(i + 1))
        phase_colors.append(color)
        phase_names.append(name)
        if 'name' in phase:
            given_names = True

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

    # Determine which facets are visible
    vis_regions = set()
    invis_regions = set(range(-6, 0))
    f_front = set([i for i, fn in enumerate(pmesh.facet_neighbors)
                   if min(fn) < 0])
    while f_front and n_dim > 2:
        new_front = set()
        for f in f_front:
            neighs = set(pmesh.facet_neighbors[f])
            for n in neighs - invis_regions:
                p = pmesh.phase_numbers[n]
                p_type = phases[p].get('material_type', 'solid')
                if p_type in _misc.kw_void:
                    new_front |= set(pmesh.regions[n])
                else:
                    vis_regions.add(n)
        new_front -= f_front
        f_front = new_front
    if n_dim < 3:
        vis_regions = set(range(len(pmesh.regions)))

    # Determine facet colors based on visibility
    seed_colors = _seed_colors(seeds, phases, color_by, colormap)
    facet_colors = []
    facet_phases = []
    for i, fn in enumerate(pmesh.facet_neighbors):
        if _f_plottable(fn, vis_regions, invis_regions):
            r = list(set(fn) - invis_regions)[0]
            s = pmesh.seed_numbers[r]
            color = seed_colors[s]
            phase = seeds[s].phase
        else:
            color = 'none'
            phase = -1
        facet_colors.append(color)
        facet_phases.append(phase)

    # plot triangle mesh
    edge_kwargs.setdefault('linewidths', {2: 0.5, 3: 0.1}[n_dim])
    edge_kwargs.setdefault('edgecolors', 'k')
    if color_by in ('material', 'material number'):
        n = len(phases)
        if n_dim == 2:
            cs = seed_colors
        else:
            cs = facet_colors
            cs.append('none')

        cs = [_phase_color_by(i, phases, color_by, colormap) for i in range(n)]
        cs.append('none')

        old_e_att = np.copy(tmesh.element_attributes)
        old_f_att = np.copy(tmesh.facet_attributes)
        tmesh.element_attributes = [seeds[i].phase for i in old_e_att]
        tmesh.facet_attributes = [facet_phases[a] for a in old_f_att]

        if given_names:
            tmesh.plot(facecolors=cs, index_by='attribute',
                       material=phase_names, **edge_kwargs)
        else:
            tmesh.plot(facecolors=cs, index_by='attribute', **edge_kwargs)

        tmesh.element_attributes = old_e_att
        tmesh.facet_attributes = old_f_att
    else:
        fcs = {2: seed_colors, 3: facet_colors}[n_dim]
        tmesh.plot(facecolors=fcs, index_by='attribute', **edge_kwargs)

    # save plot
    for fname in plot_files:
        if n_dim == 3:
            fig.subplots_adjust(**_misc.plt_3d_adj)
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.15)
        else:
            plt.savefig(fname, bbox_inches='tight', pad_inches=0)

    plt.close('all')


def _f_plottable(n_pair, vis, invis):
    if set(n_pair) <= vis or set(n_pair) <= invis:
        return False
    return True


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
    For the histogram, the first row of this CDF should be the *n* bin heights
    and the second row should be the *n+1* bin locations.

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

    # Convert lists
    if isinstance(dictionary, list):
        return [dict_convert(d) for d in dictionary]

    # Convert strings
    if isinstance(dictionary, str):
        s = _misc.from_str(dictionary)
        if isinstance(s, str) and ',' in s:
            s = [_misc.from_str(ss) for ss in s.split(',')]
        return s

    # Convert Nones
    if dictionary is None:
        return {}

    # Convert filepaths
    for key in dictionary:
        val = dictionary[key]
        if any([s in key.lower() for s in ('filename', 'directory')]):
            val = os.path.expanduser(val)
            if not os.path.isabs(val) and filepath:
                new_val = os.path.abspath(os.path.join(filepath, val))
            else:
                new_val = val
            dictionary[key] = new_val

    # Convert SciPy.stats distributions
    if 'dist_type' in dictionary:
        return _dist_convert(dictionary)

    # Convert Dictionaries
    new_dict = collections.OrderedDict()
    for key in dictionary:
        val = dictionary[key]
        new_val = dict_convert(val, filepath)
        new_dict[key] = new_val
    return new_dict


def _dist_convert(dist_dict):
    """Convert distribution dictionary to distribution"""

    dist_type = dist_dict['dist_type'].strip().lower()
    params = {k: _misc.from_str(v) for k, v in dist_dict.items()}
    del params['dist_type']

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
