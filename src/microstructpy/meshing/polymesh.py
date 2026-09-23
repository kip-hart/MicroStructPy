"""Polygon Meshing

This module contains the class definition for the PolyMesh class.

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
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pyvoro
from matplotlib import collections
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import distance

from microstructpy import _misc
from microstructpy import geometry

__all__ = ['PolyMesh']
__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# PolyMesh Class                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #
class PolyMesh(object):
    """Polygonal/Polyhedral mesh.

    The PolyMesh class contains the points, edges, regions, etc. in a polygon
    (2D) or polyhedron (3D) mesh.

    The points attribute is a numpy array containing the (x, y) or (x, y, z)
    coordinates of each point in the mesh. This is the only attribute that
    contains floating point numbers. The rest contain indices/integers.

    The facets attribute describes the interfaces between the polygons/
    polyhedra. In 2D, these interfaces are line segments and each facet
    contains the indices of the points at each end of the line segment. These
    indices are unorderd. In 3D, the interfaces are polygons so each facet
    contains the indices of the points on that polygon. These indices are
    ordered such that neighboring keypoints are connected by line segments
    that form the polygon.

    The regions attribute contains the area (2D) or volume (3D). In 2D, a
    region is given by an ordered list of facets, or edges, that enclose the
    polygon. In 3D, the region is given by an un-ordered list of facets,
    or polygons, that enclose the polyhedron.

    For each region, there is also an associated seed number and material
    phase. These data are stored in the seed_number and phase_number
    attributes, which have the same length as the regions list.

    Args:
        points (list or numpy.ndarray): An Nx2 or Nx3 array of coordinates
            in the mesh.
        facets (list): List of facets between regions. In 2D, this is a list
            of edges (Nx2). In 3D, this is a list of 3D polygons.
        regions (list): A list of polygons (2D) or polyhedra (3D), with each
            element of the list being a list of facet indices.
        seed_numbers (list or numpy.ndarray): *(optional)* The seed number
            associated with each region.
            Defaults to 0 for all regions.
        phase_numbers (list or numpy.ndarray): *(optional)* The phase number
            associated with each region.
            Defaults to 0 for all regions.
        facet_neighbors (list or numpy.ndarray): *(optional)* The region
            numbers on either side of each facet.
            If not givien, a neighbor list is computed from ``regions``.
        volumes (list or numpy.ndarray): *(optional)* The area/volume of each
            region.
            If not given, region volumes are calculated based on ``points``,
            ``facets``, and ``regions``.

    """

    # ----------------------------------------------------------------------- #
    # Constructors                                                            #
    # ----------------------------------------------------------------------- #
    def __init__(self, points, facets, regions, seed_numbers=None,
                 phase_numbers=None, facet_neighbors=None, volumes=None,
                 periodic_axes=None, periodic_points=None,
                 periodic_facets=None):

        self.points = points
        self.facets = facets
        self.regions = regions

        # Periodicity: flags per axis, and the pairs of (low face, high face)
        # points and facets that are periodic images of each other, per axis
        self.periodic_axes = periodic_axes
        self.periodic_points = periodic_points
        self.periodic_facets = periodic_facets

        if facet_neighbors is None:
            # Find facet neighbors
            facet_neighs = [[-1, -1] for _ in facets]
            n_neighs = [0 for _ in facets]
            for r_num, region in enumerate(regions):
                for f_num in region:
                    ind = n_neighs[f_num]
                    facet_neighs[f_num][ind] = r_num
                    n_neighs[f_num] += 1

            # update negative neighbor numbers to follow the voro++
            # convention, described in the %n section of this website:
            # http://math.lbl.gov/voro++/doc/custom.html
            pt_arr = np.array(points)
            pt_mins = pt_arr.min(axis=0)
            pt_maxs = pt_arr.max(axis=0)
            for fnum, facet in enumerate(facets):
                if facet_neighs[fnum][-1] != -1:
                    continue

                f_pts = pt_arr[facet, :]
                min_match = np.all(np.isclose(f_pts, pt_mins), axis=0)
                if np.any(min_match):
                    ld = 3 - len(min_match)
                    mask = np.pad(min_match, (0, ld), 'constant',
                                  constant_values=(False, False))
                    id = np.array([-1, -3, -5])[mask][0]
                    facet_neighs[fnum][-1] = id

                max_match = np.all(np.isclose(f_pts, pt_maxs), axis=0)
                if np.any(max_match):
                    ld = 3 - len(max_match)
                    mask = np.pad(max_match, (0, ld), 'constant',
                                  constant_values=(False, False))
                    id = np.array([-2, -4, -6])[mask][0]
                    facet_neighs[fnum][-1] = id

            self.facet_neighbors = facet_neighs
        else:
            self.facet_neighbors = facet_neighbors

        if seed_numbers is None:
            self.seed_numbers = [0 for _ in regions]
        else:
            self.seed_numbers = seed_numbers

        if phase_numbers is None:
            self.phase_numbers = [0 for _ in regions]
        else:
            self.phase_numbers = phase_numbers

        if volumes is None:
            vols = np.zeros(len(self.regions))
            n = len(self.points[0])
            for i, region in enumerate(self.regions):
                # 'center' of region is arbitrary, since region is convex
                cen = np.array(self.points)[self.facets[region[0]][0]]
                for f_num in region:
                    facet = np.array(self.facets[f_num])
                    j_max = len(facet) - n + 2
                    # convert facet into (n-1)D simplices
                    for j in range(1, j_max):
                        inds = np.append(np.arange(j, j + n - 1), 0)
                        simplex = facet[inds]
                        facet_pts = np.array(self.points)[simplex]
                        rel_pos = facet_pts - cen
                        # simplex volume is |det([Dx1, Dy1; Dx2, Dy2])| in 2D
                        vols[i] += np.abs(np.linalg.det(rel_pos))

            # the 1/2 out front in 2D and 1/6 in 3D
            while n > 1:
                vols /= n
                n -= 1
            self.volumes = vols
        else:
            self.volumes = volumes

    # ----------------------------------------------------------------------- #
    # Representation and String Functions                                     #
    # ----------------------------------------------------------------------- #
    def __repr__(self):
        repr_str = 'PolyMesh('
        repr_str += repr(self.points)
        repr_str += ', '
        repr_str += repr(self.facets)
        repr_str += ', '
        repr_str += repr(self.regions)
        for att in ('seed_numbers', 'phase_numbers'):
            repr_str += ', '
            vals = self.__dict__[att]
            if all([n == 0 for n in vals]):
                repr_str += repr(None)
            else:
                repr_str += repr(vals)
        repr_str += ')'
        return repr_str

    def __str__(self):
        nv = len(self.points)

        # points are written with full precision (repr of a float is the
        # shortest string that round-trips exactly)
        str_str = 'Mesh Points: ' + str(nv) + '\n'
        str_str += ''.join(['\t' + ', '.join([repr(float(x)) for x in p])
                            + '\n' for p in self.points])

        str_str += 'Mesh Facets: ' + str(len(self.facets)) + '\n'
        str_str += ''.join(['\t' + str(tuple(f))[1:-1] + '\n'
                            for f in self.facets])

        str_str += 'Facet Neighbors: ' + str(len(self.facet_neighbors)) + '\n'
        str_str += ''.join(['\t' + str(tuple(n))[1:-1] + '\n'
                            for n in self.facet_neighbors])

        str_str += 'Mesh Regions: ' + str(len(self.regions)) + '\n'
        str_str += ''.join(['\t' + str(tuple(r))[1:-1] + '\n'
                            for r in self.regions])

        str_str += 'Seed Numbers: ' + str(len(self.seed_numbers)) + '\n'
        str_str += ''.join(['\t' + str(n) + '\n' for n in self.seed_numbers])

        str_str += 'Phase Numbers: ' + str(len(self.phase_numbers)) + '\n'
        str_str += ''.join(['\t' + str(n) + '\n' for n in self.phase_numbers])

        str_str += 'Volumes: ' + str(len(self.volumes)) + '\n'
        str_str += '\n'.join(['\t' + str(v) for v in self.volumes])

        if self.periodic_axes is not None and any(self.periodic_axes):
            flags = [int(bool(f)) for f in self.periodic_axes]
            str_str += '\nPeriodic Axes: ' + str(len(flags)) + '\n'
            str_str += '\t' + ', '.join([str(f) for f in flags])
            for name, pairs in (('Periodic Points', self.periodic_points),
                                ('Periodic Facets', self.periodic_facets)):
                rows = [(ax, lo, hi) for ax in sorted(pairs or {})
                        for lo, hi in pairs[ax]]
                str_str += '\n' + name + ': ' + str(len(rows))
                str_str += ''.join(['\n\t' + ', '.join([str(n) for n in row])
                                    for row in rows])
        return str_str

    # ----------------------------------------------------------------------- #
    # Read and Write Functions                                                #
    # ----------------------------------------------------------------------- #
    def write(self, filename, format='txt'):
        """Write the mesh to a file.

        This function writes the polygon/polyhedron mesh to a file.
        See the :ref:`s_poly_file_io` section of the
        :ref:`c_file_formats` guide for more information about the available
        output file formats.

        Args:
            filename (str): Name of the file to be written.
            format (str): *(optional)* {'txt' | 'poly' | 'ply' | 'vtk' }
                Format of the data in the file. Defaults to ``'txt'``.

        """
        if format in ('str', 'txt'):
            with open(filename, 'w') as f:
                f.write(str(self) + '\n')

        elif format == 'poly':
            nv = len(self.points)
            nd = len(self.points[0])
            nf = len(self.facets)
            assert nd == 2

            poly = '# Polygon Mesh\n'
            poly += ' '.join([str(n) for n in (nv, 2, 0, 0)]) + '\n'

            # vertices
            poly += '# Vertices\n'
            poly += ''.join([str(i) + ''.join([' {: e}'.format(x) for x in pt])
                             + '\n' for i, pt in enumerate(self.points)])

            # facets
            poly += '# Segments\n'
            poly += ' '.join([str(n) for n in (nf, 0)]) + '\n'
            poly += ''.join([' '.join([str(n) for n in (nv + i, k1, k2)])
                             + '\n' for i, (k1, k2) in enumerate(self.facets)])

            with open(filename, 'w') as f:
                f.write(poly)

        elif format == 'ply':
            nv = len(self.points)
            nd = len(self.points[0])
            nf = len(self.facets)
            nr = len(self.regions)
            assert nd <= 3

            # Force 3D points
            pts = np.zeros((nv, 3))
            pts[:, :nd] = self.points
            axes = ['x', 'y', 'z']

            # header
            ply = 'ply\n'
            ply += 'format ascii 1.0\n'
            ply += 'element vertex ' + str(nv) + '\n'
            ply += ''.join(['property float32 ' + a + '\n' for a in axes])
            if nd == 2:
                n_faces = nr
            else:
                n_faces = nf
            ply += 'element face {}\n'.format(n_faces)
            ply += 'property list uchar int vertex_indices\n'
            ply += 'end_header\n'

            # vertices
            ply += ''.join([' '.join(['{: e}'.format(x) for x in pt]) + '\n'
                            for pt in pts])

            # faces
            if nd == 2:  # regions -> faces
                facets = np.array(self.facets)
                ply += ''.join([str(len(r)) + ''.join([' ' + str(kp) for kp in
                                                       kp_loop(facets[r])])
                                + '\n' for r in self.regions])

            else:  # facets -> faces
                ply += ''.join([str(len(f)) + ''.join([' ' + str(kp)
                                                       for kp in f])
                                + '\n' for f in self.facets])

            with open(filename, 'w') as f:
                f.write(ply)

        elif format == 'vtk':
            vtk_s = '# vtk DataFile Version 2.0\n'
            vtk_s += 'Polygonal Mesh\n'
            vtk_s += 'ASCII\n'
            vtk_s += 'DATASET UNSTRUCTURED_GRID\n'
            if len(self.points[0]) == 2:
                # Points
                vtk_s += 'POINTS {} float\n'.format(len(self.points))
                for pt in self.points:
                    vtk_s += ' '.join(['{: e}'.format(x) for x in pt]) + ' 0\n'
                vtk_s += '\n'

                # Cells
                n_cells = len(self.regions)
                n_data_total = 0
                cells = 'CELLS {}'.format(n_cells) + ' {}\n'
                pts = np.array(self.points)
                for facets in self.regions:
                    vloop = kp_loop([self.facets[f] for f in facets])
                    n_kp = len(vloop)

                    v1 = pts[vloop[1]] - pts[vloop[0]]
                    v2 = pts[vloop[2]] - pts[vloop[0]]
                    cross_p = np.cross(v1, v2)

                    cells += '{} '.format(n_kp)
                    if cross_p > 0:
                        cells += ' '.join([str(kp) for kp in vloop])
                    else:
                        cells += ' '.join([str(kp) for kp in vloop[::-1]])
                    cells += '\n'
                    n_data_total += 1 + n_kp
                vtk_s += cells.format(n_data_total)

                # Cell Types
                vtk_s += 'CELL_TYPES {}\n'.format(n_cells)
                vtk_s += ''.join(n_cells * ['7\n'])

            else:
                # Points
                vtk_s += 'POINTS {} float\n'.format(len(self.points))
                for pt in self.points:
                    vtk_s += ' '.join(['{: e}'.format(x) for x in pt]) + '\n'
                vtk_s += '\n'

                # Cells
                n_cells = len(self.regions)
                cells = 'CELLS {} '.format(n_cells) + '{}\n'
                n_data_total = 0
                pts = np.array(self.points)
                for facets in self.regions:
                    # Get region center
                    kps = list({kp for f in facets for kp in self.facets[f]})
                    cen = pts[kps].mean(axis=0)  # estimate of center

                    # Write facets
                    n_data_region = 1
                    line = '{} ' + str(len(facets))
                    for f_num in facets:
                        facet = self.facets[f_num]
                        f_len = len(facet)

                        # Determine clockwise or counter-clockwise
                        v1 = pts[facet[1]] - pts[facet[0]]
                        v2 = pts[facet[2]] - pts[facet[0]]
                        norm_vec = np.cross(v1, v2)
                        cen_rel = cen - pts[facet[0]]
                        dot_p = np.dot(norm_vec, cen_rel)

                        line += ' {} '.format(f_len)
                        if dot_p < 0:
                            line += ' '.join([str(kp) for kp in facet])
                        else:
                            line += ' '.join([str(kp) for kp in facet[::-1]])
                        n_data_region += 1 + f_len
                    line += '\n'
                    cells += line.format(n_data_region)
                    n_data_total += 1 + n_data_region
                vtk_s += cells.format(n_data_total)
                vtk_s += '\n'

                # Cell Types
                vtk_s += 'CELL_TYPES {}\n'.format(n_cells)
                vtk_s += ''.join(n_cells * ['42\n'])

            # Cell Data
            vtk_s += '\nCELL_DATA ' + str(n_cells) + '\n'
            vtk_s += 'SCALARS seed int 1 \n'
            vtk_s += 'LOOKUP_TABLE seed\n'
            vtk_s += ''.join([str(a) + '\n' for a in self.seed_numbers])

            vtk_s += 'SCALARS phase int 1 \n'
            vtk_s += 'LOOKUP_TABLE phase\n'
            vtk_s += ''.join([str(a) + '\n' for a in self.phase_numbers])

            vtk_s += 'SCALARS volume float 1 \n'
            vtk_s += 'LOOKUP_TABLE volume\n'
            vtk_s += ''.join([str(a) + '\n' for a in self.volumes])

            with open(filename, 'w') as file:
                file.write(vtk_s)

        else:
            e_str = 'Cannot understand format string ' + str(format) + '.'
            raise ValueError(e_str)

    @classmethod
    def from_file(cls, filename):
        """Read PolyMesh from file.

        This function reads in a polygon mesh from a file and creates an
        instance from that file. Currently the only supported file type
        is the output from :meth:`.write` with the ``format='txt'`` option.

        Args:
            filename (str): Name of file to read from.

        Returns:
            PolyMesh: The instance of the class written to the file.

        """
        with open(filename, 'r') as file:
            stage = 0
            pts = []
            facets = []
            f_neighbors = []
            regions = []
            seed_numbers = []
            phase_numbers = []
            volumes = []
            per_axes = None
            per_pts = []
            per_fts = []
            for line in file.readlines():
                if 'Periodic Axes'.lower() in line.lower():
                    stage = 'periodic axes'
                elif 'Periodic Points'.lower() in line.lower():
                    stage = 'periodic points'
                elif 'Periodic Facets'.lower() in line.lower():
                    stage = 'periodic facets'
                elif 'Mesh Points'.lower() in line.lower():
                    n_pts = int(line.split(':')[1])
                    stage = 'points'
                elif 'Mesh Facets'.lower() in line.lower():
                    n_fts = int(line.split(':')[1])
                    stage = 'facets'
                elif 'Facet Neighbors'.lower() in line.lower():
                    n_nns = int(line.split(':')[1])
                    stage = 'facet neighbors'
                elif 'Mesh Regions'.lower() in line.lower():
                    n_rns = int(line.split(':')[1])
                    stage = 'regions'
                elif 'Seed Numbers'.lower() in line.lower():
                    n_sns = int(line.split(':')[1])
                    stage = 'seed numbers'
                elif 'Phase Numbers'.lower() in line.lower():
                    n_pns = int(line.split(':')[1])
                    stage = 'phase numbers'
                elif 'Volumes'.lower() in line.lower():
                    n_vols = int(line.split(':')[1])
                    stage = 'volumes'
                else:
                    if stage == 'points':
                        pts.append([float(x) for x in line.split(',')])
                    elif stage == 'facets':
                        facets.append([int(kp) for kp in line.split(',')])
                    elif stage == 'facet neighbors':
                        f_neighbors.append([int(n) for n in line.split(',')])
                    elif stage == 'regions':
                        regions.append([int(f) for f in line.split(',')])
                    elif stage == 'seed numbers':
                        seed_numbers.append(_misc.from_str(line))
                    elif stage == 'phase numbers':
                        phase_numbers.append(_misc.from_str(line))
                    elif stage == 'volumes':
                        volumes.append(_misc.from_str(line))
                    elif stage == 'periodic axes':
                        per_axes = [bool(int(f)) for f in line.split(',')]
                    elif stage == 'periodic points':
                        per_pts.append([int(n) for n in line.split(',')])
                    elif stage == 'periodic facets':
                        per_fts.append([int(n) for n in line.split(',')])
                    else:
                        pass

        # check the inputs
        assert len(pts) == n_pts
        assert len(facets) == n_fts
        assert len(regions) == n_rns
        assert len(seed_numbers) == n_sns
        assert len(phase_numbers) == n_pns
        assert len(volumes) == n_vols

        if len(f_neighbors) == 0:
            f_neighbors = None
        else:
            assert len(f_neighbors) == n_nns

        per_points = None
        per_facets = None
        if per_axes is not None:
            per_points = {ax: [] for ax, f in enumerate(per_axes) if f}
            per_facets = {ax: [] for ax, f in enumerate(per_axes) if f}
            for ax, lo, hi in per_pts:
                per_points[ax].append((lo, hi))
            for ax, lo, hi in per_fts:
                per_facets[ax].append((lo, hi))

        return cls(pts, facets, regions, seed_numbers, phase_numbers,
                   volumes=volumes, facet_neighbors=f_neighbors,
                   periodic_axes=per_axes, periodic_points=per_points,
                   periodic_facets=per_facets)

    # ----------------------------------------------------------------------- #
    # Construct from Seed List                                                #
    # ----------------------------------------------------------------------- #
    @classmethod
    def from_seeds(cls, seedlist, domain, edge_opt=False, n_iter=100,
                   verbose=False, periodic=False):
        """Create from :class:`.SeedList` and a domain.

        This function creates a polygon/polyhedron mesh from a seed list and
        a domain. It relies on the pyvoro package, which wraps `Voro++`_.
        The mesh is a Voronoi power diagram / Laguerre tessellationself.

        The pyvoro package operates on rectangular domains, so other domains
        are meshed in 2D by meshing in a bounding box then the boundary cells
        are clipped to the domain boundary.
        Currently non-rectangular domains in 3D are not supported.

        This function also includes the option to maximize the shortest edges
        in the polygonal/polyhedral mesh. Short edges cause numerical
        issues in finite element analysis - setting `edge_opt` to True can
        improve mesh quality with minimal changes to the microstructure.

        Args:
            seedlist (SeedList): A list of seeds in the microstructure.
            domain (from :mod:`microstructpy.geometry`): The domain to be
                filled by the seed.
            edge_opt (bool): *(optional)* This option will maximize the minimum
                edge length in the PolyMesh. The seeds associated with the
                shortest edge are displaced randomly to find improvement and
                this process iterates until `n_iter` attempts have been made
                for a given edge. Defaults to False.
            n_iter (int): *(optional)* Maximum number of iterations per edge
                during optimization. Ignored if `edge_opt` set to False.
                Defaults to 100.
            verbose (bool): *(optional)* Print status of edge optimization to
                screen. Defaults to False.
            periodic (bool, list, or str): *(optional)* Periodicity of the
                microstructure: True for all axes, a list of booleans (one
                per axis), or the names of the periodic axes such as
                ``'x'`` or ``'xy'``. The tessellation is then periodic across
                those faces of the (rectangular) domain: cells that cross a
                periodic face are split into pieces that tile the domain,
                and the points and facets on opposite faces are paired
                (see ``periodic_points`` and ``periodic_facets``).
                Defaults to False.

        Returns:
            PolyMesh: A polygon/polyhedron mesh.

        .. _`Voro++`: http://math.lbl.gov/voro++/

        """
        per_axes = _misc.periodic_axes(periodic, domain.n_dim)
        is_periodic = any(per_axes)
        if is_periodic:
            dom_lims = _misc.periodic_domain_limits(domain)
            if domain.n_dim != 2:
                e_str = 'Periodic tessellations are currently supported in '
                e_str += '2D only.'
                raise NotImplementedError(e_str)

        # Collect all breakdowns
        bkdwn2seed = np.array([], dtype='int')
        bkdwns = np.array([])
        for seed_num, seed in enumerate(seedlist):
            if len(seed.breakdown) == 0:
                seed.update_breakdown()
            bkdwn = np.array(seed.breakdown).reshape(-1, domain.n_dim + 1)
            if is_periodic:
                # centers outside the domain along a periodic axis are
                # wrapped into it (Voro++ needs the particles in the box)
                bkdwn = _wrap_points(bkdwn, dom_lims, per_axes)
            in_mask = domain.within(bkdwn[:, :-1])
            breakdown = bkdwn[in_mask]

            m, n = breakdown.shape
            bkdwns = np.concatenate((bkdwns.reshape(-1, n), breakdown))
            bkdwn2seed = np.append(bkdwn2seed, np.full(m, seed_num))

        n_pts = bkdwns.shape[0]
        n_dim = bkdwns.shape[1] - 1

        # modify point list and boundaries if necessary
        geom = {2: geometry.Rectangle, 3: geometry.Box}
        voro_dom = geom[n_dim](limits=domain.limits)

        # get domain limits
        lims = voro_dom.limits

        # clip points from voro domain
        flag_val = min(bkdwn2seed) - 1
        n_pad = bkdwns.shape[0] - n_pts
        bkdwn2seed = np.pad(bkdwn2seed, (0, n_pad), 'constant',
                            constant_values=(0, flag_val))

        cens = bkdwns[:, :-1]
        rads = bkdwns[:, -1]

        # get block size
        sz = 2 * max(rads)
        if np.isclose(sz, 0):
            sz = 0.1 * np.min([ub - lb for lb, ub in lims])

        # remove extraneous breakdowns
        removing_pts = True
        while removing_pts:
            # Create a temporary file to run pyvoro
            call_str = 'import pyvoro\n\n'

            call_str += 'pts = ['
            beg_str = ',\n' + len('pts = [') * ' '
            call_str += beg_str.join([str(np.array(p).tolist()) for p in cens])
            call_str += ']\n\n'

            call_str += 'lims = ' + str(np.array(lims).tolist()) + '\n\n'

            call_str += 'sz = ' + str(sz) + '\n\n'

            call_str += 'rads = ['
            beg_str = ',\n' + len('rads = [') * ' '
            call_str += beg_str.join([str(rad) for rad in rads])
            call_str += ']\n\n'

            call_str += 'pyvoro.compute_'
            if n_dim == 2:
                call_str += '2d_'
            call_str += 'voronoi(pts, lims, sz, rads, periodic='
            call_str += str([bool(f) for f in per_axes]) + ')\n'

            file = tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                               delete=False)
            file.write(call_str)
            call_filename = file.name
            file.close()

            # Run pyvoro
            p = subprocess.Popen([sys.executable, call_filename],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            p_out, _ = p.communicate()
            try:
                p.terminate()
            except OSError:
                pass

            os.remove(call_filename)

            # if there is output, remove those cells from the list
            out_str = p_out.decode('utf-8')
            if out_str:
                inds = [int(s) for s in out_str.split(':')[-1].split()]
                mask = np.full(len(rads), True)
                mask[inds] = False
                cens = cens[mask]
                rads = rads[mask]
                bkdwn2seed = bkdwn2seed[mask]
            else:
                removing_pts = False

        missing_seeds = set(range(len(seedlist))) - set(bkdwn2seed)
        assert not missing_seeds, str(missing_seeds)
        # compute voronoi diagram
        voro_fun = {2: pyvoro.compute_2d_voronoi,
                    3: pyvoro.compute_voronoi}[n_dim]
        voro = voro_fun(cens, lims, sz, rads,
                        periodic=[bool(f) for f in per_axes])

        if is_periodic:
            # Cells of a periodic tessellation wrap across the periodic
            # faces: split them at those faces and translate the outside
            # pieces into the domain
            voro, bkdwn2seed = _periodic_pieces_2d(voro, bkdwn2seed, lims,
                                                   per_axes)

        # Get only the cells within the domain
        cell_mask = np.full(len(bkdwn2seed), True, dtype='bool')
        rect_doms = ['square', 'cube', 'rectangle', 'box', 'nbox']
        if type(domain).__name__.lower() not in rect_doms:
            if n_dim == 2:
                # Clip the cells to the domain. Cells that do not intersect
                # the domain are removed.
                for cell_num, cell in enumerate(voro):
                    clipped_cell = _clip_cell(cell, domain)
                    if clipped_cell is None:
                        cell_mask[cell_num] = False
                    else:
                        voro[cell_num] = clipped_cell
            else:
                for cell_num, cell in enumerate(voro):
                    cell_pts = np.array(cell['vertices'])
                    cell_mask[cell_num] = np.any(domain.within(cell_pts))
        bkdwn2seed = bkdwn2seed[cell_mask]

        new_cell_nums = np.full(len(cell_mask), -1, dtype='int')
        new_cell_nums[cell_mask] = np.arange(np.sum(cell_mask))

        reduced_voro = []
        for old_cell_num, cell in enumerate(voro):
            # update the numbers of adjacent cells
            faces = cell['faces']
            for face in faces:
                old_adj_cell_num = face['adjacent_cell']
                if old_adj_cell_num >= 0:
                    new_adj_cell_num = new_cell_nums[old_adj_cell_num]
                    face['adjacent_cell'] = new_adj_cell_num
            cell['faces'] = faces

            # add cell to voro
            if cell_mask[old_cell_num]:
                reduced_voro.append(cell)

        # Clip cells to domain (2D cells have already been clipped)
        if n_dim == 2:
            voro = reduced_voro
        else:
            voro = [_clip_cell(c, domain) for c in reduced_voro]

        # create global key point and facet lists
        pts_global = []
        pts_conn = []
        local_kp_conn = {}

        for cell_num, cell_data in enumerate(voro):
            pts_local = cell_data['vertices']

            for face_data in cell_data['faces']:
                adj_cell = face_data['adjacent_cell']
                simplex_local = face_data['vertices']
                for kp_local in simplex_local:
                    key = (cell_num, kp_local)
                    if key in local_kp_conn:
                        kp_global = local_kp_conn[key]

                    else:
                        kp_global = len(pts_global)
                        pt = pts_local[kp_local]
                        pts_global.append(pt)
                        pts_conn.append({cell_num: kp_local})
                        local_kp_conn[key] = kp_global

                    if (adj_cell >= 0) and (adj_cell < len(voro)):
                        conn_info = pts_conn[kp_global]
                        if adj_cell not in conn_info:
                            adj_cell_data = voro[adj_cell]
                            adj_pts_local = np.array(adj_cell_data['vertices'])
                            rel_pos = adj_pts_local - pts_global[kp_global]
                            sq_dist = np.sum(rel_pos * rel_pos, axis=-1)
                            adj_kp_local = np.argmin(sq_dist)

                            adj_key = (adj_cell, adj_kp_local)
                            local_kp_conn[adj_key] = kp_global
                            pts_conn[kp_global][adj_cell] = adj_kp_local

        # create facet and region lists
        facet_list = []
        facet_neighbor_list = []
        region_list = [[] for cell in voro]
        for cell_num, cell_data in enumerate(voro):
            for face_data in cell_data['faces']:
                adj_cell_num = face_data['adjacent_cell']
                if adj_cell_num >= len(voro):
                    adj_cell_num = -1

                if adj_cell_num < cell_num:
                    neighbor_pair = (adj_cell_num, cell_num)
                    s_lcl = face_data['vertices']
                    s_glbl = [local_kp_conn[(cell_num, kp)] for kp in s_lcl]
                    if adj_cell_num < 0:
                        pts_f = [pts_global[kp] for kp in s_glbl]
                        if not _is_outward(pts_f, adj_cell_num):
                            s_glbl.reverse()

                    face_num = len(facet_list)
                    facet_list.append(s_glbl)
                    facet_neighbor_list.append(neighbor_pair)

                    for f_cell_num in neighbor_pair:
                        if (f_cell_num >= 0) and (f_cell_num < len(voro)):
                            region_list[f_cell_num].append(face_num)

        # create phase number list
        phase_nums = [seedlist[i].phase for i in bkdwn2seed]

        # Create volume list
        vols = [cell['volume'] for cell in voro]

        # Create initial mesh
        pmesh = cls(pts_global, facet_list, region_list, bkdwn2seed,
                    phase_nums, facet_neighbor_list, vols)
        if is_periodic:
            pmesh._set_periodic_pairs(per_axes, dom_lims)

        # short edge optimization
        if edge_opt:
            # Find the shortest edge
            edge_lens = _edge_lengths(pmesh)
            min_edge = _shortest_edge(edge_lens)
            min_len = edge_lens[min_edge]['length']

            # Format verbose print string
            n_kps = len(pmesh.points)
            n_kp_space = int(np.log10(n_kps)) + 1
            n_iter_space = int(np.log10(n_iter))
            v_fmt = 'min length: {0:.3e} | '
            v_fmt += 'edge: {1[0]:' + str(n_kp_space) + 'd}, '
            v_fmt += '{1[1]:' + str(n_kp_space) + 'd} | '
            v_fmt += 'n iter: {2:' + str(n_iter_space) + 'd} / '
            v_fmt += str(n_iter)

            i_n_attempts = 0
            while i_n_attempts < n_iter:
                if verbose:
                    print(v_fmt.format(min_len, min_edge, i_n_attempts))

                # Seeds adjacent to the shortest edge
                e_regions = [r for r in edge_lens[min_edge]['regions']
                             if r >= 0]
                e_seeds = sorted({int(pmesh.seed_numbers[r])
                                  for r in e_regions})
                edge_pts = np.array(pmesh.points)[list(min_edge)]

                # Displace the seeds rigidly, on a copy of the seed list.
                # The step is a random fraction of 0.1 x the equivalent
                # radius of the seed, normal to the edge.
                trial_seeds = copy.deepcopy(seedlist)
                steps = {}
                for seed_num in e_seeds:
                    seed = seedlist[seed_num]
                    pos = np.array(seed.position, dtype='float')
                    with np.errstate(divide='ignore', invalid='ignore'):
                        e_norm_vec = _point_line_vec(pos, edge_pts)
                    if not np.all(np.isfinite(e_norm_vec)):
                        continue
                    if n_dim == 2:
                        r_eq = np.sqrt(seed.volume / np.pi)
                    else:
                        r_eq = np.cbrt(3 * seed.volume / (4 * np.pi))
                    step_frac = 2 * np.random.rand() - 1  # [-1, 1]
                    steps[seed_num] = 0.1 * step_frac * r_eq * e_norm_vec
                    _displace_seed(trial_seeds[seed_num], steps[seed_num])

                # Create New Polygonal Mesh
                try:
                    new_pmesh = cls.from_seeds(trial_seeds, domain,
                                               edge_opt=False,
                                               periodic=periodic)
                except AssertionError:
                    i_n_attempts += 1
                    continue

                new_edge_lens = _edge_lengths(new_pmesh)
                new_min_edge = _shortest_edge(new_edge_lens)
                new_min_len = new_edge_lens[new_min_edge]['length']

                if new_min_len > min_len:
                    # Accept the trial: apply the same displacements to the
                    # caller's seeds, so that they reproduce the new mesh
                    for seed_num, step in steps.items():
                        _displace_seed(seedlist[seed_num], step)

                    if new_min_edge != min_edge:
                        i_n_attempts = 0
                    else:
                        i_n_attempts += 1
                    edge_lens = new_edge_lens
                    pmesh = new_pmesh
                    min_len = new_min_len
                    min_edge = new_min_edge
                else:
                    i_n_attempts += 1
        return pmesh

    # ----------------------------------------------------------------------- #
    # Periodicity                                                             #
    # ----------------------------------------------------------------------- #
    def _set_periodic_pairs(self, per_axes, dom_lims):
        """Pair the points and facets on opposite periodic faces.

        For each periodic axis, every point on the lower face is matched
        with its image on the upper face; the coordinates of the pair are
        snapped so that the image is exactly the point translated by the
        domain length. Facets lying on the faces are paired likewise.
        The results are stored in ``periodic_axes``, ``periodic_points``
        (dict: axis -> list of (lower, upper) point numbers) and
        ``periodic_facets`` (dict: axis -> list of (lower, upper) facet
        numbers).

        Raises:
            ValueError: If a point or facet on a periodic face has no
                image on the opposite face.

        """
        pts, per_points = _misc.pair_periodic_points(self.points, per_axes,
                                                     dom_lims)
        per_facets = _misc.pair_periodic_facets(self.facets, per_points)

        self.points = pts.tolist()
        self.periodic_axes = [bool(f) for f in per_axes]
        self.periodic_points = per_points
        self.periodic_facets = per_facets

    # ----------------------------------------------------------------------- #
    # Plot Mesh                                                               #
    # ----------------------------------------------------------------------- #
    def plot(self, index_by='seed', material=[], loc=0, **kwargs):
        """Plot the mesh.

        This function plots the polygon mesh.
        In 2D, this creates a class:`matplotlib.collections.PolyCollection`
        and adds it to the current axes.
        In 3D, it creates a
        :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection` and
        adds it to the current axes.
        The keyword arguments are passed though to matplotlib.

        Args:
            index_by (str): *(optional)* {'facet' | 'material' | 'seed'}
                Flag for indexing into the other arrays passed into the
                function. For example,
                ``plot(index_by='material', color=['blue', 'red'])`` will plot
                the regions with ``phase_number`` equal to 0 in blue, and
                regions with ``phase_number`` equal to 1 in red. The facet
                option is only available for 3D plots. Defaults to 'seed'.
            material (list): *(optional)* Names of material phases. One entry
                per material phase (the ``index_by`` argument is ignored).
                If this argument is set, a legend is added to the plot with
                one entry per material.
            loc (int or str): *(optional)* The location of the legend,
                if 'material' is specified. This argument is passed directly
                through to :func:`matplotlib.pyplot.legend`. Defaults to 0,
                which is 'best' in matplotlib.
            **kwargs: Keyword arguments for matplotlib.

        """
        n_dim = len(self.points[0])
        if n_dim == 2 or plt.gcf().axes:
            ax = plt.gca()
        else:
            ax = plt.gcf().add_subplot(projection=Axes3D.name)
        n_obj = _misc.ax_objects(ax)
        if n_obj > 0:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        else:
            xlim = [float('inf'), -float('inf')]
            ylim = [float('inf'), -float('inf')]
        if n_dim == 2:
            # create vertex loops for each poly
            vloops = [kp_loop([self.facets[f] for f in r]) for r in
                      self.regions]
            # create poly input
            xy = [np.array([self.points[kp] for kp in lp]) for lp in vloops]

            plt_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, (list, np.ndarray)):
                    plt_value = []
                    for s, p in zip(self.seed_numbers, self.phase_numbers):
                        if index_by == 'material':
                            region_value = value[p]
                        elif index_by == 'seed':
                            region_value = value[s]
                        else:
                            e_str = 'Cannot index by {}.'.format(index_by)
                            raise ValueError(e_str)
                        plt_value.append(region_value)
                else:
                    plt_value = value
                plt_kwargs[key] = plt_value
            pc = collections.PolyCollection(xy, **plt_kwargs)
            ax.add_collection(pc)
            ax.autoscale_view()
        elif n_dim == 3:
            if n_obj > 0:
                zlim = ax.get_zlim()
            else:
                zlim = [float('inf'), -float('inf')]
            self.plot_facets(index_by=index_by, **kwargs)

        else:
            raise NotImplementedError('Cannot plot in ' + str(n_dim) + 'D.')

        # Add legend
        if material and index_by in ('seed', 'material'):
            p_kwargs = [{'label': m} for m in material]
            s2p = {s: p for s, p in zip(self.seed_numbers, self.phase_numbers)}
            for key, value in kwargs.items():
                if isinstance(value, (list, np.ndarray)):
                    if index_by == 'material':
                        for p, v in enumerate(value):
                            p_kwargs[p][key] = v
                    else:
                        for s, v in enumerate(value):
                            p = s2p[s]
                            p_kwargs[p][key] = v
                else:
                    for i, m in enumerate(material):
                        p_kwargs[i][key] = value

            # Replace plural keywords
            for p_kw in p_kwargs:
                for kw in _misc.mpl_plural_kwargs:
                    if kw in p_kw:
                        p_kw[kw[:-1]] = p_kw[kw]
                        del p_kw[kw]
            handles = [patches.Patch(**p_kw) for p_kw in p_kwargs]
            ax.legend(handles=handles, loc=loc)

        # Adjust Axes
        mins = np.array(self.points).min(axis=0)
        maxs = np.array(self.points).max(axis=0)
        xlim = (min(xlim[0], mins[0]), max(xlim[1], maxs[0]))
        ylim = (min(ylim[0], mins[1]), max(ylim[1], maxs[1]))
        if n_dim == 2:
            plt.axis('square')
            plt.xlim(xlim)
            plt.ylim(ylim)
        elif n_dim == 3:
            zlim = (min(zlim[0], mins[2]), max(zlim[1], maxs[2]))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

            _misc.axisEqual3D(ax)

    def plot_facets(self, index_by='seed', hide_interior=True, **kwargs):
        """Plot PolyMesh facets.

        This function plots the facets of the polygon mesh, rather than the
        regions.
        In 2D, it adds a :class:`matplotlib.collections.LineCollection` to the
        current axes.
        In 3D, it adds a
        :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection`
        with ``facecolors='none'``.
        The keyword arguments are passed though to matplotlib.

        Args:
            index_by (str): *(optional)* {'facet' | 'material' | 'seed'}
                Flag for indexing into the other arrays passed into the
                function. For example,
                ``plot(index_by='material', color=['blue', 'red'])`` will plot
                the regions with ``phase_number`` equal to 0 in blue, and
                regions with ``phase`` equal to 1 in red. The facet option is
                only available for 3D plots. Defaults to 'seed'.
            hide_interior (bool): If True, removes interior facets from the
                output plot. This avoids occasional matplotlib issue where
                interior facets are shown in output plots.
            **kwargs (dict): Keyword arguments for matplotlib.

        """
        f_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, (list, np.ndarray)):
                f_values = []
                for fn in range(len(self.facets)):
                    neighs = self.facet_neighbors[fn]
                    r = max(neighs)
                    sn = self.seed_numbers[r]
                    pn = self.phase_numbers[r]
                    if index_by == 'facet':
                        ind = fn
                    elif index_by == 'material':
                        ind = pn
                    elif index_by == 'seed':
                        ind = sn
                    else:
                        e_str = 'Cannot index by {}.'.format(index_by)
                        raise ValueError(e_str)
                    v = value[ind]
                    f_values.append(v)
                f_kwargs[key] = f_values
            else:
                f_kwargs[key] = value

        n_dim = len(self.points[0])
        if n_dim == 2 or plt.gcf().axes:
            ax = plt.gca()
        else:
            ax = plt.gcf().add_subplot(projection=Axes3D.name)
        n_obj = _misc.ax_objects(ax)
        if n_obj > 0:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
        else:
            xlim = [float('inf'), -float('inf')]
            ylim = [float('inf'), -float('inf')]
        if n_dim == 2:
            xy = [np.array([self.points[kp] for kp in f]) for f in self.facets]

            pc = collections.LineCollection(xy, **f_kwargs)
            ax.add_collection(pc)
            ax.autoscale_view()
        else:

            if ax.has_data:
                zlim = ax.get_zlim()
            else:
                zlim = [float('inf'), -float('inf')]

            if hide_interior:
                f_mask = [min(fn) < 0 for fn in self.facet_neighbors]
                xy = [np.array([self.points[kp] for kp in f]) for m, f in
                      zip(f_mask, self.facets) if m]
                list_kws = [k for k, vl in f_kwargs.items()
                            if isinstance(vl, list)]
                plt_kwargs = {k: vl for k, vl in f_kwargs.items() if
                              k not in list_kws}
                for k in list_kws:
                    v = [val for val, m in zip(f_kwargs[k], f_mask) if m]
                    plt_kwargs[k] = v
            else:
                xy = [np.array([self.points[kp] for kp in f]) for f in
                      self.facets]
                plt_kwargs = f_kwargs
            pc = Poly3DCollection(xy, **plt_kwargs)
            ax.add_collection(pc)

        # Adjust Axes
        mins = np.array(self.points).min(axis=0)
        maxs = np.array(self.points).max(axis=0)

        xlim = (min(xlim[0], mins[0]), max(xlim[1], maxs[0]))
        ylim = (min(ylim[0], mins[1]), max(ylim[1], maxs[1]))
        if n_dim == 2:
            plt.axis('square')
            plt.xlim(xlim)
            plt.ylim(ylim)
        if n_dim == 3:
            zlim = (min(zlim[0], mins[2]), max(zlim[1], maxs[2]))

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            _misc.axisEqual3D(ax)

    # ----------------------------------------------------------------------- #
    # Mesh Equality                                                           #
    # ----------------------------------------------------------------------- #
    def __eq__(self, other_mesh):
        # check type
        if type(other_mesh) is not PolyMesh:
            return False

        # check that the lengths are all the same
        same = True
        same &= len(self.points) == len(other_mesh.points)
        same &= len(self.facets) == len(other_mesh.facets)
        same &= len(self.regions) == len(other_mesh.regions)
        same &= len(self.seed_numbers) == len(other_mesh.seed_numbers)
        same &= len(self.phase_numbers) == len(other_mesh.phase_numbers)
        if not same:
            return False

        # check that the vertices have the same coordinates
        pt_dists = distance.cdist(self.points, other_mesh.points)
        same_pt = np.isclose(pt_dists, 0)
        same_ints = same_pt.astype(int)
        same &= np.all(same_ints.sum(axis=0) == 1)
        same &= np.all(same_ints.sum(axis=1) == 1)
        if not same:
            return False

        # kp_other[i] is the point in other_mesh that matches point i
        kp_other = np.argmax(same_pt, axis=1)

        # check that the facets are the same
        o_fnum = _match_index_sets(
            [[kp_other[kp] for kp in f] for f in self.facets],
            other_mesh.facets)
        if o_fnum is None:
            return False

        # check that the regions are the same
        o_rnum = _match_index_sets(
            [[o_fnum[f] for f in r] for r in self.regions],
            other_mesh.regions)
        if o_rnum is None:
            return False

        # check that the seed numbers are the same
        s_seed_nums = np.array(self.seed_numbers)
        o_seed_nums = np.array(other_mesh.seed_numbers)
        same &= np.all(s_seed_nums == o_seed_nums[o_rnum])

        # check that the phase numbers are the same
        s_phase_nums = np.array(self.phase_numbers)
        o_phase_nums = np.array(other_mesh.phase_numbers)
        same &= np.all(s_phase_nums == o_phase_nums[o_rnum])

        return bool(same)


def _wrap_points(bkdwn, dom_lims, per_axes):
    """Wrap the centers of a breakdown into the domain along periodic axes.

    Args:
        bkdwn (numpy.ndarray): N x (d + 1) array of (center, radius) rows.
        dom_lims (list): (lower, upper) bounds of the domain, per axis.
        per_axes (list): Periodicity flag of each axis.

    Returns:
        numpy.ndarray: The wrapped breakdown.

    """
    bkdwn = np.array(bkdwn, dtype='float')
    for axis, flag in enumerate(per_axes):
        if not flag:
            continue
        lb, ub = dom_lims[axis]
        bkdwn[:, axis] = lb + np.mod(bkdwn[:, axis] - lb, ub - lb)
    return bkdwn


def _cell_loop(cell):
    """Vertex loop of a 2D pyvoro cell and the adjacent cell of each edge.

    Returns:
        tuple: The vertices in loop order (N x 2 array) and a list with the
        adjacent cell of the edge that starts at each vertex.

    """
    faces = cell['faces']
    loop = kp_loop([f['vertices'] for f in faces])
    edge_adj = {frozenset(f['vertices']): f['adjacent_cell'] for f in faces}
    pts = np.array(cell['vertices'], dtype='float')[loop]
    n_kp = len(loop)
    adj = [edge_adj[frozenset((loop[k], loop[(k + 1) % n_kp]))]
           for k in range(n_kp)]
    return pts, adj


def _clip_loop(pts, adj, axis, value, keep_below, wall, tol):
    """Clip a convex polygon by an axis-aligned line (Sutherland-Hodgman).

    Args:
        pts (numpy.ndarray): Vertices of the polygon, in loop order.
        adj (list): Adjacent cell of the edge starting at each vertex.
        axis (int): Axis of the clipping line.
        value (float): Position of the clipping line along the axis.
        keep_below (bool): Keep the side below the line (True) or above it.
        wall (int): Adjacent cell id given to the edges created on the
            line (a negative wall id).
        tol (float): Points within this distance of the line are on it.

    Returns:
        tuple: The clipped vertices and their edge adjacencies (empty if
        the polygon lies entirely on the other side).

    """
    n_kp = len(pts)
    if keep_below:
        inside = pts[:, axis] <= value + tol
    else:
        inside = pts[:, axis] >= value - tol

    new_pts = []
    new_adj = []
    for k in range(n_kp):
        k1 = (k + 1) % n_kp
        p, q = pts[k], pts[k1]
        if inside[k]:
            new_pts.append(p)
            new_adj.append(adj[k])
        if inside[k] != inside[k1]:
            t = (value - p[axis]) / (q[axis] - p[axis])
            x = p + t * (q - p)
            x[axis] = value
            new_pts.append(x)
            # leaving the kept side: the next edge lies on the line;
            # entering it: the edge from the crossing to q is the original
            new_adj.append(wall if inside[k] else adj[k])

    if len(new_pts) < 3:
        return np.zeros((0, pts.shape[1])), []

    # edges that lie on the line are walls
    new_pts = np.array(new_pts)
    for k in range(len(new_pts)):
        k1 = (k + 1) % len(new_pts)
        if (abs(new_pts[k, axis] - value) <= tol and
                abs(new_pts[k1, axis] - value) <= tol):
            new_adj[k] = wall
    return new_pts, new_adj


def _periodic_pieces_2d(voro, bkdwn2seed, lims, per_axes):
    """Split the cells of a periodic 2D tessellation at the periodic faces.

    The cells computed by Voro++ in periodic mode wrap across the periodic
    faces of the domain. Each cell is cut at those faces and the pieces
    outside the domain are translated into it, so that the pieces tile the
    domain. The cut edges become domain boundary facets (Voro++ wall ids
    -1/-2 for the x faces, -3/-4 for the y faces) and the adjacent cell of
    every other edge is resolved to the piece that shares it.

    Args:
        voro (list): The cells from pyvoro.
        bkdwn2seed (numpy.ndarray): Seed number of each cell.
        lims (list): (lower, upper) bounds of the domain, per axis.
        per_axes (list): Periodicity flag of each axis.

    Returns:
        tuple: The pieces, in the pyvoro cell format, and the seed number
        of each piece.

    Raises:
        ValueError: If a cell is wider than the domain (too few seeds for
            a periodic tessellation) or an edge cannot be matched.

    """
    lengths = [ub - lb for lb, ub in lims]
    tol = 1e-10 * max(lengths)
    area_tol = 1e-12 * np.prod(lengths)

    # Cut the cells at the periodic faces
    pieces = []  # (cell number, vertices, edge adjacencies)
    for cell_num, cell in enumerate(voro):
        parts = [_cell_loop(cell)]
        for axis, flag in enumerate(per_axes):
            if not flag:
                continue
            lb, ub = lims[axis]
            length = ub - lb
            wall_lo = -(2 * axis + 1)
            wall_hi = -(2 * axis + 2)
            new_parts = []
            for pts, adj in parts:
                extent = pts[:, axis].max() - pts[:, axis].min()
                if extent > length + tol:
                    e_str = 'A cell of the periodic tessellation is wider '
                    e_str += 'than the domain along axis ' + str(axis)
                    e_str += '. More seeds are needed for a periodic '
                    e_str += 'microstructure.'
                    raise ValueError(e_str)
                # part below the lower face, translated to the upper side
                below = _clip_loop(pts, adj, axis, lb, True, wall_hi, tol)
                rest = _clip_loop(pts, adj, axis, lb, False, wall_lo, tol)
                if len(rest[0]) == 0:
                    inner, above = rest, rest
                else:
                    inner = _clip_loop(rest[0], rest[1], axis, ub, True,
                                       wall_hi, tol)
                    above = _clip_loop(rest[0], rest[1], axis, ub, False,
                                       wall_lo, tol)
                for (p_pts, p_adj), shift in ((below, length), (inner, 0),
                                              (above, -length)):
                    if len(p_pts) < 3:
                        continue
                    if _loop_area(p_pts, list(range(len(p_pts)))) < area_tol:
                        continue
                    p_pts = np.array(p_pts)
                    p_pts[:, axis] += shift
                    new_parts.append((p_pts, p_adj))
            parts = new_parts
        for pts, adj in parts:
            pieces.append((cell_num, pts, adj))

    # Resolve the adjacent cells of the edges to pieces
    cell_pieces = {}
    for piece_num, (cell_num, _, _) in enumerate(pieces):
        cell_pieces.setdefault(cell_num, []).append(piece_num)

    new_voro = []
    for piece_num, (cell_num, pts, adj) in enumerate(pieces):
        n_kp = len(pts)
        faces = []
        for k in range(n_kp):
            k1 = (k + 1) % n_kp
            adj_cell = adj[k]
            if adj_cell >= 0:
                candidates = [p for p in cell_pieces.get(adj_cell, [])
                              if p != piece_num]
                adj_cell = _matching_piece(pts[k], pts[k1], candidates,
                                           pieces, tol)
                if adj_cell is None:
                    adj_cell = _wall_of_edge(pts[k], pts[k1], lims, tol)
            faces.append({'adjacent_cell': int(adj_cell),
                          'vertices': [k, k1]})
        new_voro.append({'vertices': pts.tolist(),
                         'faces': faces,
                         'adjacency': [[(k - 1) % n_kp, (k + 1) % n_kp]
                                       for k in range(n_kp)],
                         'original': voro[cell_num]['original'],
                         'volume': _loop_area(pts, list(range(n_kp)))})
    new_bkdwn2seed = np.array([bkdwn2seed[cell_num]
                               for cell_num, _, _ in pieces], dtype='int')
    return new_voro, new_bkdwn2seed


def _matching_piece(pt_a, pt_b, candidates, pieces, tol):
    """Piece among the candidates that has vertices at both points."""
    for piece_num in candidates:
        pts = pieces[piece_num][1]
        d_a = np.min(np.linalg.norm(pts - pt_a, axis=1))
        d_b = np.min(np.linalg.norm(pts - pt_b, axis=1))
        if d_a <= tol and d_b <= tol:
            return piece_num
    return None


def _wall_of_edge(pt_a, pt_b, lims, tol):
    """Wall id of an edge lying on a face of the domain."""
    for axis, (lb, ub) in enumerate(lims):
        if abs(pt_a[axis] - lb) <= tol and abs(pt_b[axis] - lb) <= tol:
            return -(2 * axis + 1)
        if abs(pt_a[axis] - ub) <= tol and abs(pt_b[axis] - ub) <= tol:
            return -(2 * axis + 2)
    e_str = 'Cannot resolve the neighbor of the edge between '
    e_str += str(pt_a.tolist()) + ' and ' + str(pt_b.tolist())
    e_str += ' in the periodic tessellation.'
    raise ValueError(e_str)


def _match_index_sets(items, other_items):
    """Match each item to an unused item of ``other_items`` with the same
    set of indices. Returns the list of matched positions, or None if any
    item has no match."""
    unused = {}
    for j, other_item in enumerate(other_items):
        unused.setdefault(frozenset(other_item), []).append(j)

    matches = []
    for item in items:
        candidates = unused.get(frozenset(item), [])
        if not candidates:
            return None
        matches.append(candidates.pop(0))
    return matches


def kp_loop(kp_pairs):
    loop = list(kp_pairs[0])
    kp_arr = np.array(kp_pairs[1:])
    while kp_arr.shape[0] > 0:
        kp_find = loop[-1]
        has_kp = np.any(kp_arr == kp_find, axis=1)
        row = kp_arr[has_kp]

        loop.append(row[row != kp_find][0])
        kp_arr = kp_arr[~has_kp]
    assert loop[0] == loop[-1]
    return loop[:-1]


def _clip_cell(cell_data, domain):
    """Clip a Voronoi cell to the domain.

    Rectangular domains do not require clipping and the cell is returned
    unchanged. In 2D, the (convex) cell is clipped to the (convex) domain
    and ``None`` is returned if the cell does not intersect the domain.
    Non-rectangular 3D domains are not supported: a warning is raised and
    the cell is returned unchanged.

    Args:
        cell_data (dict): A cell from pyvoro, with the 'vertices', 'faces',
            'adjacency', 'original', and 'volume' keys.
        domain (from :mod:`microstructpy.geometry`): The domain.

    Returns:
        dict or None: The clipped cell, or None if it is outside the domain.

    """
    domain_name = type(domain).__name__.lower()
    if domain_name in ['rectangle', 'square', 'box', 'cube']:
        return cell_data

    if domain.n_dim == 2:
        return _clip_cell_2d(cell_data, domain)

    w_str = 'Cannot clip cells to fit to a ' + domain_name + '.'
    w_str += ' Currently 3D geometries are not supported, other than boxes.'
    warnings.warn(w_str, RuntimeWarning)
    return cell_data


def _clip_cell_2d(cell_data, domain, n_samples=64, n_bnd_pts=64):
    """Clip a convex 2D cell to a convex domain.

    The vertex loop of the cell is walked and the points inside the domain
    are kept: the vertices within the domain and the points where the edges
    cross the domain boundary (an edge with both ends outside the domain can
    cross it twice). Consecutive kept points that lie on the same edge of the
    cell are joined by that edge, every other gap is an arc of the domain
    boundary and is closed by a boundary face (``'adjacent_cell': -1``).
    If the domain lies entirely within the cell, the cell becomes a polygon
    that approximates the domain boundary.

    Args:
        cell_data (dict): A cell from pyvoro.
        domain (from :mod:`microstructpy.geometry`): The 2D domain.
        n_samples (int): Number of points sampled along an edge with both
            ends outside the domain, to detect a double crossing.
        n_bnd_pts (int): Number of points on the domain boundary, when the
            domain is entirely within the cell.

    Returns:
        dict or None: The clipped cell, or None if it is outside the domain.

    """
    pts = np.array(cell_data['vertices'], dtype='float')
    faces = cell_data['faces']
    within = domain.within(pts)
    if np.all(within):
        return cell_data

    # order the vertices of the cell in a loop
    loop = kp_loop([f['vertices'] for f in faces])
    n_kp = len(loop)
    edge_faces = {frozenset(f['vertices']): f for f in faces}

    # points closer than this are considered coincident
    tol = 1e-12 * max(np.max(np.abs(pts)), np.finfo(float).tiny)

    # walk the loop and collect the points inside the domain, and for each
    # point the face that joins it to the next point (None: domain boundary)
    kept_pts = []
    kept_faces = []

    def add_point(pt, face):
        if kept_pts and np.linalg.norm(pt - kept_pts[-1]) <= tol:
            kept_faces[-1] = face
        else:
            kept_pts.append(pt)
            kept_faces.append(face)

    for i in range(n_kp):
        kp_a = loop[i]
        kp_b = loop[(i + 1) % n_kp]
        face = edge_faces[frozenset((kp_a, kp_b))]
        pt_a = pts[kp_a]
        pt_b = pts[kp_b]
        if within[kp_a] and within[kp_b]:
            add_point(pt_a, face)
        elif within[kp_a]:
            add_point(pt_a, face)
            add_point(_segment_cross([pt_a, pt_b], domain), None)
        elif within[kp_b]:
            add_point(_segment_cross([pt_a, pt_b], domain), face)
        else:
            crossings = _segment_double_cross(pt_a, pt_b, domain, n_samples,
                                              tol)
            if crossings is not None:
                add_point(crossings[0], face)
                add_point(crossings[1], None)

    if len(kept_pts) > 1:
        if np.linalg.norm(kept_pts[-1] - kept_pts[0]) <= tol:
            kept_pts.pop()
            kept_faces.pop()

    if len(kept_pts) == 0:
        # the cell is either outside the domain, or contains the domain
        if not _point_in_convex_loop(domain.center, pts[loop]):
            return None
        kept_pts = list(_domain_boundary(domain, n_bnd_pts))
        kept_faces = [None for _ in kept_pts]

    elif len(kept_pts) == 2:
        # the domain crosses a single edge of the cell, so the clipped cell
        # is bounded by that edge and an arc; the midpoint of the arc is added
        # so that the cell has a non-zero area
        n_faces = sum([f is not None for f in kept_faces])
        if n_faces != 1:
            return None
        if kept_faces[0] is None:
            kept_pts.reverse()
            kept_faces.reverse()
        pt_a, pt_b = kept_pts
        mid_pt = 0.5 * (pt_a + pt_b)
        d_pt = pt_b - pt_a
        n_vec = np.array([-d_pt[1], d_pt[0]])
        if np.dot(n_vec, pts.mean(axis=0) - mid_pt) < 0:
            n_vec *= -1
        n_vec /= np.linalg.norm(n_vec)
        far_pt = mid_pt + _domain_extent(domain) * n_vec
        kept_pts.append(_segment_cross([mid_pt, far_pt], domain))
        kept_faces.append(None)

    n_new = len(kept_pts)
    if n_new < 3:
        return None

    new_pts = np.array(kept_pts)
    new_faces = []
    for k, face in enumerate(kept_faces):
        verts = [k, (k + 1) % n_new]
        if face is None:
            new_faces.append({'adjacent_cell': -1, 'vertices': verts})
        else:
            new_faces.append({'adjacent_cell': face['adjacent_cell'],
                              'vertices': verts})
    new_adj = [[(k - 1) % n_new, (k + 1) % n_new] for k in range(n_new)]

    new_cell_data = {'adjacency': new_adj,
                     'faces': new_faces,
                     'original': cell_data['original'],
                     'vertices': new_pts.tolist(),
                     'volume': _loop_area(new_pts, list(range(n_new)))}
    return new_cell_data


def _segment_cross(pts, domain, n_iter=60):
    """Find the point where a segment crosses the domain boundary.

    One end of the segment must be inside the domain and the other outside.
    The crossing is found by bisection with a fixed number of iterations,
    so the result is accurate to machine precision for any magnitude of the
    coordinates. The result does not depend on the order of the end points.

    Args:
        pts (list or numpy.ndarray): The two end points of the segment.
        domain (from :mod:`microstructpy.geometry`): The domain.
        n_iter (int): Number of bisection iterations.

    Returns:
        numpy.ndarray: The crossing point.

    """
    end_pts = np.array(pts, dtype='float')
    within = domain.within(end_pts)
    for _ in range(n_iter):
        pt = end_pts.mean(axis=0)
        if domain.within(pt) == within[0]:
            end_pts[0] = pt
        else:
            end_pts[1] = pt
    return end_pts.mean(axis=0)


def _segment_double_cross(pt_a, pt_b, domain, n_samples=64, tol=0):
    """Find where a segment with both ends outside the domain crosses it.

    The segment is sampled to detect whether it passes through the (convex)
    domain. The end points are put in a canonical order before sampling, so
    that the two cells sharing an edge compute identical crossing points.

    Returns:
        tuple or None: The crossing points nearest to ``pt_a`` and ``pt_b``,
        or None if the segment does not cross the domain.

    """
    end_pts = np.array([pt_a, pt_b], dtype='float')
    order = np.lexsort((end_pts[:, 1], end_pts[:, 0]))
    pt_0 = end_pts[order[0]]
    pt_1 = end_pts[order[1]]

    t = np.arange(1, n_samples + 1) / (n_samples + 1)
    samples = pt_0 + t.reshape(-1, 1) * (pt_1 - pt_0)
    inside = domain.within(samples)
    if not np.any(inside):
        return None

    pt_in = samples[np.argmax(inside)]
    cross_0 = _segment_cross([pt_in, pt_0], domain)
    cross_1 = _segment_cross([pt_in, pt_1], domain)
    if np.linalg.norm(cross_1 - cross_0) <= tol:
        return None  # the segment only touches the domain

    crossings = [None, None]
    crossings[order[0]] = cross_0
    crossings[order[1]] = cross_1
    return tuple(crossings)


def _domain_extent(domain):
    """A distance that leaves the domain from any point within it."""
    return 2 * max([ub - lb for lb, ub in domain.limits])


def _domain_boundary(domain, n_pts=64):
    """Points on the boundary of a convex 2D domain, in loop order.

    The points are found along rays from the center of the domain.
    """
    cen = np.array(domain.center, dtype='float')
    ext = _domain_extent(domain)
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    dirs = np.column_stack((np.cos(t), np.sin(t)))
    return np.array([_segment_cross([cen, cen + ext * u], domain)
                     for u in dirs])


def _point_in_convex_loop(pt, loop_pts):
    """Test whether a point is within a convex polygon."""
    pt = np.array(pt, dtype='float')
    loop_pts = np.array(loop_pts, dtype='float')
    d_edge = np.roll(loop_pts, -1, axis=0) - loop_pts
    d_pt = pt - loop_pts
    cross = d_edge[:, 0] * d_pt[:, 1] - d_edge[:, 1] * d_pt[:, 0]
    return bool(np.all(cross >= 0) or np.all(cross <= 0))


def _loop_area(pts, loop):
    """Area of the polygon with vertices ``pts[loop[0]]``, ``pts[loop[1]]``,
    ... (shoelace formula)."""
    double_area = 0

    n = len(loop)
    for i in range(n):
        ip1 = (i + 1) % n

        xi = pts[loop[i]][0]
        yi = pts[loop[i]][1]
        xip1 = pts[loop[ip1]][0]
        yip1 = pts[loop[ip1]][1]

        det = xi * yip1 - xip1 * yi
        double_area += det
    return 0.5 * np.abs(double_area)


def _is_outward(pt_list, voropp_face_num):
    n_dim = len(pt_list[0])

    voropp_sgn = 1-2*(voropp_face_num % 2)
    voropp_axis = int((-voropp_face_num - 1)/2)
    face_vec = voropp_sgn * np.eye(n_dim)[voropp_axis]

    if n_dim == 2:
        pt1 = pt_list[0]
        pt2 = pt_list[1]
        rel_pos = np.array(pt2) - np.array(pt1)
        n_vec = np.array([-rel_pos[1], rel_pos[0]])
    elif n_dim == 3:
        pt1 = pt_list[0]
        pt2 = pt_list[1]
        pt3 = pt_list[2]
        r1 = np.array(pt2) - np.array(pt1)
        r2 = np.array(pt3) - np.array(pt1)
        n_vec = np.cross(r1, r2)
    else:
        raise ValueError('Function does not support {}D.'.format(n_dim))

    n_u = n_vec / np.linalg.norm(n_vec)
    return np.dot(n_u, face_vec) > 0


def _edge_lengths(pmesh):
    edge_lens = {}  # (kp1, kp2): {'length': #, 'regions': set()}
    for i, f in enumerate(pmesh.facets):
        n = len(f)
        facet_kp_pairs = [(f[k], f[(k + 1) % n]) for k in range(n)]
        for pair in facet_kp_pairs:
            key = tuple(sorted(pair))
            if key not in edge_lens:  # calculate edge length
                pt1 = pmesh.points[key[0]]
                pt2 = pmesh.points[key[1]]
                rel_pos = np.array(pt2) - np.array(pt1)
                edge_len = np.linalg.norm(rel_pos)
                edge_lens[key] = {
                    'length': edge_len,
                    'regions': set(),
                    }
            neighs = pmesh.facet_neighbors[i]
            edge_lens[key]['regions'] |= set(neighs)
    return edge_lens


def _shortest_edge(edge_lens):
    min_len = float('inf')
    min_pair = (-1, -1)
    for pair in edge_lens:
        length = edge_lens[pair]['length']
        if length < min_len:
            min_len = length
            min_pair = pair
    return min_pair


def _point_line_vec(pt, line_pts):
    ptA, ptB = line_pts
    n_vec = (ptB - ptA) / np.linalg.norm(ptB - ptA)

    rel_pos = ptA - pt
    proj = np.dot(rel_pos, n_vec) * n_vec
    dist_vec = rel_pos - proj

    u_vec = dist_vec / np.linalg.norm(dist_vec)
    return u_vec


def _displace_seed(seed, step):
    """Translate a seed rigidly by ``step``.

    The position setter of the seed translates the geometry and the
    breakdown along with the position.
    """
    if isinstance(seed.breakdown, tuple):
        seed.breakdown = [list(b) for b in seed.breakdown]
    pos = np.array(seed.position, dtype='float')
    seed.position = (pos + np.array(step, dtype='float')).tolist()
