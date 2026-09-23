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
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
from scipy.spatial import cKDTree
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
                   verbose=False, periodic=False, periodic_margin=0.0,
                   min_angle=0.0):
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
        In periodic meshes, the cells that cross a periodic face are split
        into pieces, and a piece that is thin (the cell barely crosses the
        face) forces very small elements: with a positive `periodic_margin`,
        the optimization also thickens or removes the pieces thinner than
        the margin. A corner of a cell at a periodic face that is narrower
        than the minimum angle of the mesh (`min_angle`) forces very small
        elements too, since the mesher cannot reach that angle in the
        corner and refines it in shells instead: such corners are opened
        by the optimization like thin pieces.

        Args:
            seedlist (SeedList): A list of seeds in the microstructure.
            domain (from :mod:`microstructpy.geometry`): The domain to be
                filled by the seed.
            edge_opt (bool): *(optional)* This option will maximize the minimum
                edge length in the PolyMesh. The seeds associated with the
                shortest edge are displaced randomly to find improvement and
                this process iterates until `n_iter` attempts have been made
                for a given edge. A trial is kept when the shortest feature
                that it changes (an edge, or the thickness of a piece at a
                periodic face) gets longer: the features that it creates
                are all longer than the shortest one that it removes. The
                accepted displacements are applied to `seedlist`. Defaults
                to False.
            n_iter (int): *(optional)* Maximum number of iterations per edge
                (or per thin piece) during optimization. Ignored if
                `edge_opt` set to False. Defaults to 100.
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
            periodic_margin (float): *(optional)* With `edge_opt`, the
                minimum thickness of the pieces of the cells at the
                periodic faces (their extent normal to the face). The seeds
                of a thinner piece and of its neighbors are moved, normal
                to the face, until the piece is at least this thick or the
                cell no longer crosses the face. Ignored if `edge_opt` is
                False or the mesh is not periodic. Defaults to 0 (only the
                shortest edge is optimized).
            min_angle (float): *(optional)* The minimum angle (2D) or
                dihedral angle (3D) of the mesh that will be built from
                this one, in degrees (the `min_angle` of
                :meth:`.TriMesh.from_polymesh`). With `edge_opt` in
                periodic meshes, a corner of a cell at a periodic face that
                is narrower than this angle (between a facet and the face)
                is a feature like a thin piece, with the size of the small
                elements that the mesher would put in it (a quarter of the
                thickness of the wedge at the end of its shorter side), and
                the seeds on both sides of the facet are moved along it to
                open the corner. Defaults to 0 (no such corners).

        Returns:
            PolyMesh: A polygon/polyhedron mesh.

        .. _`Voro++`: http://math.lbl.gov/voro++/

        """
        per_axes = _misc.periodic_axes(periodic, domain.n_dim)
        is_periodic = any(per_axes)
        if is_periodic:
            dom_lims = _misc.periodic_domain_limits(domain)

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
            pieces_fun = {2: _periodic_pieces_2d, 3: _periodic_pieces_3d}
            voro, bkdwn2seed = pieces_fun[n_dim](voro, bkdwn2seed, lims,
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
        if is_periodic:
            # merge clusters of nearly coincident points, consistently on
            # both periodic faces
            eps = _MERGE_TOL * max([ub - lb for lb, ub in dom_lims])
            collapsed = _collapse_close_points(pts_global, facet_list,
                                               facet_neighbor_list,
                                               region_list, eps)
            pts_global, facet_list, facet_neighbor_list, region_list = \
                collapsed

        pmesh = cls(pts_global, facet_list, region_list, bkdwn2seed,
                    phase_nums, facet_neighbor_list, vols)
        if is_periodic:
            pmesh._set_periodic_pairs(per_axes, dom_lims)

        # short edge (and thin periodic piece) optimization
        if edge_opt:
            pmesh = _optimize_features(cls, pmesh, seedlist, domain, n_iter,
                                       verbose, periodic, periodic_margin,
                                       min_angle)
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
        pts, per_points, per_facets = _misc.pair_periodic_mesh(
            self.points, self.facets, per_axes, dom_lims)
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


# Vertices closer than this fraction of the largest domain length to a
# periodic face are snapped onto it before the cells are cut there.
_SNAP_TOL = 1e-5


def _snap_to_planes(pts, axis, values, snap_tol):
    """Snap the coordinates along an axis that are within a tolerance of
    the given values onto those values (returns a copy)."""
    pts = np.array(pts, dtype='float')
    for value in values:
        mask = np.abs(pts[:, axis] - value) <= snap_tol
        pts[mask, axis] = value
    return pts


# Points closer than this fraction of the largest domain length are merged
# in a periodic mesh (Voro++ can produce clusters of nearly coincident
# vertices, which the periodic faces must share consistently).
_MERGE_TOL = 1e-6


def _collapse_close_points(pts, facets, facet_neighbors, regions, eps):
    """Merge the points of a mesh that are closer than ``eps``.

    Clusters of close points are replaced by their mean. Facets left with
    fewer than ``n_dim`` distinct points are removed, along with their
    entries in the regions.

    Args:
        pts (list): The points.
        facets (list): Facets (lists of point numbers).
        facet_neighbors (list): Neighbors of each facet.
        regions (list): Regions (lists of facet numbers).
        eps (float): Merging distance.

    Returns:
        tuple: The new points, facets, facet neighbors and regions.

    """
    pts = np.array(pts, dtype='float')
    n_pts, n_dim = pts.shape
    roots, means = _cluster_points(pts, eps)
    if len(means) == n_pts:
        return pts.tolist(), facets, facet_neighbors, regions

    new_pts = []
    root_ids = {}
    for root in sorted(means):
        root_ids[root] = len(new_pts)
        new_pts.append(means[root])
    kp_new = [root_ids[roots[i]] for i in range(n_pts)]

    new_facets = []
    new_neighs = []
    f_new = {}
    for f_num, facet in enumerate(facets):
        loop = []
        for kp in facet:
            kp_n = kp_new[kp]
            if not loop or loop[-1] != kp_n:
                loop.append(kp_n)
        if len(loop) > 1 and loop[0] == loop[-1]:
            loop.pop()
        if len(set(loop)) >= n_dim:
            f_new[f_num] = len(new_facets)
            new_facets.append(loop)
            new_neighs.append(facet_neighbors[f_num])
    new_regions = [[f_new[f] for f in region if f in f_new]
                   for region in regions]
    return np.array(new_pts).tolist(), new_facets, new_neighs, new_regions


def _cluster_points(pts, tol):
    """Clusters of points closer than ``tol`` to each other (transitively).

    Returns:
        tuple: The root of the cluster of each point (its smallest point
        number), as an array, and a dictionary that maps each root to the
        mean of the points of its cluster.

    """
    n_pts = len(pts)
    sets = _misc.UnionFind(range(n_pts))
    for i, j in cKDTree(pts).query_pairs(tol):
        sets.union(i, j)
    roots = np.array([sets.find(i) for i in range(n_pts)])
    means = {}
    for root in np.unique(roots):
        means[root] = pts[np.nonzero(roots == root)[0]].mean(axis=0)
    return roots, means


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
    on_line = np.abs(pts[:, axis] - value) <= tol
    if keep_below:
        inside = (pts[:, axis] <= value + tol) | on_line
    else:
        inside = (pts[:, axis] >= value - tol) | on_line

    new_pts = []
    new_adj = []
    for k in range(n_kp):
        k1 = (k + 1) % n_kp
        p, q = pts[k], pts[k1]
        if inside[k]:
            new_pts.append(p)
            new_adj.append(adj[k])
        if inside[k] != inside[k1]:
            if inside[k] and on_line[k]:
                # p itself is the crossing point; the next edge is the cut
                new_adj[-1] = wall
                continue
            if inside[k1] and on_line[k1]:
                continue  # q itself is the crossing point
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


def _unify_cell_vertices(voro, lims, per_axes, merge_tol, snap_tol):
    """Give the cells identical coordinates for their shared vertices.

    Voro++ computes each cell on its own, so two cells that share a vertex
    hold copies of it that differ by its precision, and in periodic mode
    the copies may lie in different images of the domain. The copies that
    coincide, modulo the length of the domain along the periodic axes, are
    replaced by their mean, snapped onto the periodic faces when within
    the snapping tolerance, so that the cells are cut consistently at the
    faces and their pieces match exactly.

    Args:
        voro (list): The cells from pyvoro.
        lims (list): (lower, upper) bounds of the domain, per axis.
        per_axes (list): Periodicity flag of each axis.
        merge_tol (float): Distance below which copies are one vertex.
        snap_tol (float): Distance below which a vertex is on a face.

    Returns:
        list: The cells, with the unified vertices.

    """
    lb = np.array([lim[0] for lim in lims], dtype='float')
    lengths = np.array([ub - lo for lo, ub in lims], dtype='float')
    counts = [len(cell['vertices']) for cell in voro]
    all_pts = np.vstack([np.array(cell['vertices'], dtype='float')
                         for cell in voro])

    # the images of the vertices in the domain, along the periodic axes
    shifts = np.zeros_like(all_pts)
    for axis, flag in enumerate(per_axes):
        if flag:
            n_img = np.floor((all_pts[:, axis] - lb[axis]) / lengths[axis])
            shifts[:, axis] = n_img * lengths[axis]
    wrapped = all_pts - shifts

    # coincident copies are one vertex
    roots, means = _cluster_points(wrapped, merge_tol)
    unified = np.array([means[root] for root in roots])

    # vertices next to a periodic face are on it
    for axis, flag in enumerate(per_axes):
        if not flag:
            continue
        for value in (lb[axis], lb[axis] + lengths[axis]):
            on_face = np.abs(unified[:, axis] - value) <= snap_tol
            unified[on_face, axis] = value
    unified += shifts

    new_voro = []
    start = 0
    for cell, count in zip(voro, counts):
        new_cell = dict(cell)
        new_cell['vertices'] = unified[start:start + count].tolist()
        new_voro.append(new_cell)
        start += count
    return new_voro


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
    merge_tol = _MERGE_TOL * max(lengths)
    snap_tol = _SNAP_TOL * max(lengths)
    voro = _unify_cell_vertices(voro, lims, per_axes, merge_tol, snap_tol)

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
                # vertices next to a cut line are snapped onto it, so that
                # the two cells sharing an edge are cut consistently and
                # no sliver pieces are created
                pts = _snap_to_planes(pts, axis, (lb, ub), snap_tol)
                _check_cell_width(pts, axis, length, tol)
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
                    p_pts = np.array(p_pts)
                    if p_pts[:, axis].max() - p_pts[:, axis].min() <= tol:
                        continue  # flat piece, lies on the cut line
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
                adj_cell = _matching_piece(pts[[k, k1]], candidates, pieces,
                                           merge_tol)
                if adj_cell is None:
                    adj_cell = _wall_of_points(pts[[k, k1]], lims, tol)
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


def _check_cell_width(pts, axis, length, tol):
    """Raise if a cell is wider than the domain along an axis: it cannot
    be cut into pieces that tile the domain (too few seeds)."""
    extent = pts[:, axis].max() - pts[:, axis].min()
    if extent > length + tol:
        e_str = 'A cell of the periodic tessellation is wider '
        e_str += 'than the domain along axis ' + str(axis)
        e_str += '. More seeds are needed for a periodic '
        e_str += 'microstructure.'
        raise ValueError(e_str)


# --------------------------------------------------------------------------- #
#                                                                             #
# Periodic Tessellation - 3D                                                  #
#                                                                             #
# --------------------------------------------------------------------------- #
def _clip_polyhedron(verts, faces, axis, value, keep_below, wall, tol):
    """Clip a convex polyhedron by an axis-aligned plane.

    Args:
        verts (numpy.ndarray): N x 3 vertices.
        faces (list): (vertex loop, adjacent cell) pairs.
        axis (int): Axis of the clipping plane.
        value (float): Position of the plane along the axis.
        keep_below (bool): Keep the side below the plane (True) or above.
        wall (int): Adjacent cell id of the face created on the plane.
        tol (float): Vertices within this distance of the plane are on it.

    Returns:
        tuple: The clipped vertices and faces (empty if nothing is kept).

    """
    on_plane = np.abs(verts[:, axis] - value) <= tol
    if keep_below:
        inside = (verts[:, axis] <= value + tol) | on_plane
    else:
        inside = (verts[:, axis] >= value - tol) | on_plane
    if np.all(inside):
        return verts, faces
    if not np.any(inside):
        return np.zeros((0, 3)), []

    new_verts = []
    kp_map = {}
    for kp, is_in in enumerate(inside):
        if is_in:
            kp_map[kp] = len(new_verts)
            new_verts.append(verts[kp])

    # intersection points are computed once per edge, so that the two
    # faces sharing the edge use the same point
    edge_cut = {}

    def cut_point(kp_a, kp_b):
        key = (min(kp_a, kp_b), max(kp_a, kp_b))
        if key not in edge_cut:
            p, q = verts[key[0]], verts[key[1]]
            t = (value - p[axis]) / (q[axis] - p[axis])
            x = p + t * (q - p)
            x[axis] = value
            edge_cut[key] = len(new_verts)
            new_verts.append(x)
        return edge_cut[key]

    new_faces = []
    for loop, adj in faces:
        n_kp = len(loop)
        new_loop = []
        for k in range(n_kp):
            kp_a, kp_b = loop[k], loop[(k + 1) % n_kp]
            if inside[kp_a]:
                new_loop.append(kp_map[kp_a])
            if inside[kp_a] != inside[kp_b]:
                # A vertex on the plane is itself the crossing point: no
                # new (coincident) vertex is created for it.
                if inside[kp_a] and not on_plane[kp_a]:
                    new_loop.append(cut_point(kp_a, kp_b))
                elif inside[kp_b] and not on_plane[kp_b]:
                    new_loop.append(cut_point(kp_a, kp_b))
        # drop repeated consecutive vertices (edges lying on the plane)
        loop_out = []
        for kp in new_loop:
            if not loop_out or loop_out[-1] != kp:
                loop_out.append(kp)
        if len(loop_out) > 1 and loop_out[0] == loop_out[-1]:
            loop_out.pop()
        if len(loop_out) >= 3:
            new_faces.append((loop_out, adj))

    new_verts = np.array(new_verts)

    # The cap face is the cross-section of the (convex) polyhedron by the
    # plane: the convex hull of the cut points and the vertices on the
    # plane. This does not depend on the orientation of the faces.
    cap_ids = set(edge_cut.values())
    cap_ids |= set([kp_map[kp] for kp in range(len(verts))
                    if inside[kp] and on_plane[kp]])
    cap_loop = _plane_hull_loop(new_verts, sorted(cap_ids), axis, tol)
    if len(cap_loop) >= 3:
        new_faces.append((cap_loop, wall))
    # faces lying on the plane are walls
    for i, (loop, adj) in enumerate(new_faces):
        if np.all(np.abs(new_verts[loop, axis] - value) <= tol):
            new_faces[i] = (loop, wall)
    return new_verts, new_faces


def _plane_hull_loop(verts, ids, axis, tol):
    """Loop of the points (given by id) that bound the convex hull of a set
    of coplanar points, in a plane normal to ``axis``.

    Points that lie on an edge of the hull (collinear) are included, so
    that the loop shares every vertex with the faces around it.

    Returns:
        list: The point ids in loop order (empty if fewer than 3 points
        span the hull).

    """
    if len(ids) < 3:
        return []
    others = [i for i in range(verts.shape[1]) if i != axis]
    pts_2d = verts[ids][:, others]
    try:
        hull = ConvexHull(pts_2d)
    except QhullError:
        return []  # collinear points: the cross-section is degenerate
    hull_ids = [int(i) for i in hull.vertices]
    loop = [ids[i] for i in hull_ids]

    # insert the points lying on hull edges
    on_hull = set(hull_ids)
    rest = [i for i in range(len(ids)) if i not in on_hull]
    if rest:
        new_loop = []
        n_hull = len(hull_ids)
        for k in range(n_hull):
            i_a, i_b = hull_ids[k], hull_ids[(k + 1) % n_hull]
            p_a, p_b = pts_2d[i_a], pts_2d[i_b]
            d_ab = p_b - p_a
            length = np.linalg.norm(d_ab)
            new_loop.append(ids[i_a])
            on_edge = []
            for i in rest:
                rel = pts_2d[i] - p_a
                t = np.dot(rel, d_ab) / (length * length)
                if -1e-12 < t < 1 + 1e-12:
                    dist = abs(rel[0] * d_ab[1] - rel[1] * d_ab[0]) / length
                    if dist <= tol:
                        on_edge.append((t, ids[i]))
            new_loop.extend([kp for _, kp in sorted(on_edge)])
        loop = new_loop
    return loop


def _polyhedron_volume(verts, faces):
    """Volume of a convex polyhedron given by its faces (fan from the
    centroid of the vertices)."""
    cen = verts.mean(axis=0)
    volume = 0.0
    for loop, _ in faces:
        p0 = verts[loop[0]]
        for k in range(1, len(loop) - 1):
            p1, p2 = verts[loop[k]], verts[loop[k + 1]]
            volume += abs(np.dot(np.cross(p1 - p0, p2 - p0), cen - p0))
    return volume / 6.0


def _periodic_pieces_3d(voro, bkdwn2seed, lims, per_axes):
    """Split the cells of a periodic 3D tessellation at the periodic faces.

    The 3D counterpart of :func:`_periodic_pieces_2d`: each wrapped cell is
    clipped by the planes of the periodic faces, the outside pieces are
    translated into the domain, cut faces become domain boundary facets
    (Voro++ wall ids -1 ... -6) and the adjacent cells of the other faces
    are resolved to the pieces that share them.

    Args:
        voro (list): The cells from pyvoro.
        bkdwn2seed (numpy.ndarray): Seed number of each cell.
        lims (list): (lower, upper) bounds of the domain, per axis.
        per_axes (list): Periodicity flag of each axis.

    Returns:
        tuple: The pieces, in the pyvoro cell format, and the seed number
        of each piece.

    """
    lengths = [ub - lb for lb, ub in lims]
    tol = 1e-10 * max(lengths)
    merge_tol = _MERGE_TOL * max(lengths)
    snap_tol = _SNAP_TOL * max(lengths)
    voro = _unify_cell_vertices(voro, lims, per_axes, merge_tol, snap_tol)

    pieces = []  # (cell number, vertices, faces)
    for cell_num, cell in enumerate(voro):
        verts = np.array(cell['vertices'], dtype='float')
        faces = [(list(f['vertices']), f['adjacent_cell'])
                 for f in cell['faces']]
        parts = [(verts, faces)]
        for axis, flag in enumerate(per_axes):
            if not flag:
                continue
            lb, ub = lims[axis]
            length = ub - lb
            wall_lo = -(2 * axis + 1)
            wall_hi = -(2 * axis + 2)
            new_parts = []
            for p_verts, p_faces in parts:
                # vertices next to a cut plane are snapped onto it (see
                # _periodic_pieces_2d)
                p_verts = _snap_to_planes(p_verts, axis, (lb, ub), snap_tol)
                _check_cell_width(p_verts, axis, length, tol)
                below = _clip_polyhedron(p_verts, p_faces, axis, lb, True,
                                         wall_hi, tol)
                rest = _clip_polyhedron(p_verts, p_faces, axis, lb, False,
                                        wall_lo, tol)
                if len(rest[0]) == 0:
                    inner, above = rest, rest
                else:
                    inner = _clip_polyhedron(rest[0], rest[1], axis, ub,
                                             True, wall_hi, tol)
                    above = _clip_polyhedron(rest[0], rest[1], axis, ub,
                                             False, wall_lo, tol)
                for (q_verts, q_faces), shift in ((below, length),
                                                  (inner, 0),
                                                  (above, -length)):
                    if len(q_verts) < 4 or len(q_faces) < 4:
                        continue
                    q_verts = np.array(q_verts)
                    if q_verts[:, axis].max() - q_verts[:, axis].min() <= tol:
                        continue  # flat piece, lies on the cut plane
                    q_verts[:, axis] += shift
                    new_parts.append((q_verts, q_faces))
            parts = new_parts
        for p_verts, p_faces in parts:
            pieces.append((cell_num, p_verts, p_faces))

    # Resolve the adjacent cells of the faces to pieces
    cell_pieces = {}
    for piece_num, (cell_num, _, _) in enumerate(pieces):
        cell_pieces.setdefault(cell_num, []).append(piece_num)

    new_voro = []
    for piece_num, (cell_num, verts, faces) in enumerate(pieces):
        out_faces = []
        for loop, adj_cell in faces:
            if adj_cell >= 0:
                candidates = [p for p in cell_pieces.get(adj_cell, [])
                              if p != piece_num]
                adj_cell = _matching_piece(verts[loop], candidates, pieces,
                                           merge_tol)
                if adj_cell is None:
                    adj_cell = _wall_of_points(verts[loop], lims, tol)
            out_faces.append({'adjacent_cell': int(adj_cell),
                              'vertices': list(loop)})
        adjacency = [[] for _ in range(len(verts))]
        for face in out_faces:
            loop = face['vertices']
            for k in range(len(loop)):
                kp_a, kp_b = loop[k], loop[(k + 1) % len(loop)]
                if kp_b not in adjacency[kp_a]:
                    adjacency[kp_a].append(kp_b)
                if kp_a not in adjacency[kp_b]:
                    adjacency[kp_b].append(kp_a)
        new_voro.append({'vertices': verts.tolist(),
                         'faces': out_faces,
                         'adjacency': adjacency,
                         'original': voro[cell_num]['original'],
                         'volume': _polyhedron_volume(verts, faces)})
    new_bkdwn2seed = np.array([bkdwn2seed[cell_num]
                               for cell_num, _, _ in pieces], dtype='int')
    return new_voro, new_bkdwn2seed


def _matching_piece(face_pts, candidates, pieces, tol):
    """Piece among the candidates that has vertices at all the points (the
    two ends of an edge in 2D, the vertices of a face in 3D)."""
    for piece_num in candidates:
        pts = pieces[piece_num][1]
        dists = np.linalg.norm(face_pts[:, None, :] - pts[None, :, :],
                               axis=2)
        if np.all(dists.min(axis=1) <= tol):
            return piece_num
    return None


def _wall_of_points(face_pts, lims, tol):
    """Wall id of the face of the domain on which all the points lie (the
    ends of an edge in 2D, the vertices of a face in 3D)."""
    for axis, (lb, ub) in enumerate(lims):
        if np.all(np.abs(face_pts[:, axis] - lb) <= tol):
            return -(2 * axis + 1)
        if np.all(np.abs(face_pts[:, axis] - ub) <= tol):
            return -(2 * axis + 2)
    e_str = 'Cannot resolve the neighbor of a facet of the periodic '
    e_str += 'tessellation at ' + str(np.round(face_pts.mean(axis=0), 6))
    e_str += '.'
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


def _point_line_vec(pt, line_pts):
    ptA, ptB = line_pts
    n_vec = (ptB - ptA) / np.linalg.norm(ptB - ptA)

    rel_pos = ptA - pt
    proj = np.dot(rel_pos, n_vec) * n_vec
    dist_vec = rel_pos - proj

    u_vec = dist_vec / np.linalg.norm(dist_vec)
    return u_vec


def _displace_seed(seed, step, dom_lims=None, per_axes=None):
    """Translate a seed rigidly by ``step``.

    The position setter of the seed translates the geometry and the
    breakdown along with the position. Along the periodic axes (when
    ``dom_lims`` and ``per_axes`` are given), the position is wrapped back
    into the domain: a seed translated by the period is the same seed.
    """
    if isinstance(seed.breakdown, tuple):
        seed.breakdown = [list(b) for b in seed.breakdown]
    pos = np.array(seed.position, dtype='float')
    pos += np.array(step, dtype='float')
    if dom_lims is not None:
        for axis, flag in enumerate(per_axes):
            if flag:
                lb, ub = dom_lims[axis]
                pos[axis] = lb + (pos[axis] - lb) % (ub - lb)
    seed.position = pos.tolist()


# --------------------------------------------------------------------------- #
# Edge / thin piece optimization                                              #
# --------------------------------------------------------------------------- #
_MAX_OPT_TRIALS_PER_ITER = 50  # safety cap: total trials <= this * n_iter


def _optimize_features(cls, pmesh, seedlist, domain, n_iter, verbose,
                       periodic, periodic_margin, min_angle=0.0):
    """Lengthen the shortest features of a mesh by moving its seeds.

    The features are the edges of the mesh and, in periodic meshes, the
    thicknesses of the pieces of the cells at the periodic faces and the
    corners of the cells at the faces narrower than ``min_angle``. The
    target is the shortest feature or, with a positive margin, the
    thinnest piece or corner under the margin. The seeds around the target are
    displaced on a copy of the seed list and the trial is kept when the
    shortest feature that it changes gets longer: every feature that it
    creates is longer than the shortest one that it removes (for the
    shortest edge of the mesh, this is the usual criterion that the
    shortest edge gets longer). A target that does not improve in
    ``n_iter`` consecutive trials is left alone and the next one is taken;
    the optimization ends when no target is left. The accepted
    displacements are applied to ``seedlist``.

    Returns:
        PolyMesh: The optimized mesh.

    """
    n_dim = domain.n_dim
    per_axes = _misc.periodic_axes(periodic, n_dim)
    dom_lims = None
    if any(per_axes):
        dom_lims = _misc.periodic_domain_limits(domain)
    scale = max([ub - lb for lb, ub in domain.limits])
    tol = 1e-9 * scale

    features = _mesh_features(pmesh, per_axes, dom_lims, scale, min_angle)
    n_kp_space = int(np.log10(max(len(pmesh.points), 1))) + 1
    n_iter_space = int(np.log10(max(n_iter, 1))) + 1

    stuck = set()
    last_key = None
    n_attempts = 0
    n_trials = 0
    max_trials = _MAX_OPT_TRIALS_PER_ITER * n_iter
    while n_trials < max_trials:
        target = _select_target(features, periodic_margin, stuck)
        if target is None:
            break
        if target['key'] != last_key:
            last_key = target['key']
            n_attempts = 0
        if verbose:
            print(_target_string(target, n_attempts, n_iter, n_kp_space,
                                 n_iter_space))

        steps = _trial_steps(target, seedlist, n_attempts, periodic_margin,
                             dom_lims, per_axes, n_dim)
        n_trials += 1
        accepted = False
        if steps:
            trial_seeds = copy.deepcopy(seedlist)
            for seed_num, step in steps.items():
                _displace_seed(trial_seeds[seed_num], step, dom_lims,
                               per_axes)
            try:
                new_pmesh = cls.from_seeds(trial_seeds, domain,
                                           edge_opt=False, periodic=periodic)
            except (AssertionError, ValueError):
                new_pmesh = None
            if new_pmesh is not None:
                new_features = _mesh_features(new_pmesh, per_axes, dom_lims,
                                              scale, min_angle)
                accepted = _accept_trial(new_features, features, tol)

        if accepted:
            # apply the same displacements to the caller's seeds, so that
            # they reproduce the new mesh
            for seed_num, step in steps.items():
                _displace_seed(seedlist[seed_num], step, dom_lims, per_axes)
            pmesh = new_pmesh
            features = new_features
            n_attempts = 0
        else:
            n_attempts += 1
            if n_attempts >= n_iter or not steps:
                stuck.add(target['key'])
    return pmesh


def _region_points(pmesh, region):
    """Sorted point numbers of a region (a list of facet numbers)."""
    return sorted({kp for f in region for kp in pmesh.facets[f]})


def _mesh_features(pmesh, per_axes, dom_lims, scale, min_angle=0.0):
    """Edges of the mesh, pieces of the cells at the periodic faces and
    corners of the cells at the faces narrower than ``min_angle``.

    Each feature is a dict with its ``kind`` ('edge', 'piece' or 'wedge'),
    ``size`` (length, thickness normal to the face, or the size of the
    elements that the corner forces), a ``key`` that identifies it
    geometrically across re-tessellations and the ``seeds`` around it.
    """
    pts = np.array(pmesh.points, dtype='float')
    seed_nums = [int(s) for s in pmesh.seed_numbers]
    features = []
    edge_lens = _edge_lengths(pmesh)
    for (kp1, kp2), info in edge_lens.items():
        regions = [r for r in info['regions'] if r >= 0]
        mid = 0.5 * (pts[kp1] + pts[kp2])
        features.append({
            'kind': 'edge',
            'size': info['length'],
            'key': ('edge',) + tuple(np.round(mid / scale, 6)),
            'seeds': sorted({seed_nums[r] for r in regions}),
            'kps': (kp1, kp2),
            'pts': pts[[kp1, kp2]],
            })
    if dom_lims is None:
        return features

    for r, region in enumerate(pmesh.regions):
        walls = set()
        neighs = set()
        for f in region:
            for n in pmesh.facet_neighbors[f]:
                if n < 0:
                    walls.add(n)
                elif n != r:
                    neighs.add(seed_nums[n])
        if not walls:
            continue
        kps = _region_points(pmesh, region)
        cen = pts[kps].mean(axis=0)
        for wall in sorted(walls):
            axis, side = _misc.wall_axis_side(wall)
            if not per_axes[axis]:
                continue
            # the region touches the wall: its extent normal to the wall
            # is its thickness
            features.append({
                'kind': 'piece',
                'size': np.ptp(pts[kps, axis]),
                'key': ('piece', seed_nums[r], axis, side) +
                tuple(np.round(cen / scale, 6)),
                'seeds': sorted(neighs | {seed_nums[r]}),
                'seed': seed_nums[r],
                'axis': axis,
                'side': side,
                })
    features += _wedge_features(pmesh, pts, seed_nums, per_axes, scale,
                                min_angle)
    return features


def _wedge_features(pmesh, pts, seed_nums, per_axes, scale, min_angle):
    """Corners of the cells at the periodic faces narrower than
    ``min_angle`` (degrees): the angle between a facet of the cell and the
    face, at a vertex on the face in 2D and along an edge on the face in
    3D. The mesher cannot reach the minimum angle in such a corner and
    refines it in shells of elements down to about a quarter of the
    thickness of the wedge at the end of its shorter side, which is the
    ``size`` of the feature. The facet turns when the seeds on its two
    sides move along it: ``u_vec`` is that direction, away from the
    face."""
    q_min = np.radians(min_angle)
    if q_min <= 0:
        return []
    n_dim = pts.shape[1]
    features = []
    for r, region in enumerate(pmesh.regions):
        walls = []
        inner = []
        for f in region:
            neighs = pmesh.facet_neighbors[f]
            if min(neighs) < 0:
                axis, side = _misc.wall_axis_side(min(neighs))
                if per_axes[axis]:
                    walls.append((f, axis, side))
            else:
                inner.append(f)
        if not walls:
            continue
        kps = _region_points(pmesh, region)
        cen = pts[kps].mean(axis=0)
        for f_wall, axis, side in walls:
            wall_set = set(pmesh.facets[f_wall])
            for f in inner:
                shared = [kp for kp in pmesh.facets[f] if kp in wall_set]
                if len(shared) != n_dim - 1:
                    continue
                wedge = _wedge_geometry(pts, pmesh.facets[f_wall],
                                        pmesh.facets[f], shared, cen)
                if wedge is None:
                    continue
                angle, length, u_vec, where = wedge
                if not angle < q_min:
                    continue
                n_r = [n for n in pmesh.facet_neighbors[f] if n != r][0]
                if seed_nums[n_r] == seed_nums[r]:
                    continue
                features.append({
                    'kind': 'wedge',
                    'size': 0.25 * length * np.sin(angle),
                    'key': ('wedge', seed_nums[r], axis, side) +
                    tuple(np.round(where / scale, 6)),
                    'seeds': sorted({seed_nums[r], seed_nums[n_r]}),
                    'seed': seed_nums[r],
                    'neighbor': seed_nums[n_r],
                    'axis': axis,
                    'side': side,
                    'angle': angle,
                    'min_angle': q_min,
                    'u_vec': u_vec,
                    })
    return features


def _wedge_geometry(pts, wall_facet, facet, shared, cen):
    """Angle of the corner between a wall facet and a facet of a convex
    cell at their shared vertex (2D) or edge (3D), the shorter extent of
    the two facets from it, the unit vector along the facet away from the
    wall, and the location of the corner; None if degenerate."""
    if len(shared) == 1:
        kp = shared[0]
        a_vec = pts[[k for k in wall_facet if k != kp][0]] - pts[kp]
        b_vec = pts[[k for k in facet if k != kp][0]] - pts[kp]
        len_a = np.linalg.norm(a_vec)
        len_b = np.linalg.norm(b_vec)
        if len_a == 0 or len_b == 0:
            return None
        cos_ang = np.dot(a_vec, b_vec) / (len_a * len_b)
        angle = np.arccos(np.clip(cos_ang, -1, 1))
        return angle, min(len_a, len_b), b_vec / len_b, pts[kp]

    e_pt = pts[shared[0]]
    e_vec = pts[shared[1]] - e_pt
    e_len = np.linalg.norm(e_vec)
    if e_len == 0:
        return None
    e_vec /= e_len
    normals = []
    extents = []
    for loop in (wall_facet, facet):
        loop_pts = pts[loop]
        n_vec = np.zeros(3)
        for i in range(len(loop_pts)):
            n_vec += np.cross(loop_pts[i], loop_pts[(i + 1) % len(loop_pts)])
        n_len = np.linalg.norm(n_vec)
        if n_len == 0:
            return None
        n_vec /= n_len
        if np.dot(n_vec, loop_pts.mean(axis=0) - cen) < 0:
            n_vec = -n_vec  # outward from the cell
        normals.append(n_vec)
        rel = loop_pts - e_pt
        rel -= np.outer(rel @ e_vec, e_vec)
        extents.append(np.max(np.linalg.norm(rel, axis=1)))
    # the interior dihedral angle, from the outward normals
    cos_ang = np.dot(normals[0], normals[1])
    angle = np.pi - np.arccos(np.clip(cos_ang, -1, 1))
    u_vec = pts[facet].mean(axis=0) - e_pt
    u_vec -= np.dot(u_vec, e_vec) * e_vec
    u_len = np.linalg.norm(u_vec)
    if u_len == 0 or min(extents) == 0:
        return None
    where = 0.5 * (pts[shared[0]] + pts[shared[1]])
    return angle, min(extents), u_vec / u_len, where


def _select_target(features, margin, stuck):
    """The shortest feature, or the thinnest piece or corner under the
    margin, that is not stuck; None when there is no such target."""
    sizes = np.array([f['size'] for f in features])
    order = np.argsort(sizes, kind='stable')
    cands = [order[0]]
    cands += [i for i in order if features[i]['kind'] in ('piece', 'wedge')
              and sizes[i] < margin]
    for i in cands:
        if features[i]['key'] not in stuck:
            return features[i]
    return None


def _trial_steps(target, seedlist, n_attempts, margin, dom_lims, per_axes,
                 n_dim):
    """Displacements of the seeds around the target for one trial.

    Edge: each seed is moved normal to the edge by a random fraction of
    0.1 times its equivalent radius. Piece: the first trial pushes the
    cell of the piece so that the piece reaches the margin, the second
    retracts it so that the cell ends a margin inside the face (the cell
    boundary moves by about half the displacement of the seed), and the
    others move the seed of the piece normal to the face, and the seeds
    of its neighbors along their lines to the seed of the piece (normal to
    their facets with it), by random fractions of the margin or of 0.1
    times their equivalent radii, whichever is larger. Wedge: the facet
    turns with the line between the seeds on its two sides, so the first
    trial moves them along the facet, in opposite directions, by the
    amount that opens the corner to the minimum angle plus 2 degrees, and
    the others move them along the facet by random fractions as above.
    """
    steps = {}
    if target['kind'] == 'wedge':
        u_vec = target['u_vec']
        seed_w, seed_n = target['seed'], target['neighbor']
        if n_attempts == 0:
            pos_w = np.array(seedlist[seed_w].position, dtype='float')
            pos_n = np.array(seedlist[seed_n].position, dtype='float')
            if dom_lims is not None:
                pos_n = _nearest_image(pos_n.reshape(1, -1), pos_w,
                                       dom_lims, per_axes)[0]
            phi = target['min_angle'] + np.radians(2) - target['angle']
            delta = 0.5 * phi * np.linalg.norm(pos_n - pos_w)
            # the corner opens when the facet turns away from the wall:
            # the line between the seeds turns towards the facet
            return {seed_w: delta * u_vec, seed_n: -delta * u_vec}
    if target['kind'] == 'piece':
        axis = target['axis']
        grow = np.zeros(n_dim)
        grow[axis] = 1.0 if target['side'] == 0 else -1.0
        thickness = target['size']
        if n_attempts == 0 and margin > thickness:
            steps[target['seed']] = 2.2 * (margin - thickness) * grow
            return steps
        if n_attempts == 1:
            steps[target['seed']] = -2 * (thickness + margin) * grow
            return steps
        pos_piece = np.array(seedlist[target['seed']].position,
                             dtype='float')

    for seed_num in target['seeds']:
        seed = seedlist[seed_num]
        pos = np.array(seed.position, dtype='float')
        if n_dim == 2:
            r_eq = np.sqrt(seed.volume / np.pi)
        else:
            r_eq = np.cbrt(3 * seed.volume / (4 * np.pi))
        step_max = 0.1 * r_eq
        if target['kind'] == 'wedge':
            step_max = max(step_max, margin)
            u_vec = target['u_vec']
        elif target['kind'] == 'piece':
            step_max = max(step_max, margin)
            if seed_num == target['seed']:
                u_vec = grow
            else:
                ref = _nearest_image(pos_piece.reshape(1, -1), pos,
                                     dom_lims, per_axes)[0]
                u_vec = ref - pos
                if np.linalg.norm(u_vec) == 0:
                    continue
                u_vec /= np.linalg.norm(u_vec)
        else:
            edge_pts = target['pts']
            if dom_lims is not None:
                edge_pts = _nearest_image(edge_pts, pos, dom_lims, per_axes)
            with np.errstate(divide='ignore', invalid='ignore'):
                u_vec = _point_line_vec(pos, edge_pts)
            if not np.all(np.isfinite(u_vec)):
                continue
        step_frac = 2 * np.random.rand() - 1  # [-1, 1]
        steps[seed_num] = step_frac * step_max * u_vec
    return steps


def _nearest_image(pts, ref, dom_lims, per_axes):
    """Translate ``pts`` (as a whole) by periods so that their center is
    closest to ``ref``."""
    pts = np.array(pts, dtype='float')
    cen = pts.mean(axis=0)
    shift = np.zeros(len(ref))
    for axis, flag in enumerate(per_axes):
        if flag:
            length = dom_lims[axis][1] - dom_lims[axis][0]
            shift[axis] = length * np.round((ref[axis] - cen[axis]) / length)
    return pts + shift


def _accept_trial(new_features, old_features, tol):
    """Whether a trial improves the features that it changes: the shortest
    feature that it creates is longer than the shortest one that it
    removes (the features are matched by their keys, and a removed and a
    created feature of the same size cancel out)."""
    old_keys = set([f['key'] for f in old_features])
    new_keys = set([f['key'] for f in new_features])
    removed = sorted([f['size'] for f in old_features
                      if f['key'] not in new_keys])
    added = sorted([f['size'] for f in new_features
                    if f['key'] not in old_keys])
    i = j = 0
    kept_removed = []
    kept_added = []
    while i < len(removed) and j < len(added):
        if abs(removed[i] - added[j]) <= tol:
            i += 1
            j += 1
        elif removed[i] < added[j]:
            kept_removed.append(removed[i])
            i += 1
        else:
            kept_added.append(added[j])
            j += 1
    kept_removed += removed[i:]
    kept_added += added[j:]
    if not kept_removed:
        return False
    if not kept_added:
        return True
    return kept_added[0] > kept_removed[0] + tol


def _target_string(target, n_attempts, n_iter, n_kp_space, n_iter_space):
    if target['kind'] == 'edge':
        kp_fmt = '{0:' + str(n_kp_space) + 'd}'
        s = 'min length: {0:.3e} | edge: '.format(target['size'])
        s += ', '.join([kp_fmt.format(kp) for kp in target['kps']])
    elif target['kind'] == 'piece':
        face = 'xyz'[target['axis']] + '-+'[target['side']]
        s = 'thickness: {0:.3e} | piece: seed {1:d}, face {2}'
        s = s.format(target['size'], target['seed'], face)
    else:
        face = 'xyz'[target['axis']] + '-+'[target['side']]
        s = 'corner: {0:.3e} | wedge: seed {1:d}, face {2}, {3:.1f} deg'
        s = s.format(target['size'], target['seed'], face,
                     np.degrees(target['angle']))
    s += ' | n iter: {0:' + str(n_iter_space) + 'd} / {1:d}'
    return s.format(n_attempts, n_iter)
