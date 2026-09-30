"""Triangle/Tetrahedron Meshing

This module contains the class definition for the TriMesh class.

"""
# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #


from __future__ import division
from __future__ import print_function

import itertools

import meshpy.tet
import meshpy.triangle
import numpy as np
import pygmsh as pg
from matplotlib import collections
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

from microstructpy import _misc

__all__ = ['TriMesh']
__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# TriMesh Class                                                               #
#                                                                             #
# --------------------------------------------------------------------------- #
class TriMesh(object):
    """Triangle/Tetrahedron mesh.

    The TriMesh class contains the points, facets, and elements in a triangle/
    tetrahedron mesh, also called an unstructured grid.

    The points attribute is an Nx2 or Nx3 list of points in the mesh.
    The elements attribute contains the Nx3 or Nx4 list of the points at
    the corners of each triangle/tetrahedron. A list of facets can also be
    included, though it is optional and does not need to include every facet
    in the mesh. Attributes can also be assigned to the elements and facets,
    though they are also optional.

    Args:
        points (list, numpy.ndarray): List of coordinates in the mesh.
        elements (list, numpy.ndarray): List of indices of the points at
            the corners of each element. The shape should be Nx3 in 2D or
            Nx4 in 3D.
        element_attributes (list, numpy.ndarray): *(optional)* A number
            associated with each element.
            Defaults to None.
        facets (list, numpy.ndarray): *(optional)* A list of facets in the
            mesh. The shape should be Nx2 in 2D or Nx3 in 3D.
            Defaults to None.
        facet_attributes (list, numpy.ndarray): *(optional)* A number
            associated with each facet.
            Defaults to None.

    """
    # ----------------------------------------------------------------------- #
    # Constructors                                                            #
    # ----------------------------------------------------------------------- #
    def __init__(self, points, elements, element_attributes=None, facets=None,
                 facet_attributes=None, periodic_axes=None,
                 periodic_nodes=None, periodic_facets=None):
        self.points = points
        self.elements = elements
        self.element_attributes = element_attributes
        self.facets = facets
        self.facet_attributes = facet_attributes

        # Periodicity: flags per axis, and the pairs of (low face, high face)
        # nodes and facets that are periodic images of each other, per axis
        self.periodic_axes = periodic_axes
        self.periodic_nodes = periodic_nodes
        self.periodic_facets = periodic_facets

    @classmethod
    def from_file(cls, filename):
        """Read TriMesh from file.

        This function reads in a triangular mesh from a file and creates an
        instance from that file. Currently the only supported file type
        is the output from :meth:`.write` with the ``format='str'`` option.

        Args:
            filename (str): Name of file to read from.

        Returns:
            TriMesh: An instance of the class.

        """
        with open(filename, 'r') as file:
            stage = 0
            pts = []
            elems = []
            elem_atts = []
            facets = []
            facet_atts = []

            n_eas = 0
            n_facets = 0
            n_fas = 0
            per_axes = None
            per_nodes = []
            per_fts = []
            for line in file.readlines():
                if 'Periodic Axes'.lower() in line.lower():
                    stage = 'periodic axes'
                elif 'Periodic Nodes'.lower() in line.lower():
                    stage = 'periodic nodes'
                elif 'Periodic Facets'.lower() in line.lower():
                    stage = 'periodic facets'
                elif 'Mesh Points'.lower() in line.lower():
                    n_pts = int(line.split(':')[1])
                    stage = 'points'
                elif 'Mesh Elements'.lower() in line.lower():
                    n_elems = int(line.split(':')[1])
                    stage = 'elements'
                elif 'Element Attributes'.lower() in line.lower():
                    n_eas = int(line.split(':')[1])
                    stage = 'element attributes'
                elif 'Facets'.lower() in line.lower():
                    n_facets = int(line.split(':')[1])
                    stage = 'facets'
                elif 'Facet Attributes'.lower() in line.lower():
                    n_fas = int(line.split(':')[1])
                    stage = 'facet attributes'
                else:
                    if stage == 'points':
                        pts.append([float(x) for x in line.split(',')])
                    elif stage == 'elements':
                        elems.append([int(kp) for kp in line.split(',')])
                    elif stage == 'element attributes':
                        elem_atts.append(_misc.from_str(line))
                    elif stage == 'facets':
                        if n_facets > 0:
                            facets.append([int(kp) for kp in line.split(',')])
                    elif stage == 'facet attributes':
                        if n_fas > 0:
                            facet_atts.append(_misc.from_str(line))
                    elif stage == 'periodic axes':
                        per_axes = [bool(int(f)) for f in line.split(',')]
                    elif stage == 'periodic nodes':
                        per_nodes.append([int(n) for n in line.split(',')])
                    elif stage == 'periodic facets':
                        per_fts.append([int(n) for n in line.split(',')])
                    else:
                        pass

        # check the inputs
        assert len(pts) == n_pts
        assert len(elems) == n_elems
        assert len(elem_atts) == n_eas
        assert len(facets) == n_facets
        assert len(facet_atts) == n_fas

        per_node_pairs = None
        per_facet_pairs = None
        if per_axes is not None:
            per_node_pairs = {ax: [] for ax, f in enumerate(per_axes) if f}
            per_facet_pairs = {ax: [] for ax, f in enumerate(per_axes) if f}
            for ax, lo, hi in per_nodes:
                per_node_pairs[ax].append((lo, hi))
            for ax, lo, hi in per_fts:
                per_facet_pairs[ax].append((lo, hi))

        return cls(pts, elems, elem_atts, facets, facet_atts,
                   periodic_axes=per_axes, periodic_nodes=per_node_pairs,
                   periodic_facets=per_facet_pairs)

    @classmethod
    def from_polymesh(cls, polymesh, phases=None, mesher='Triangle/Tetgen',
                      min_angle=0, max_volume=float('inf'),
                      max_edge_length=float('inf'), mesh_size=float('inf')):
        """Create TriMesh from PolyMesh.

        This constuctor creates a triangle/tetrahedron mesh from a polygon
        mesh (:class:`.PolyMesh`). Polygons of the same seed number are
        merged and the element attribute is set to the seed number it is
        within. The facets between seeds are saved to the mesh and the index
        of the facet is stored in the facet attributes.

        Since the PolyMesh can include phase numbers for each region,
        additional information about the phases can be included as an input.
        The "phases" input should be a list of material phase dictionaries,
        formatted according to the :ref:`phase_dict_guide` guide.

        The minimum angle, maximum volume, and maximum edge length options
        provide quality controls for the mesh. The phase type option can take
        one of several values, described below.

        * **crystalline**: granular, solid
        * **amorphous**: glass, matrix
        * **void**: crack, hole

        The **crystalline** option creates a mesh where cells of the same seed
        number are merged, but cells are not merged across seeds. _This is
        the default material type._

        The **amorphous** option creates a mesh where cells of the same
        phase number are merged to create an amorphous region in the mesh.

        Finally, the **void** option will merge neighboring void cells and
        treat them as holes in the mesh.

        Args:
            polymesh (PolyMesh): A polygon/polyhedron mesh.
            phases (list): *(optional)* A list of dictionaries containing
                options for each phase.
                Default is
                ``{'material_type': 'solid', 'max_volume': float('inf')}``.
            mesher (str): {'Triangle/TetGen' | 'Triangle'  | 'TetGen' | 'gmsh'}
                specify the mesh generator. Default is 'Triangle/TetGen'.
            min_angle (float): The minimum interior angle, in degrees, of an
                element. This option is used with Triangle or TetGen and in 3D
                is the minimum *dihedral* angle. Defaults to 0.
            max_volume (float): The default maximum cell volume, used if one
                is not set for each phase. This option is used with Triangle or
                TetGen. Defaults to infinity, which turns off this control.
            max_edge_length (float): The maximum edge length of elements
                along grain boundaries: of the segments in 2D and of the
                triangles on the facets in 3D. This option is used with
                Triangle/TetGen and gmsh. Defaults to infinity, which turns
                off this control.
            mesh_size (float): The target size of the mesh elements. This
                option is used with gmsh. Default is infinity, whihch turns off
                this control.

        Note:
            The facets of the mesh are listed with their nodes in ascending
            order and in lexicographic order, whatever the mesher: the order
            in which Triangle and TetGen list the edges/faces of a mesh
            varies from one run to the next, and the sorted facets make the
            mesh, and its files, reproducible.

        """
        # A periodic polygon mesh gives a periodic triangular mesh: the
        # nodes on opposite periodic faces are images of each other
        per_axes = getattr(polymesh, 'periodic_axes', None)
        periodic = per_axes is not None and any(per_axes)

        key = str(mesher).lower().strip()
        if key in ('triangle/tetgen', 'triangle', 'tetgen'):
            tri_args = _call_meshpy(polymesh, phases, min_angle, max_volume,
                                    max_edge_length, periodic=periodic)
        elif key == 'gmsh':
            if periodic:
                e_str = 'Periodic meshes are not supported with gmsh; use '
                e_str += 'the Triangle/TetGen mesher.'
                raise NotImplementedError(e_str)
            tri_args = _call_gmsh(polymesh, phases, mesh_size, max_edge_length)
        else:
            e_str = 'Unknown mesher ' + repr(mesher) + '. Options are '
            e_str += "'Triangle/TetGen', 'Triangle', 'TetGen', and 'gmsh'."
            raise ValueError(e_str)

        tri_pts, tri_elems, tri_e_atts, tri_f, tri_fa = tri_args
        tri_f, tri_fa = _sorted_facets(tri_f, tri_fa)
        mesh = cls(tri_pts, tri_elems, tri_e_atts, tri_f, tri_fa)
        if periodic:
            dom_lims = _misc.periodic_bounds(polymesh.points, per_axes)
            mesh._set_periodic_pairs(per_axes, dom_lims)
        return mesh

    # ----------------------------------------------------------------------- #
    # Periodicity                                                             #
    # ----------------------------------------------------------------------- #
    def _set_periodic_pairs(self, per_axes, dom_lims):
        """Pair the nodes and facets on opposite periodic faces.

        The nodes on the lower face of each periodic axis are matched with
        their images on the upper face and snapped to exact translates;
        the facets on the faces are paired likewise. The results are stored
        in ``periodic_axes``, ``periodic_nodes`` (dict: axis -> list of
        (lower, upper) node numbers) and ``periodic_facets`` (dict: axis ->
        list of (lower, upper) facet numbers).

        Raises:
            ValueError: If a node or facet on a periodic face has no image.

        """
        pts, per_nodes, per_facets = _misc.pair_periodic_mesh(
            self.points, self.facets, per_axes, dom_lims)
        self.points = pts
        self.periodic_axes = [bool(f) for f in per_axes]
        self.periodic_nodes = per_nodes
        self.periodic_facets = per_facets

    # ----------------------------------------------------------------------- #
    # String and Representation Functions                                     #
    # ----------------------------------------------------------------------- #
    def __str__(self):
        nv = len(self.points)

        # Points are written with the shortest representation that
        # round-trips exactly (repr of a float), so that a mesh read back
        # from the file is identical to the one written.
        str_str = 'Mesh Points: ' + str(nv) + '\n'
        str_str += ''.join(['\t' + ', '.join([repr(float(x)) for x in p]) +
                            '\n' for p in self.points])

        str_str += 'Mesh Elements: ' + str(len(self.elements)) + '\n'
        str_str += '\n'.join(['\t' + ', '.join([str(int(kp)) for kp in e])
                              for e in self.elements])

        # Optional attributes and facets: only write the sections that exist,
        # so that the file never contains a dangling header.
        if self.element_attributes is not None:
            str_str += '\nElement Attributes: '
            str_str += str(len(self.element_attributes)) + '\n'
            str_str += '\n'.join(['\t' + str(a) for a in
                                  self.element_attributes])

        if self.facets is not None:
            str_str += '\nFacets: ' + str(len(self.facets)) + '\n'
            str_str += '\n'.join(['\t' + ', '.join([str(int(kp)) for kp in
                                                    f]) for f in self.facets])

        if self.facet_attributes is not None:
            str_str += '\nFacet Attributes: '
            str_str += str(len(self.facet_attributes)) + '\n'
            str_str += '\n'.join(['\t' + str(a) for a in
                                  self.facet_attributes])

        if self.periodic_axes is not None and any(self.periodic_axes):
            flags = [int(bool(f)) for f in self.periodic_axes]
            str_str += '\nPeriodic Axes: ' + str(len(flags)) + '\n'
            str_str += '\t' + ', '.join([str(f) for f in flags])
            for name, pairs in (('Periodic Nodes', self.periodic_nodes),
                                ('Periodic Facets', self.periodic_facets)):
                rows = [(ax, lo, hi) for ax in sorted(pairs or {})
                        for lo, hi in pairs[ax]]
                str_str += '\n' + name + ': ' + str(len(rows))
                str_str += ''.join(['\n\t' + ', '.join([str(n) for n in row])
                                    for row in rows])

        return str_str

    def __repr__(self):
        repr_str = 'TriMesh('
        repr_str += ', '.join([repr(v) for v in (self.points, self.elements,
                               self.element_attributes, self.facets,
                               self.facet_attributes)])
        repr_str += ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Write Function                                                          #
    # ----------------------------------------------------------------------- #
    def write(self, filename, format='txt', seeds=None, polymesh=None):
        """Write mesh to file.

        This function writes the contents of the mesh to a file.
        The format options are 'abaqus', 'tet/tri', 'txt', and 'vtk'.
        See the :ref:`s_tri_file_io` section of the :ref:`c_file_formats`
        guide for more details on these formats.

        Args:
            filename (str): The name of the file to write. In the cases of
                TetGen/Triangle, this is the basename of the files.
            format (str): {'abaqus' | 'tet/tri' | 'txt' | 'vtk'}
                *(optional)* The format of the output file.
                Default is 'txt'.
            seeds (SeedList): *(optional)* List of seeds. If given, VTK files
                will also include the phase number of of each element in the
                mesh. This assumes the ``element_attributes``
                field contains the seed number of each element.
            polymesh (PolyMesh): *(optional)* Polygonal mesh used for
                generating the triangular mesh. If given, will add surface
                unions to Abaqus files - for easier specification of
                boundary conditions.

        """  # NOQA: E501
        fmt = format.lower()
        if fmt in ('abaqus', 'tet/tri', 'vtk'):
            # These formats infer the element type from the number of nodes
            # per element, so make sure the elements are simplices.
            n_dim = len(self.points[0])
            n_kp = len(self.elements[0])
            if n_kp != n_dim + 1:
                e_str = 'TriMesh elements must be triangles/tetrahedra with '
                e_str += str(n_dim + 1) + ' nodes each to write the '
                e_str += repr(format) + ' format, but the elements have '
                e_str += str(n_kp) + ' nodes.'
                raise ValueError(e_str)

        if fmt == 'abaqus':
            # write top matter
            abaqus = '*Heading\n'
            abaqus += '** Job name: microstructure '
            abaqus += 'Model name: microstructure_model\n'
            abaqus += '** Generated by: MicroStructPy\n'

            # write parts
            abaqus += '**\n** PARTS\n**\n'
            abaqus += '*Part, name=Part-1\n'

            abaqus += '*Node\n'
            abaqus += ''.join([str(i + 1) + ''.join([', ' + str(x) for x in
                               pt]) + '\n' for i, pt in
                               enumerate(self.points)])

            n_dim = len(self.points[0])
            elem_type = {2: 'CPS3', 3: 'C3D4'}[n_dim]

            abaqus += '*Element, type=' + elem_type + '\n'
            abaqus += ''.join([str(i + 1) + ''.join([', ' + str(int(kp) + 1)
                                                     for kp in elm]) + '\n' for
                               i, elm in enumerate(self.elements)])

            # Node sets - periodic faces (in paired order)
            abaqus += _abaqus_periodic_nsets(self)

            # Element sets - seed number
            elset_n_per = 16
            if self.element_attributes is None:
                elem_atts = np.array([])
            else:
                elem_atts = np.array(self.element_attributes)
            for att in np.unique(elem_atts):
                elset_name = 'Set-E-Seed-' + str(att)
                elset_str = '*Elset, elset=' + elset_name + '\n'
                elem_groups = [[]]
                for elem_ind, elem_att in enumerate(elem_atts):
                    if ~np.isclose(elem_att, att):
                        continue
                    if len(elem_groups[-1]) >= elset_n_per:
                        elem_groups.append([])
                    elem_groups[-1].append(elem_ind + 1)
                for group in elem_groups:
                    elset_str += ','.join([str(i) for i in group])
                    elset_str += '\n'

                abaqus += elset_str

            # Element Sets - phase number
            if seeds is not None:
                phase_nums = np.array([seed.phase for seed in seeds])
                for phase_num in np.unique(phase_nums):
                    mask = phase_nums == phase_num
                    seed_nums = np.nonzero(mask)[0]

                    elset_name = 'Set-E-Material-' + str(phase_num)
                    elset_str = '*Elset, elset=' + elset_name + '\n'
                    groups = [[]]
                    for seed_num in seed_nums:
                        if seed_num not in elem_atts:
                            continue
                        if len(groups[-1]) >= elset_n_per:
                            groups.append([])
                        seed_elset_name = 'Set-E-Seed-' + str(seed_num)
                        groups[-1].append(seed_elset_name)
                    for group in groups:
                        elset_str += ','.join(group)
                        elset_str += '\n'
                    abaqus += elset_str

            # Surfaces - Exterior and Interior
            defined_surfs = set()
            has_facets = self.facets is not None and len(self.facets) > 0
            if has_facets and self.facet_attributes is not None:
                facets = np.array(self.facets)
                facet_atts = np.array(self.facet_attributes)

                face_ids = {2: [2, 3, 1], 3: [3, 4, 2, 1]}[n_dim]

                for att in np.unique(facet_atts):
                    facet_name = 'Surface-' + str(att)
                    surf_str = '*Surface, name=' + facet_name
                    surf_str += ', type=element\n'

                    att_facets = facets[facet_atts == att]
                    for facet in att_facets:
                        mask = np.isin(self.elements, facet)
                        n_match = mask.astype('int').sum(axis=1)
                        i_elem = np.argmax(n_match)
                        elem_id = i_elem + 1

                        i_missing = np.argmin(mask[i_elem])
                        face_id = face_ids[i_missing]

                        surf_str += str(elem_id) + ', S' + str(face_id) + '\n'

                    abaqus += surf_str
                    defined_surfs.add(int(att))

            # Surfaces - Exterior (unions of the surfaces on each domain face)
            if polymesh is not None:
                abaqus += _abaqus_exterior_unions(polymesh, defined_surfs)

            # End Part
            abaqus += '*End Part\n\n'

            # Assembly
            abaqus += '**\n'
            abaqus += '** ASSEMBLY\n'
            abaqus += '**\n'

            abaqus += '*Assembly, name=assembly\n'
            abaqus += '**\n'

            # Instances
            abaqus += '*Instance, name=I-Part-1, part=Part-1\n'
            abaqus += '*End Instance\n'

            # End Assembly
            abaqus += '**\n'
            abaqus += '*End Assembly\n'

            with open(filename, 'w') as file:
                file.write(abaqus)
        elif fmt in ('str', 'txt'):
            with open(filename, 'w') as file:
                file.write(str(self) + '\n')

        elif fmt == 'tet/tri':
            n_pts, n_dim = np.array(self.points).shape
            elem_arr = np.array(self.elements)
            n_ele, n_kp = elem_arr.shape

            # Boundary markers: the faces of the elements that belong to a
            # single element are on the boundary of the mesh.
            face_counts = {}
            for elem in elem_arr:
                for i in range(n_kp):
                    key = tuple(sorted(np.delete(elem, i).tolist()))
                    face_counts[key] = face_counts.get(key, 0) + 1

            bnd_mkrs = np.full(n_pts, 0, dtype='int')
            for key, count in face_counts.items():
                if count == 1:
                    bnd_mkrs[list(key)] = 1

            # write vertices
            nodes = ' '.join([str(n) for n in (n_pts, n_dim, 0, 1)]) + '\n'
            nodes += ''.join([str(i) + ''.join([' ' + str(x) for x in pt]) +
                              ' ' + str(bnd_mkrs[i]) + '\n' for i, pt in
                              enumerate(self.points)])

            with open(filename + '.node', 'w') as file:
                file.write(nodes)

            # write elements
            is_att = self.element_attributes is not None
            n_att = int(is_att)
            eles = ' '.join([str(n) for n in (n_ele, n_kp, n_att)]) + '\n'
            for i, simplex in enumerate(self.elements):
                e_str = str(i) + ''.join([' ' + str(int(kp)) for kp in
                                          simplex])
                if is_att:
                    e_str += ' ' + str(self.element_attributes[i])
                e_str += '\n'
                eles += e_str

            with open(filename + '.ele', 'w') as file:
                file.write(eles)

            # Write edges/faces
            # Format: '<# of edges/faces> <# of boundary markers (0 or 1)>'
            # followed by '<index> <node> <node> [<node>] <marker>' lines.
            if self.facets is not None and len(self.facets) > 0:
                ext = {2: '.edge', 3: '.face'}[n_dim]

                n_facet = len(self.facets)
                edge = str(n_facet) + ' 1\n'
                for i, facet in enumerate(self.facets):
                    key = tuple(sorted([int(kp) for kp in facet]))
                    mkr = int(face_counts.get(key, 0) == 1)
                    edge += str(i) + ''.join([' ' + str(k) for k in facet])
                    edge += ' ' + str(mkr) + '\n'
                with open(filename + ext, 'w') as file:
                    file.write(edge)

        elif fmt == 'vtk':
            n_kp = len(self.elements[0])
            mesh_type = {3: 'Triangular', 4: 'Tetrahedral'}[n_kp]
            pt_fmt = '{: f} {: f} {: f}\n'
            # write heading
            vtk = '# vtk DataFile Version 2.0\n'
            vtk += '{} mesh\n'.format(mesh_type)
            vtk += 'ASCII\n'
            vtk += 'DATASET UNSTRUCTURED_GRID\n'

            # Write points
            vtk += 'POINTS ' + str(len(self.points)) + ' float\n'
            if len(self.points[0]) == 2:
                vtk += ''.join([pt_fmt.format(x, y, 0) for x, y in
                                self.points])
            else:
                vtk += ''.join([pt_fmt.format(x, y, z) for x, y, z in
                                self.points])

            # write elements
            n_elem = len(self.elements)
            cell_fmt = str(n_kp) + n_kp * ' {}' + '\n'
            cell_sz = (1 + n_kp) * n_elem
            vtk += '\nCELLS ' + str(n_elem) + ' ' + str(cell_sz) + '\n'
            vtk += ''.join([cell_fmt.format(*el) for el in self.elements])

            # write cell type
            vtk += '\nCELL_TYPES ' + str(n_elem) + '\n'
            cell_type = {3: '5', 4: '10'}[n_kp]
            vtk += ''.join(n_elem * [cell_type + '\n'])

            # write element attributes
            if self.element_attributes is not None:
                try:
                    int(self.element_attributes[0])
                    att_type = 'int'
                except TypeError:
                    att_type = 'float'

                vtk += '\nCELL_DATA ' + str(n_elem) + '\n'
                vtk += 'SCALARS element_attributes ' + att_type + ' 1 \n'
                vtk += 'LOOKUP_TABLE element_attributes\n'
                vtk += ''.join([str(a) + '\n' for a in
                                self.element_attributes])

                # Write phase numbers
                if seeds is not None:
                    vtk += '\nSCALARS phase_numbers int 1 \n'
                    vtk += 'LOOKUP_TABLE phase_numbers\n'
                    vtk += ''.join([str(seeds[a].phase) + '\n' for a in
                                    self.element_attributes])

            with open(filename, 'w') as file:
                file.write(vtk)

        else:
            e_str = 'Cannot write file type ' + str(format) + ' yet.'
            raise NotImplementedError(e_str)

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, index_by='element', material=[], loc=0, **kwargs):
        """Plot the mesh.

        This method plots the mesh using matplotlib.
        In 2D, this creates a :class:`matplotlib.collections.PolyCollection`
        and adds it to the current axes.
        In 3D, it creates a
        :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection` and
        adds it to the current axes.
        The keyword arguments are passed though to matplotlib.

        Args:
            index_by (str): *(optional)* {'element' | 'attribute'}
                Flag for indexing into the other arrays passed into the
                function. For example,
                ``plot(index_by='attribute', color=['blue', 'red'])`` will plot
                the elements with ``element_attribute`` equal to 0 in blue, and
                elements with ``element_attribute`` equal to 1 in red.
                Note that in 3D the facets are plotted instead of the elements,
                so kwarg lists must be based on ``facets`` and
                ``facet_attributes``. Defaults to 'element'.
            material (list): *(optional)* Names of material phases. One entry
                per material phase (the ``index_by`` argument is ignored).
                If this argument is set, a legend is added to the plot with
                one entry per material. Note that the ``element_attributes``
                in 2D or the ``facet_attributes`` in 3D must be the material
                numbers for the legend to be formatted properly.
            loc (int or str): *(optional)* The location of the legend,
                if 'material' is specified. This argument is passed directly
                through to :func:`matplotlib.pyplot.legend`. Defaults to 0,
                which is 'best' in matplotlib.
            **kwargs: Keyword arguments that are passed through to matplotlib.

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
            _plot_2d(ax, self, index_by, **kwargs)
        else:
            if n_obj > 0:
                zlim = ax.get_zlim()
            else:
                zlim = [float('inf'), -float('inf')]

            xy = [np.array([self.points[kp] for kp in f]) for f in self.facets]

            plt_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, (list, np.ndarray)):
                    plt_value = []
                    for f_num, f_att in enumerate(self.facet_attributes):
                        if index_by == 'element':
                            ind = f_num
                        elif index_by == 'attribute':
                            ind = int(f_att)
                        else:
                            e_str = 'Cannot index by {}.'.format(index_by)
                            raise ValueError(e_str)
                        if ind < len(value):
                            v = value[ind]
                        else:
                            v = 'none'
                        plt_value.append(v)
                else:
                    plt_value = value
                plt_kwargs[key] = plt_value
            pc = Poly3DCollection(xy, **plt_kwargs)
            ax.add_collection(pc)

        # Add legend
        if material and index_by == 'attribute':
            p_kwargs = [{'label': m} for m in material]
            for key, value in kwargs.items():
                if not isinstance(value, (list, np.ndarray)):
                    for kws in p_kwargs:
                        kws[key] = value

                for i, m in enumerate(material):
                    if isinstance(value, (list, np.ndarray)):
                        p_kwargs[i][key] = value[i]
                    else:
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


# --------------------------------------------------------------------------- #
#                                                                             #
# RasterMesh Class                                                            #
#                                                                             #
# --------------------------------------------------------------------------- #
class RasterMesh(TriMesh):
    """Raster mesh.

    The RasterMesh class contains the points and elements in a raster mesh,
    also called an regular grid.

    The points attribute is an Nx2 or Nx3 list of points in the mesh.
    The elements attribute contains the Nx4 or Nx8 list of the points at
    the corners of each pixel/voxel. A list of facets can also be
    included, though it is optional and does not need to include every facet
    in the mesh. Attributes can also be assigned to the elements and facets,
    though they are also optional.

    Args:
        points (list, numpy.ndarray): List of coordinates in the mesh.
        elements (list, numpy.ndarray): List of indices of the points at
            the corners of each element. The shape should be Nx3 in 2D or
            Nx4 in 3D.
        element_attributes (list, numpy.ndarray): *(optional)* A number
            associated with each element.
            Defaults to None.
        facets (list, numpy.ndarray): *(optional)* A list of facets in the
            mesh. The shape should be Nx2 in 2D or Nx3 in 3D.
            Defaults to None.
        facet_attributes (list, numpy.ndarray): *(optional)* A number
            associated with each facet.
            Defaults to None.

    """
    # ----------------------------------------------------------------------- #
    # Constructors                                                            #
    # ----------------------------------------------------------------------- #
    # Inherited from TriMesh

    @classmethod
    def from_polymesh(cls, polymesh, mesh_size, phases=None):
        """Create RasterMesh from PolyMesh.

        This constuctor creates a raster mesh from a polygon
        mesh (:class:`.PolyMesh`). Polygons of the same seed number are
        merged and the element attribute is set to the seed number it is
        within. The facets between seeds are saved to the mesh and the index
        of the facet is stored in the facet attributes.

        Since the PolyMesh can include phase numbers for each region,
        additional information about the phases can be included as an input.
        The "phases" input should be a list of material phase dictionaries,
        formatted according to the :ref:`phase_dict_guide` guide.

        The mesh_size option determines the side length of each pixel/voxel.
        Element attributes are sampled at the center of each pixel/voxel.
        If an edge of a domain is not an integer multiple of the mesh_size, it
        will be clipped. For example, if mesh_size is 3 and an edge has
        bounds [0, 11], the sides of the pixels will be at 0, 3, 6, and 9 while
        the centers of the pixels will be at 1.5, 4.5, 7.5.

        The phase type option can take one of several values, described below.

        * **crystalline**: granular, solid
        * **amorphous**: glass, matrix
        * **void**: crack, hole

        The **crystalline** option creates a mesh where cells of the same seed
        number are merged, but cells are not merged across seeds. _This is
        the default material type._

        The **amorphous** option creates a mesh where cells of the same
        phase number are merged to create an amorphous region in the mesh.

        Finally, the **void** option will merge neighboring void cells and
        treat them as holes in the mesh.

        Args:
            polymesh (PolyMesh): A polygon/polyhedron mesh.
            mesh_size (float): The side length of each pixel/voxel.
            phases (list): *(optional)* A list of dictionaries containing
                options for each phase.
                Default is
                ``{'material_type': 'solid', 'max_volume': float('inf')}``.

        """
        if phases is None:
            phases = _default_phases(polymesh)

        # 1. Create node and element grids
        p_pts = np.array(polymesh.points)
        mins = p_pts.min(axis=0)
        maxs = p_pts.max(axis=0)
        lens = (maxs - mins) * (1 + 1e-9)
        sides = [lb + np.arange(0, dlen, mesh_size) for lb, dlen in
                 zip(mins, lens)]

        # A periodic polymesh gives a periodic raster mesh: the grid must
        # then reach the opposite faces exactly
        per_axes = getattr(polymesh, 'periodic_axes', None)
        periodic = per_axes is not None and any(per_axes)
        if periodic:
            for axis, flag in enumerate(per_axes):
                if not flag:
                    continue
                length = maxs[axis] - mins[axis]
                n_pix = int(round(length / mesh_size))
                misfit = abs(n_pix * mesh_size - length)
                if n_pix < 1 or misfit > 1e-8 * length:
                    e_str = 'The mesh size of a periodic raster mesh must '
                    e_str += 'divide the domain length along axis '
                    e_str += str(axis) + ' (' + str(length) + ').'
                    raise ValueError(e_str)
                sides[axis] = np.linspace(mins[axis], maxs[axis], n_pix + 1)

        n_dim = len(mins)
        if n_dim not in _RASTER_CORNERS:
            e_str = 'Cannot create a raster mesh in ' + str(n_dim) + 'D.'
            raise NotImplementedError(e_str)

        # 'ij' indexing: node_nums[i, j(, k)] is the node at
        # (sides[0][i], sides[1][j](, sides[2][k]))
        mgrid = np.meshgrid(*sides, indexing='ij')
        nodes = np.array([g.flatten() for g in mgrid]).T
        node_nums = np.arange(mgrid[0].size).reshape(mgrid[0].shape)

        # Elements are counter-clockwise (2D) / right-handed with nodes 1-4
        # on the bottom face and 5-8 on the top face (3D).
        pix_shape = tuple([n - 1 for n in node_nums.shape])
        kp_cols = []
        for offset in _RASTER_CORNERS[n_dim]:
            slices = [slice(o, o + n) for o, n in zip(offset, pix_shape)]
            kp_cols.append(node_nums[tuple(slices)].flatten())
        elems = np.array(kp_cols).T
        n_elems = elems.shape[0]
        elem_grid = np.arange(n_elems).reshape(pix_shape)

        # 2. Compute element centers
        cens = nodes[elems[:, 0]] + 0.5 * mesh_size

        # 3. For each region: assign the pixels/voxels with centers inside
        cell_geom = _CellGeometry(polymesh, p_pts)
        i_remain = np.arange(n_elems)
        elem_regs = np.full(n_elems, -1)
        seed_nums = np.full(n_elems, -1)
        for r_num in range(len(polymesh.regions)):
            # A. Isolate element centers with the bounding box of the cell
            r_mins, r_maxs = cell_geom.limits(r_num)
            r_cens = cens[i_remain]
            in_box = np.all((r_cens >= r_mins) & (r_cens <= r_maxs), axis=1)
            r_i_remain = i_remain[in_box]

            # B. Remove centers on the wrong side of the facets
            # note: regions are convex, so mean pt is on correct side
            _, normals, centers = cell_geom.facets(r_num)
            rel_pos = cens[r_i_remain][:, np.newaxis, :] - centers
            dp = np.einsum('efd,fd->ef', rel_pos, normals)
            r_i_remain = r_i_remain[np.all(dp >= 0, axis=1)]

            # C. Assign remaining centers to region
            elem_regs[r_i_remain] = r_num
            seed_nums[r_i_remain] = polymesh.seed_numbers[r_num]
            i_remain = np.setdiff1d(i_remain, r_i_remain)

        # 4. Combine regions of the same seed number
        conv_dict = _amorphous_seed_numbers(polymesh, phases)
        elem_atts = np.array([conv_dict.get(s, s) for s in seed_nums])

        # 5. Elements to keep: inside a cell of the polymesh and not void
        void_seeds = []
        for seed_num, phase_num in zip(polymesh.seed_numbers,
                                       polymesh.phase_numbers):
            mat_type = phases[phase_num].get('material_type', 'solid')
            if mat_type in _misc.kw_void:
                void_seeds.append(seed_num)
        keep = (elem_regs >= 0) & ~np.isin(seed_nums, void_seeds)

        # 6. Facets: faces between pixels of different cells, and faces on
        # the boundary of the domain, with the polymesh facet number as
        # the attribute
        facets, facet_atts = _raster_facets(polymesh, phases, cell_geom,
                                            elems, elem_grid, elem_regs,
                                            keep, cens, mesh_size)

        # 7. Remove voids and excess cells, re-number nodes
        elems = elems[keep]
        elem_atts = elem_atts[keep]

        nodes_mask = np.full(nodes.shape[0], False)
        nodes_mask[elems] = True
        node_n_conv = np.full(nodes.shape[0], -1)
        node_n_conv[nodes_mask] = np.arange(np.sum(nodes_mask))

        nodes = nodes[nodes_mask]
        elems = node_n_conv[elems]
        facets = node_n_conv[facets]

        mesh = cls(nodes, elems, elem_atts, facets, facet_atts)
        if periodic:
            dom_lims = [(float(lb), float(ub)) for lb, ub in zip(mins, maxs)]
            mesh._set_periodic_pairs(per_axes, dom_lims)
        return mesh

    # ----------------------------------------------------------------------- #
    # String and Representation Functions                                     #
    # ----------------------------------------------------------------------- #
    # __str__ inherited from TriMesh

    def __repr__(self):
        repr_str = 'RasterMesh('
        repr_str += ', '.join([repr(v) for v in (self.points, self.elements,
                               self.element_attributes, self.facets,
                               self.facet_attributes)])
        repr_str += ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Write Function                                                          #
    # ----------------------------------------------------------------------- #
    def write(self, filename, format='txt', seeds=None, polymesh=None):
        """Write mesh to file.

        This function writes the contents of the mesh to a file.
        The format options are 'abaqus', 'txt', and 'vtk'.
        See the :ref:`s_tri_file_io` section of the :ref:`c_file_formats`
        guide for more details on these formats.

        VTK files use the `RECTILINEAR_GRID` data type.

        Args:
            filename (str): The name of the file to write.
            format (str): {'abaqus' | 'txt' | 'vtk'}
                *(optional)* The format of the output file.
                Default is 'txt'.
            seeds (SeedList): *(optional)* List of seeds. If given, VTK files
                will also include the phase number of of each element in the
                mesh. This assumes the ``element_attributes``
                field contains the seed number of each element.
            polymesh (PolyMesh): *(optional)* Polygonal mesh used for
                generating the raster mesh. If given, will add surface
                unions to Abaqus files - for easier specification of
                boundary conditions.

        """  # NOQA: E501
        fmt = format.lower()
        if fmt == 'abaqus':
            # write top matter
            abaqus = '*Heading\n'
            abaqus += '** Job name: microstructure '
            abaqus += 'Model name: microstructure_model\n'
            abaqus += '** Generated by: MicroStructPy\n'

            # write parts
            abaqus += '**\n** PARTS\n**\n'
            abaqus += '*Part, name=Part-1\n'

            abaqus += '*Node\n'
            abaqus += ''.join([str(i + 1) + ''.join([', ' + str(x) for x in
                               pt]) + '\n' for i, pt in
                               enumerate(self.points)])

            n_dim = len(self.points[0])
            elem_type = {2: 'CPS4', 3: 'C3D8'}[n_dim]

            abaqus += '*Element, type=' + elem_type + '\n'
            abaqus += ''.join([str(i + 1) + ''.join([', ' + str(int(kp) + 1)
                                                     for kp in elem]) + '\n'
                               for i, elem in enumerate(self.elements)])

            # Node sets - periodic faces (in paired order)
            abaqus += _abaqus_periodic_nsets(self)

            # Element sets - seed number
            elset_n_per = 16
            if self.element_attributes is None:
                elem_atts = np.array([])
            else:
                elem_atts = np.array(self.element_attributes)
            for att in np.unique(elem_atts):
                elset_name = 'Set-E-Seed-' + str(att)
                elset_str = '*Elset, elset=' + elset_name + '\n'
                elem_groups = [[]]
                for elem_ind, elem_att in enumerate(elem_atts):
                    if ~np.isclose(elem_att, att):
                        continue
                    if len(elem_groups[-1]) >= elset_n_per:
                        elem_groups.append([])
                    elem_groups[-1].append(elem_ind + 1)
                for group in elem_groups:
                    elset_str += ','.join([str(i) for i in group])
                    elset_str += '\n'

                abaqus += elset_str

            # Element Sets - phase number
            if seeds is not None:
                phase_nums = np.array([seed.phase for seed in seeds])
                for phase_num in np.unique(phase_nums):
                    mask = phase_nums == phase_num
                    seed_nums = np.nonzero(mask)[0]

                    elset_name = 'Set-E-Material-' + str(phase_num)
                    elset_str = '*Elset, elset=' + elset_name + '\n'
                    groups = [[]]
                    for seed_num in seed_nums:
                        if seed_num not in elem_atts:
                            continue
                        if len(groups[-1]) >= elset_n_per:
                            groups.append([])
                        seed_elset_name = 'Set-E-Seed-' + str(seed_num)
                        groups[-1].append(seed_elset_name)
                    for group in groups:
                        elset_str += ','.join(group)
                        elset_str += '\n'
                    abaqus += elset_str

            # Surfaces - Exterior and Interior
            # Each facet is the face of a pixel/voxel. The Abaqus face id
            # (S1, S2, ...) is found from the local node numbers of the face.
            defined_surfs = set()
            has_facets = self.facets is not None and len(self.facets) > 0
            if has_facets and self.facet_attributes is not None:
                elem_faces = {}
                face_ids = _ABAQUS_FACE_IDS[n_dim]
                for i, elem in enumerate(self.elements):
                    for face, local_kps in _RASTER_FACES[n_dim].items():
                        key = tuple(sorted([int(elem[k]) for k in local_kps]))
                        elem_faces.setdefault(key, (i + 1, face_ids[face]))

                facets = np.array(self.facets)
                facet_atts = np.array(self.facet_attributes)
                for att in np.unique(facet_atts):
                    facet_name = 'Surface-' + str(att)
                    surf_str = '*Surface, name=' + facet_name
                    surf_str += ', type=element\n'

                    for facet in facets[facet_atts == att]:
                        key = tuple(sorted([int(kp) for kp in facet]))
                        if key not in elem_faces:
                            e_str = 'Facet ' + str(list(key))
                            e_str += ' is not a face of any element.'
                            raise ValueError(e_str)
                        elem_id, face_id = elem_faces[key]
                        surf_str += str(elem_id) + ', S' + str(face_id) + '\n'

                    abaqus += surf_str
                    defined_surfs.add(int(att))

            # Surfaces - Exterior (unions of the surfaces on each domain face)
            if polymesh is not None:
                abaqus += _abaqus_exterior_unions(polymesh, defined_surfs)

            # End Part
            abaqus += '*End Part\n\n'

            # Assembly
            abaqus += '**\n'
            abaqus += '** ASSEMBLY\n'
            abaqus += '**\n'

            abaqus += '*Assembly, name=assembly\n'
            abaqus += '**\n'

            # Instances
            abaqus += '*Instance, name=I-Part-1, part=Part-1\n'
            abaqus += '*End Instance\n'

            # End Assembly
            abaqus += '**\n'
            abaqus += '*End Assembly\n'

            with open(filename, 'w') as file:
                file.write(abaqus)
        elif fmt in ('str', 'txt'):
            with open(filename, 'w') as file:
                file.write(str(self) + '\n')
        elif fmt == 'vtk':
            n_kp = len(self.elements[0])
            mesh_type = {4: 'Pixel', 8: 'Voxel'}[n_kp]

            # Element attributes on the full grid, -1 where there is no
            # element (voids, outside the domain)
            has_atts = self.element_attributes is not None
            arr = self.as_array(element_attributes=has_atts)

            # Dimensions
            pts = np.array(self.points)
            mins = pts.min(axis=0)
            sz = self.mesh_size
            coords = [mins[i] + sz * np.arange(n + 1) for i, n in
                      enumerate(arr.shape)]
            if len(coords) < 3:
                coords.append(np.array([0.0]))  # force z=0 for 2D meshes
            dims = [len(c) for c in coords]

            # write heading
            vtk = '# vtk DataFile Version 2.0\n'
            vtk += '{} mesh\n'.format(mesh_type)
            vtk += 'ASCII\n'
            vtk += 'DATASET RECTILINEAR_GRID\n'
            vtk += 'DIMENSIONS {} {} {}\n'.format(*dims)

            # write points
            for ind, ax in enumerate(['X', 'Y', 'Z']):
                vtk += '{}_COORDINATES {} float\n'.format(ax, dims[ind])
                vtk += _vtk_lines(['{:f}'.format(x) for x in coords[ind]])

            # write element attributes, in the order VTK expects the cells
            # (x index varying fastest, then y, then z)
            vals = arr.flatten(order='F')
            if np.issubdtype(arr.dtype, np.integer):
                att_type = 'int'
                att_strs = [str(int(v)) for v in vals]
            else:
                att_type = 'float'
                att_strs = ['{:f}'.format(v) for v in vals]

            vtk += 'CELL_DATA {}\n'.format(len(vals))
            vtk += 'SCALARS element_attributes {} 1\n'.format(att_type)
            vtk += 'LOOKUP_TABLE default\n'
            vtk += _vtk_lines(att_strs)

            # write phase numbers
            if seeds is not None and has_atts:
                phase_strs = []
                for v in vals:
                    if v < 0:
                        phase_strs.append('-1')
                    else:
                        phase_strs.append(str(int(seeds[int(v)].phase)))
                vtk += 'SCALARS phase_numbers int 1\n'
                vtk += 'LOOKUP_TABLE default\n'
                vtk += _vtk_lines(phase_strs)

            with open(filename, 'w') as file:
                file.write(vtk)

        else:
            e_str = 'Cannot write file type ' + str(format) + ' yet.'
            raise NotImplementedError(e_str)

    # ----------------------------------------------------------------------- #
    # As Array Functions                                                      #
    # ----------------------------------------------------------------------- #
    @property
    def mesh_size(self):
        """Side length of elements."""
        e0 = self.elements[0]
        s0 = np.array(self.points[e0[1]]) - np.array(self.points[e0[0]])
        return np.linalg.norm(s0)

    def as_array(self, element_attributes=True):
        """numpy.ndarray containing element attributes.

        Array contains -1 where there are no elements (e.g. circular domains).

        Args:
            element_attributes (bool): *(optional)* Flag to return element
                attributes in the array. Set to True return attributes and
                set to False to return element indices. Defaults to True.

        Returns:
            numpy.ndarray: Array of values of element atttributes, or indices.

        """
        # 1. Convert 1st node of each element into array indices
        pts = np.array(self.points)
        mins = pts.min(axis=0)
        sz = self.mesh_size

        corner_pts = pts[np.array(self.elements)[:, 0]]
        rel_pos = corner_pts - mins
        elem_tups = np.round(rel_pos / sz).astype(int)

        # 2. Create array full of -1 values
        inds_maxs = elem_tups.max(axis=0)
        if element_attributes:
            vals = np.asarray(self.element_attributes)
        else:
            vals = np.arange(elem_tups.shape[0])
        if vals.dtype.kind in 'biu':
            arr = np.full(inds_maxs + 1, -1)
        else:
            arr = np.full(inds_maxs + 1, -1, dtype=vals.dtype)

        # 3. Populate array with element attributes (or indices)
        arr[tuple(elem_tups.T)] = vals

        return arr

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, index_by='element', material=[], loc=0, **kwargs):
        """Plot the mesh.

        This method plots the mesh using matplotlib.
        In 2D, this creates a :class:`matplotlib.collections.PolyCollection`
        and adds it to the current axes.
        In 3D, it creates a
        :meth:`mpl_toolkits.mplot3d.axes3d.Axes3D.voxels` and
        adds it to the current axes.
        The keyword arguments are passed though to matplotlib.

        Args:
            index_by (str): *(optional)* {'element' | 'attribute'}
                Flag for indexing into the other arrays passed into the
                function. For example,
                ``plot(index_by='attribute', color=['blue', 'red'])`` will plot
                the elements with ``element_attribute`` equal to 0 in blue, and
                elements with ``element_attribute`` equal to 1 in red.
                Defaults to 'element'.
            material (list): *(optional)* Names of material phases. One entry
                per material phase (the ``index_by`` argument is ignored).
                If this argument is set, a legend is added to the plot with
                one entry per material. Note that the ``element_attributes``
                must be the material numbers for the legend to be
                formatted properly.
            loc (int or str): *(optional)* The location of the legend,
                if 'material' is specified. This argument is passed directly
                through to :func:`matplotlib.pyplot.legend`. Defaults to 0,
                which is 'best' in matplotlib.
            **kwargs: Keyword arguments that are passed through to matplotlib.

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
            _plot_2d(ax, self, index_by, **kwargs)
        else:
            if n_obj > 0:
                zlim = ax.get_zlim()
            else:
                zlim = [float('inf'), -float('inf')]

            inds = self.as_array(element_attributes=index_by == 'attribute')
            plt_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, (list, np.ndarray)):
                    plt_value = np.empty(inds.shape, dtype=object)
                    for i, val_i in enumerate(value):
                        plt_value[inds == i] = val_i
                    if 'color' in key:
                        unset_mask = np.equal(plt_value, None)
                        plt_value[unset_mask] = 'k'
                        inds[unset_mask] = -1

                else:
                    plt_value = value
                plt_kwargs[key] = plt_value

            # Corners of the voxels
            pts = np.array(self.points)
            mins = pts.min(axis=0)
            sz = self.mesh_size
            axes = [mins[i] + sz * np.arange(n + 1) for i, n in
                    enumerate(inds.shape)]
            grids = np.meshgrid(*axes, indexing='ij')
            ax.voxels(*grids, inds >= 0, **plt_kwargs)

        # Add legend
        if material and index_by == 'attribute':
            p_kwargs = [{'label': m} for m in material]
            for key, value in kwargs.items():
                if not isinstance(value, (list, np.ndarray)):
                    for kws in p_kwargs:
                        kws[key] = value

                for i, m in enumerate(material):
                    if isinstance(value, (list, np.ndarray)):
                        p_kwargs[i][key] = value[i]
                    else:
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


def facet_check(neighs, polymesh, phases):
    if any([n < 0 for n in neighs]):
        add_facet = True
    else:
        seed_nums = [polymesh.seed_numbers[n] for n in neighs]
        phase_nums = [polymesh.phase_numbers[n] for n in neighs]
        m1, m2 = [phases[n].get('material_type', 'solid') for n in
                  phase_nums]

        same_seed = seed_nums[0] == seed_nums[1]
        same_phase = phase_nums[0] == phase_nums[1]

        if (m1 in _misc.kw_solid) and same_seed:
            add_facet = False
        elif (m1 in _misc.kw_amorph) and same_phase:
            add_facet = False
        elif (m1 in _misc.kw_void) and (m2 in _misc.kw_void):
            add_facet = False
        else:
            add_facet = True

    return add_facet


def _pt_ab(i, pt):
    return str(i + 1) + ''.join([', ' + str(x) for x in pt]) + '\n'


def _call_meshpy(polymesh, phases=None, min_angle=0, max_volume=float('inf'),
                 max_edge_length=float('inf'), periodic=False):

    # condition the phases input
    if phases is None:
        phases = _default_phases(polymesh)

    # create point and facet lists
    kps = {}
    pts = []
    facets = []
    facet_neighs = []
    facet_nums = []
    for i in range(len(polymesh.facets)):
        facet = polymesh.facets[i]
        neighs = polymesh.facet_neighbors[i]
        if facet_check(neighs, polymesh, phases):
            new_facet = []
            for kp_old in facet:
                if kp_old not in kps:
                    kp_new = len(pts)
                    pts.append(polymesh.points[kp_old])
                    kps[kp_old] = kp_new
                else:
                    kp_new = kps[kp_old]
                new_facet.append(kp_new)
            facets.append(new_facet)
            facet_neighs.append(neighs)
            facet_nums.append(i + 1)

    # Subdivide facets
    n_dim = len(pts[0])
    if n_dim == 2:
        n_subs = np.ones(len(facets), dtype='int')
        for i, facet in enumerate(facets):
            pt1 = np.array(pts[facet[0]])
            pt2 = np.array(pts[facet[1]])
            rel_pos = pt2 - pt1
            n_float = np.linalg.norm(rel_pos) / max_edge_length
            n_int = max(1, np.ceil(n_float))
            n_subs[i] = n_int

        # Facets on opposite periodic faces are subdivided identically, so
        # that their nodes are images of each other
        if periodic:
            f_index = {f_num - 1: i for i, f_num in enumerate(facet_nums)}
            for axis_pairs in (polymesh.periodic_facets or {}).values():
                for f_lo, f_hi in axis_pairs:
                    if f_lo in f_index and f_hi in f_index:
                        n_max = max(n_subs[f_index[f_lo]],
                                    n_subs[f_index[f_hi]])
                        n_subs[f_index[f_lo]] = n_max
                        n_subs[f_index[f_hi]] = n_max

        sub_out = meshpy.triangle.subdivide_facets(n_subs, pts, facets,
                                                   facet_nums)
        pts, facets, facet_nums = sub_out

    # create groups/regions
    pts_arr = np.array(polymesh.points)
    regions = []
    holes = []

    # Merged cells are labelled with the smallest seed number among them
    # (the same convention as the other writers and meshers), which is not
    # the seed number of the first cell of the group in a periodic mesh.
    labels = _merged_seed_numbers(polymesh, phases)

    ungrouped = np.full(len(polymesh.regions), True, dtype='?')
    while np.any(ungrouped):
        cell_ind = np.argmax(ungrouped)

        # compute cell center
        facet_list = polymesh.regions[cell_ind]
        cell_kps = {kp for n in facet_list for kp in polymesh.facets[n]}
        cell_cen = pts_arr[list(cell_kps)].mean(axis=0)

        # seed number and phase type
        seed_num = int(labels[cell_ind])
        phase_num = polymesh.phase_numbers[cell_ind]
        phase = phases[phase_num]
        phase_type = phase.get('material_type', 'crystalline')
        phase_vol = phase.get('max_volume', max_volume)

        # get all cell numbers in group
        cell_nums = set([cell_ind])
        old_len = len(cell_nums)
        searching_front = True
        while searching_front:
            front = set()
            for n in cell_nums:
                neighs = set()
                for facet_num in polymesh.regions[n]:
                    f_neighs = polymesh.facet_neighbors[facet_num]
                    neigh_ind = [i for i in f_neighs if i != n][0]
                    if neigh_ind < 0:
                        continue
                    if not facet_check(f_neighs, polymesh, phases):
                        neighs.add(neigh_ind)
                assert ungrouped[list(neighs)].all()
                front.update(neighs)
            cell_nums |= front
            new_len = len(cell_nums)
            searching_front = new_len != old_len
            old_len = new_len

        ungrouped[list(cell_nums)] = False

        # update appropriate list
        if phase_type in _misc.kw_void:
            holes.append(cell_cen)
        else:
            regions.append(cell_cen.tolist() + [seed_num, phase_vol])

    # run MeshPy
    # The maximum element volume is set per region above, using the global
    # value as the default for the phases that do not set their own. Only
    # these regional constraints are passed to Triangle/TetGen: a fixed
    # (global) constraint would cap the per-phase values and, in 2D, an
    # infinite one is formatted as 'ainf', which Triangle reads as the
    # switches -a -i -n -f.
    # A periodic mesh is built like a non-periodic one, then made periodic
    # (see _build_periodic_2d and _build_periodic_3d).
    if n_dim == 2:
        if periodic:
            tri_pts, tri_elems, tri_e_atts = _build_periodic_2d(
                polymesh, phases, labels, kps, pts, facets, facet_nums,
                holes, regions, min_angle, max_volume)
        else:
            tri_mesh = _build_2d(pts, facets, facet_nums, holes, regions,
                                 min_angle, True)
    else:
        opts = meshpy.tet.Options('pq')
        opts.mindihedral = min_angle
        opts.varvolume = 1
        opts.fixedvolume = 0
        opts.regionattrib = 1
        opts.facesout = 1
        if periodic:
            tri_mesh = _build_periodic_3d(polymesh, phases, kps, pts,
                                          facet_nums, holes, regions, opts,
                                          max_volume, max_edge_length)
        else:
            if np.isfinite(max_edge_length):
                # the facets are triangulated to the maximum edge length
                # (TetGen has no such control) and TetGen refines the
                # interior of the cells to the maximum volume
                pts, facets, facet_nums = _triangulate_facets_3d(
                    polymesh, phases, kps, pts, facet_nums, max_volume,
                    max_edge_length, {}, {})
            info = _tet_info(pts, facets, facet_nums, holes, regions)
            tri_mesh = meshpy.tet.build(info, options=opts)

    # return mesh
    if periodic:
        # The element attributes and the facets are taken from the
        # geometry of the polymesh (TetGen can leave sub-faces unmarked
        # when it may not modify the facets, and its region attributes
        # then leak between cells)
        if n_dim == 3:
            tri_pts = np.array(tri_mesh.points)
            tri_elems = np.array(tri_mesh.elements)
        tri_e_atts, tri_f, tri_fa = _attributes_from_polymesh(
            tri_pts, tri_elems, polymesh, labels)
    else:
        tri_pts = np.array(tri_mesh.points)
        tri_elems = np.array(tri_mesh.elements)
        tri_e_atts = np.array(tri_mesh.element_attributes, dtype='int')
        tri_faces = np.array(tri_mesh.faces)
        tri_f_atts = np.array(tri_mesh.face_markers)
        f_mask = tri_f_atts > 0
        tri_f = tri_faces[f_mask]
        tri_fa = tri_f_atts[f_mask] - 1

    tri_args = (tri_pts, tri_elems, tri_e_atts, tri_f, tri_fa)
    return tri_args


def _sorted_facets(facets, facet_atts):
    """Facets with their nodes in ascending order, in lexicographic order.

    Triangle and TetGen list the edges/faces of a mesh in an order, and
    with an orientation, that vary from one run to the next; the sorted
    facets make a mesh, and its files, reproducible. MicroStructPy does not
    use the orientation of the facets.

    Args:
        facets (list or numpy.ndarray): The facets (node numbers).
        facet_atts (list or numpy.ndarray): The attribute of each facet.

    Returns:
        tuple: The sorted facets and their attributes, as arrays.

    """
    facets = np.array(facets, dtype='int')
    facet_atts = np.array(facet_atts)
    if facets.size == 0:
        return facets, facet_atts
    facets = np.sort(facets, axis=1)
    order = np.lexsort(facets.T[::-1])
    return facets[order], facet_atts[order]


def _call_gmsh(pmesh, phases, res, edge_res):
    if phases is None:
        phases = _default_phases(pmesh)
    if res == float('inf'):
        res = None
    # If edge length not specified, default to mesh size input
    if edge_res == float('inf'):
        edge_res = res

    amorph_seeds = _amorphous_seed_numbers(pmesh, phases)

    # ---------------------------------------------------------------------- #
    # CREATE CONNECTIVITY DATA
    # ---------------------------------------------------------------------- #
    # Extract edges from facets list
    facets_info = {}
    edges_info = {}
    edge_keys = []
    edge_lines = []
    n_edges = 0
    for i, f in enumerate(pmesh.facets):
        # Determine if facet should be skipped (interior to seed)
        keep = True
        ns = pmesh.facet_neighbors[i]
        if min(ns) >= 0:
            keep = pmesh.seed_numbers[ns[0]] != pmesh.seed_numbers[ns[1]]
        if not keep:
            continue

        facets_info[i] = {'facet': f, 'seeds': []}
        n = len(f)
        facet_kp_pairs = [(f[k], f[(k + 1) % n]) for k in range(n)]
        edge_numbers = []
        edge_signs = []
        for pair in facet_kp_pairs:
            key = tuple(sorted(pair))
            if pair == key:
                edge_sign = 1
            else:
                edge_sign = -1

            if key not in edge_keys:
                edges_info[key] = {'ind': n_edges, 'facets': [], 'seeds': []}
                edge_keys.append(key)
                n_edges += 1
            edges_info[key]['facets'].append(i)
            edge_num = edges_info[key]['ind']

            # Add seeds
            neighs = pmesh.facet_neighbors[i]
            edges_info[key]['neighbors'] = neighs
            for neigh_cell in neighs:
                if neigh_cell < 0:
                    seed_num = neigh_cell
                else:
                    seed_num = pmesh.seed_numbers[neigh_cell]
                edges_info[key]['seeds'].append(seed_num)

            edge_numbers.append(edge_num)
            edge_signs.append(edge_sign)
        facets_info[i]['neighbors'] = pmesh.facet_neighbors[i]
        facets_info[i]['edge_numbers'] = edge_numbers
        facets_info[i]['edge_signs'] = edge_signs
    for cell_num, seed_num in enumerate(pmesh.seed_numbers):
        facet_nums = [f for f in pmesh.regions[cell_num] if f in facets_info]
        for facet_num in facet_nums:
            facets_info[facet_num]['seeds'].append(seed_num)

    # ---------------------------------------------------------------------- #
    # CREATE GEOMETRY
    # ---------------------------------------------------------------------- #
    with pg.geo.Geometry() as geom:
        # Add points
        pt_arr = np.array(pmesh.points)
        pts = [geom.add_point(_pt3d(pt), edge_res) for pt in pmesh.points]
        n_dim = len(pmesh.points[0])

        # Add edges to geometry
        phys_facets = []
        phys_seeds = []
        for edge in edge_keys:
            line = geom.add_line(*[pts[kp] for kp in edge])
            edge_lines.append(line)

            if n_dim == 2:
                lbl = 'facet-{}'.format(edges_info[edge]['facets'][0])
                geom.add_physical(edge_lines[-1], lbl)
                if facet_check(edges_info[edge]['neighbors'], pmesh, phases):
                    phys_facets.append(lbl)

        if n_dim == 2:
            # Add surfaces to geometry
            loops = []
            surfs = []
            seed_facets = {}
            seed_phases = {}
            for i, r in enumerate(pmesh.regions):
                s = pmesh.seed_numbers[i]
                seed_facets.setdefault(s, set()).symmetric_difference_update(r)
                seed_phases[s] = pmesh.phase_numbers[i]
            for i in seed_facets:
                region = list(seed_facets[i])
                sorted_pairs = _sort_facets([pmesh.facets[f] for f in region])
                loop = []
                for facet in sorted_pairs:
                    key = tuple(sorted(facet))
                    if facet[0] == key[0]:
                        sgn = 1
                    else:
                        sgn = -1

                    n = edges_info[key]['ind']
                    line = edge_lines[n]
                    if sgn > 0:
                        loop.append(line)
                    else:
                        loop.append(-line)

                loops.append(geom.add_curve_loop(loop))
                surfs.append(geom.add_plane_surface(loops[-1]))
                lbl = 'seed-' + str(i)
                geom.add_physical(surfs[-1], lbl)
                p_num = seed_phases[i]
                mat_type = phases[p_num].get('material_type', 'solid')
                if mat_type not in _misc.kw_void:
                    phys_seeds.append(lbl)
                    # Add mesh size control points to 'centers' of regions
                    if res is not None:
                        kps = list({kp for p in sorted_pairs for kp in p})
                        cen = pt_arr[kps].mean(axis=0)  # estimate of center
                        pt = geom.add_point(_pt3d(cen), res)
                        geom.in_surface(pt, surfs[-1])

        elif n_dim == 3:
            # Add surfaces to geometry
            loops = []
            surfs = []
            seed_surfs = {}
            surf_kps = {}
            seed_phases = dict(zip(pmesh.seed_numbers, pmesh.phase_numbers))
            for i in facets_info:
                info = facets_info[i]
                facet_seeds = info['seeds']
                to_add = len(facet_seeds) < 2
                to_add |= facet_seeds[0] != facet_seeds[1]
                if not to_add:
                    surfs.append('')
                    continue

                loop = []
                for n, sgn in zip(info['edge_numbers'], info['edge_signs']):
                    line = edge_lines[n]
                    if sgn > 0:
                        loop.append(line)
                    else:
                        loop.append(-line)
                loops.append(geom.add_curve_loop(loop))
                surfs.append(geom.add_plane_surface(loops[-1]))
                surf_kps[surfs[-1]] = set(info['facet'])
                f_lbl = 'facet-' + str(i)
                geom.add_physical(surfs[-1], 'facet-' + str(i))
                if facet_check(info['neighbors'], pmesh, phases):
                    phys_facets.append(f_lbl)
                for seed_num in facet_seeds:
                    if seed_num not in seed_surfs:
                        seed_surfs[seed_num] = []
                    seed_surfs[seed_num].append(surfs[-1])

            # Add volumes to geometry
            surf_loops = []
            volumes = []
            for seed_num in seed_surfs:
                surf_loop = seed_surfs[seed_num]
                surf_loops.append(geom.add_surface_loop(surf_loop))
                volumes.append(geom.add_volume(surf_loops[-1]))
                lbl = 'seed-' + str(seed_num)
                geom.add_physical(volumes[-1], lbl)

                p_num = seed_phases[seed_num]
                mat_type = phases[p_num].get('material_type', 'solid')
                if mat_type not in _misc.kw_void:
                    phys_seeds.append(lbl)
                    # Add mesh size control points to 'centers' of regions
                    if res is not None:
                        kps = set().union(*[surf_kps[s] for s in surf_loop])
                        cen = pt_arr[list(kps)].mean(axis=0)  # estimate center
                        pt = geom.add_point(_pt3d(cen), res)
                        geom.in_volume(pt, volumes[-1])
        else:
            e_str = 'Points cannot have dimension ' + str(n_dim) + '.'
            raise ValueError(e_str)

        mesh = geom.generate_mesh()

    # ---------------------------------------------------------------------- #
    # CREATE MICROSTRUCTPY.MESHING.TRIMESH
    # ---------------------------------------------------------------------- #
    f_ind = {2: 0, 3: 1}[n_dim]
    e_ind = {2: 1, 3: 2}[n_dim]

    pts = np.array(mesh.points)[:, :n_dim]
    facets = mesh.cells[f_ind].data

    # Sort Element Keypoints for Positive Volume
    tets = np.array([e[_sort_element([mesh.points[k] for k in e])]
                     for e in mesh.cells[e_ind].data])

    tet_atts = np.array([-1 for tet in tets])
    facet_atts = np.array([-1 for f in facets])

    tet_set = np.array([False for tet in tets])
    facet_set = np.array([False for f in facets])

    n_facets = len(mesh.cells[f_ind].data)
    for key, elem_sets in mesh.cell_sets.items():
        set_kind, set_num_str = key.split('-')
        att = int(set_num_str)
        if set_kind == 'seed' and key in phys_seeds:
            elem_set = elem_sets[e_ind] - n_facets
            tet_atts[elem_set] = amorph_seeds.get(att, att)
            tet_set[elem_set] = True
        elif set_kind == 'facet' and key in phys_facets:
            elem_set = elem_sets[f_ind]
            facet_atts[elem_set] = att
            facet_set[elem_set] = True

    tets = tets[tet_set]
    tet_atts = tet_atts[tet_set]

    facets = facets[facet_set]
    facet_atts = facet_atts[facet_set]

    # Remove the points that are not in any element (e.g. inside voids)
    # and re-number the remaining ones
    used = np.unique(tets)
    kp_conv = np.full(len(pts), -1, dtype='int')
    kp_conv[used] = np.arange(len(used))
    pts = pts[used]
    tets = kp_conv[tets]
    if len(facets) > 0:
        f_keep = np.all(kp_conv[facets] >= 0, axis=1)
        facets = kp_conv[facets[f_keep]]
        facet_atts = facet_atts[f_keep]

    tri_args = (pts, tets, tet_atts, facets, facet_atts)
    return tri_args


def _sort_element(elem_pts):
    n_pts = len(elem_pts)
    n_dim = n_pts - 1
    if n_dim == 2:
        v1 = elem_pts[1] - elem_pts[0]
        v2 = elem_pts[2] - elem_pts[0]

        cp = np.cross(v1, v2)[-1]
        if cp < 0:
            return np.array([0, 2, 1])

        return np.arange(3)
    elif n_dim == 3:
        v1 = elem_pts[1] - elem_pts[0]
        v2 = elem_pts[2] - elem_pts[0]
        v3 = elem_pts[3] - elem_pts[0]

        cp = np.cross(v1, v2)
        dp = cp.dot(v3)
        if dp < 0:
            return np.array([0, 1, 3, 2])

        return np.arange(4)
    else:
        raise ValueError('Cannot sort for n pts: ' + str(n_pts))


def _sort_facets(pairs):
    """Chain the edges of a closed loop so that each one starts where the
    previous one ends. Raises ValueError if the edges do not form a single
    loop (e.g. the boundary of a region that is not simply connected).
    """
    remaining_inds = [i for i in range(1, len(pairs))]
    s_pairs = [pairs[0]]
    while remaining_inds:
        last_kp = s_pairs[-1][-1]
        for ind, i in enumerate(remaining_inds):
            pair = pairs[i]
            if last_kp in pair:
                break
        else:
            e_str = 'The facets do not form a single closed loop: none of '
            e_str += 'the ' + str(len(remaining_inds)) + ' remaining facets '
            e_str += 'contains point ' + str(last_kp) + '. The boundary of '
            e_str += 'a region with holes, or of a region made of '
            e_str += 'disconnected cells, cannot be sorted.'
            raise ValueError(e_str)
        del remaining_inds[ind]
        if last_kp == pair[0]:
            s_pairs.append(pair)
        else:
            s_pairs.append(list(reversed(pair)))
    return s_pairs


def _merged_seed_numbers(pmesh, phases):
    """Label (seed number) of each region after merging amorphous cells.

    Cells of the same amorphous phase that share a facet are merged into a
    single region of the mesh, labelled with the smallest seed number among
    them. In a periodic mesh, the pieces of one seed and the cells that touch
    across a periodic face belong to the same region.

    Returns:
        numpy.ndarray: The label of each region of the polymesh.
    """
    seed_nums = np.array(pmesh.seed_numbers)
    phase_nums = np.array(pmesh.phase_numbers)
    is_amorph = np.array([p.get('material_type', 'solid') in _misc.kw_amorph
                          for p in phases])
    amorph_mask = is_amorph[phase_nums]

    sets = _misc.UnionFind(range(len(seed_nums)))

    pairs = [tuple(neighs) for neighs in pmesh.facet_neighbors]
    per_facets = getattr(pmesh, 'periodic_facets', None) or {}
    for axis_pairs in per_facets.values():
        for f_lo, f_hi in axis_pairs:
            pairs.append((max(pmesh.facet_neighbors[f_lo]),
                          max(pmesh.facet_neighbors[f_hi])))
    for r_a, r_b in pairs:
        if r_a < 0 or r_b < 0:
            continue
        if amorph_mask[r_a] and phase_nums[r_a] == phase_nums[r_b]:
            sets.union(r_a, r_b)

    first_region = {}
    for r, s in enumerate(seed_nums):
        if s in first_region:
            sets.union(first_region[s], r)
        else:
            first_region[s] = r

    roots = np.array([sets.find(r) for r in range(len(seed_nums))])
    labels = seed_nums.copy()
    for root in np.unique(roots):
        members = roots == root
        labels[members] = seed_nums[members].min()
    return labels


def _amorphous_seed_numbers(pmesh, phases):
    """Seed numbers that change when amorphous cells are merged.

    Returns:
        dict: Maps the seed number of each merged cell to the label of its
        merged region (see :func:`_merged_seed_numbers`).
    """
    labels = _merged_seed_numbers(pmesh, phases)
    return {int(s): int(lbl) for s, lbl in zip(pmesh.seed_numbers, labels)
            if s != lbl}


def _default_phases(polymesh):
    """Default phases: one solid phase per phase number of the polymesh."""
    n_phases = int(np.max(polymesh.phase_numbers)) + 1
    return [{'material_type': 'solid', 'max_volume': float('inf')}
            for _ in range(n_phases)]


def _pt3d(pt):
    pt3d = np.zeros(3)
    pt3d[:len(pt)] = pt
    return pt3d


def _facet_in_normal(pts, cen_pt):
    """Inward unit normal and center of a facet of a convex cell.

    Args:
        pts (numpy.ndarray): Vertices of the facet.
        cen_pt (numpy.ndarray): A point inside the cell.

    Returns:
        tuple: The unit normal pointing into the cell and the center of
        the facet.

    """
    pts = np.asarray(pts, dtype='float')
    f_cen = pts.mean(axis=0)
    n_dim = len(cen_pt)
    if n_dim == 2:
        vt = pts[1] - pts[0]
        vn = np.array([-vt[1], vt[0]])
    else:
        # Newell's method, which is robust to collinear vertices
        rel_pts = pts - f_cen
        vn = np.zeros(3)
        for i in range(len(rel_pts)):
            vn += np.cross(rel_pts[i - 1], rel_pts[i])

    if vn.dot(cen_pt - f_cen) < 0:
        vn = -vn  # flip so center is inward
    un = vn / np.linalg.norm(vn)
    return un, f_cen


_FACE_MIN_ANGLE = 20.0  # quality of the triangles on the periodic faces (3D)
_MAX_EDGE_SUBDIVISIONS = 400
_MAX_PERIODIC_PASSES = 4


def _facet_sizes(polymesh, phases, facet_nums, max_volume, max_edge_length):
    """Target edge length of the elements on each facet.

    The facets on the periodic faces are triangulated before meshing: to
    the maximum edge length, and to the edge length of the regular
    tetrahedron with the maximum volume of the phase of the cell on the
    facet.

    Returns:
        dict: Maps the polymesh facet number to the edge length.

    """
    n_dim = len(polymesh.points[0])
    h_facets = {}
    for f_num in facet_nums:
        h_val = max_edge_length
        for reg in polymesh.facet_neighbors[f_num - 1]:
            if reg < 0:
                continue
            phase = phases[polymesh.phase_numbers[reg]]
            vol = phase.get('max_volume', max_volume)
            if np.isfinite(vol):
                if n_dim == 2:
                    h_val = min(h_val, np.sqrt(4 * vol / np.sqrt(3)))
                else:
                    h_val = min(h_val, (6 * np.sqrt(2) * vol) ** (1.0 / 3))
        h_facets[f_num - 1] = h_val
    return h_facets


def _edge_key(kp_a, kp_b):
    return (min(kp_a, kp_b), max(kp_a, kp_b))


def _points_on_segment(new_pts, pt_a, pt_b, with_ids=False):
    """Parameters (0 < t < 1) of the points that lie on a segment, sorted;
    with ``with_ids``, the indices of the points in the same order too."""
    if len(new_pts) == 0:
        return ([], []) if with_ids else []
    rel = np.array(new_pts) - pt_a
    seg = pt_b - pt_a
    len2 = np.dot(seg, seg)
    t_vals = rel.dot(seg) / len2
    dists = np.linalg.norm(rel - np.outer(t_vals, seg), axis=1)
    on_seg = (t_vals > 1e-9) & (t_vals < 1 - 1e-9)
    on_seg &= dists <= 1e-9 * np.sqrt(len2)
    ids = np.nonzero(on_seg)[0]
    order = np.argsort(t_vals[ids])
    ids = ids[order]
    if with_ids:
        return t_vals[ids].tolist(), ids.tolist()
    return t_vals[ids].tolist()


def _merge_params(t_vals, sides=None, n_max=_MAX_EDGE_SUBDIVISIONS,
                  min_gap=1e-6):
    """Subdivision of a segment from the parameters of points on it.

    The points come from the refinement of the facets that share the
    segment and of its periodic images, so each face of a periodic pair
    contributes a set of points. Points closer than ``min_gap`` (relative
    to the segment) are merged, and a point is dropped when a point of
    another side (``sides``, one value per parameter) is kept closer than
    0.4 times the gap to the next point: the subdivision is as fine as the
    finest side, not the union of the sides.

    Returns:
        list: The sorted parameters (0 < t < 1) of the subdivision points,
        at most n_max - 1 of them.

    """
    if sides is None:
        sides = [None] * len(t_vals)
    order = np.argsort(t_vals)
    ts = [t_vals[i] for i in order]
    ss = [sides[i] for i in order]
    kept_t = [0.0]
    kept_s = [None]
    for i, (t_val, side) in enumerate(zip(ts, ss)):
        if not (0 < t_val < 1 - min_gap) or t_val - kept_t[-1] <= min_gap:
            continue
        gap_next = (ts[i + 1] if i + 1 < len(ts) else 1.0) - t_val
        if (side is not None and kept_s[-1] is not None and
                kept_s[-1] != side and t_val - kept_t[-1] < 0.4 * gap_next):
            continue
        kept_t.append(t_val)
        kept_s.append(side)
    merged = kept_t[1:]
    if len(merged) >= n_max:
        merged = [i / n_max for i in range(1, n_max)]
    return merged


def _triangle_polygon(loop_pts, h_val, allow_boundary_steiner, extra_pts=(),
                      quality=True):
    """Triangulate a planar convex polygon in 3D with Triangle.

    With ``quality``, the triangles have a minimum angle of 20 degrees
    and, if ``h_val`` is finite, at most the area of the equilateral
    triangle with that edge length; Steiner points are added on the edges
    of the polygon only if allowed. Without it, the triangulation is the
    constrained Delaunay triangulation of the points. The extra points,
    inside the polygon, are vertices of the triangulation.

    Returns:
        tuple: The points (the polygon points, then the extra points, in
        order, then the new ones) as an array, and the triangles (lists of
        point indices).

    """
    loop_pts = np.asarray(loop_pts, dtype='float')
    n_pts = len(loop_pts)
    extra_pts = np.asarray(extra_pts, dtype='float').reshape(-1, 3)
    n_in = n_pts + len(extra_pts)

    # orthonormal basis of the plane of the polygon
    normal = np.zeros(3)
    for i in range(n_pts):
        normal += np.cross(loop_pts[i - 1], loop_pts[i])
    normal /= np.linalg.norm(normal)
    edges = np.roll(loop_pts, -1, axis=0) - loop_pts
    u_vec = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    u_vec = u_vec - np.dot(u_vec, normal) * normal
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    origin = loop_pts[0]
    in_pts = np.vstack([loop_pts, extra_pts])
    rel = in_pts - origin
    pts_2d = np.column_stack([rel.dot(u_vec), rel.dot(v_vec)])

    info = meshpy.triangle.MeshInfo()
    info.set_points(pts_2d.tolist())
    info.set_facets([(i, (i + 1) % n_pts) for i in range(n_pts)])
    max_area = None
    min_angle = None
    if quality:
        min_angle = _FACE_MIN_ANGLE
        if np.isfinite(h_val):
            max_area = 0.25 * np.sqrt(3) * h_val * h_val
    tri = meshpy.triangle.build(info, max_volume=max_area,
                                min_angle=min_angle, quality_meshing=quality,
                                allow_boundary_steiner=allow_boundary_steiner)

    # Triangle keeps the input points first, in order
    out_2d = np.array(tri.points)
    if len(out_2d) < n_in or not np.allclose(out_2d[:n_in], pts_2d):
        raise RuntimeError('Triangle did not keep the input points of a '
                           'facet.')
    out_pts = origin + np.outer(out_2d[:, 0], u_vec)
    out_pts += np.outer(out_2d[:, 1], v_vec)
    out_pts[:n_pts] = loop_pts  # the extra points are projected on the plane
    for axis in range(3):
        if np.ptp(loop_pts[:, axis]) <= 1e-12:
            out_pts[:, axis] = loop_pts[0, axis]
    tris = [list(elem) for elem in np.array(tri.elements)]
    return out_pts, tris


def _triangulate_facets_3d(polymesh, phases, kps, pts, facet_nums, max_volume,
                           max_edge_length, edge_t, face_pts):
    """Triangulate the facets of a periodic 3D polymesh.

    Each facet is triangulated with Triangle (with a minimum angle of 20
    degrees and at most the area given by the mesh size, see
    :func:`_facet_sizes`). The edges of the facets are subdivided where
    Triangle refines them and at the extra parameters in ``edge_t``, and
    the triangulations contain the extra points in ``face_pts``. An edge
    on a periodic face is subdivided identically to its periodic images,
    and the facets on the upper periodic faces get the images of the
    points and triangles of the facets on the lower faces, so that the
    nodes on opposite faces match.

    Args:
        polymesh (PolyMesh): The periodic polymesh.
        phases (list): The phases.
        kps (dict): Maps polymesh point numbers to the point numbers of the
            mesher input.
        pts (list): Points of the mesher input.
        facet_nums (list): Polymesh facet number + 1 of each facet of the
            mesher input.
        max_volume (float): The default maximum volume of the elements.
        max_edge_length (float): The maximum edge length.
        edge_t (dict): Maps an edge (pair of polymesh point numbers, in
            increasing order) to parameters of extra points on it.
        face_pts (dict): Maps a facet number to extra points in the facet
            (for a facet on an upper periodic face, they are stored with
            the facet on the lower face).

    Returns:
        tuple: The new points, facets (triangles) and facet numbers.

    """
    pts = [list(p) for p in pts]
    p_arr = np.array(polymesh.points)
    n_dim = p_arr.shape[1]
    lengths = p_arr.max(axis=0) - p_arr.min(axis=0)
    scale = lengths.max()
    per_pts = polymesh.periodic_points or {}
    per_facets = polymesh.periodic_facets or {}
    lo_hi = {axis: dict(pairs) for axis, pairs in per_pts.items()}
    hi_lo = {axis: {b: a for a, b in pairs} for axis, pairs in
             per_pts.items()}
    h_facets = _facet_sizes(polymesh, phases, facet_nums, max_volume,
                            max_edge_length)

    upper = {}
    for axis, f_pairs in per_facets.items():
        for f_lo, f_hi in f_pairs:
            upper[f_lo] = (axis, f_hi)
    is_upper = set([f_hi for _, f_hi in upper.values()])

    # 1. Edges of the facets; an edge and its periodic images are
    # subdivided identically
    edge_keys = set()
    for f_num in facet_nums:
        loop = polymesh.facets[f_num - 1]
        for i in range(len(loop)):
            edge_keys.add(_edge_key(loop[i - 1], loop[i]))

    classes = _misc.UnionFind(edge_keys)
    find = classes.find
    for kp_map in lo_hi.values():
        for key in edge_keys:
            if key[0] in kp_map and key[1] in kp_map:
                image = _edge_key(kp_map[key[0]], kp_map[key[1]])
                if image in edge_keys:
                    classes.attach(key, image)

    def to_root(key, t_vals):
        # the parameters along key, in the orientation of its class root
        root = find(key)
        seg = p_arr[key[1]] - p_arr[key[0]]
        seg_root = p_arr[root[1]] - p_arr[root[0]]
        if np.dot(seg, seg_root) >= 0:
            return list(t_vals)
        return [1 - t for t in t_vals]

    # 2. Parameters of the points on the edges: the edges are subdivided
    # to the maximum edge length, and at the extra parameters
    periodic_facets = set(upper) | is_upper
    splits = {}
    for f_num in facet_nums:
        f = f_num - 1
        if not np.isfinite(max_edge_length):
            continue
        loop = polymesh.facets[f]
        for i in range(len(loop)):
            key = _edge_key(loop[i - 1], loop[i])
            edge_len = np.linalg.norm(p_arr[key[1]] - p_arr[key[0]])
            n_sub = int(np.ceil(edge_len / max_edge_length))
            t_vals = [k / n_sub for k in range(1, n_sub)]
            splits.setdefault(find(key), []).extend(
                [(t, None) for t in to_root(key, t_vals)])
    for key, vals in edge_t.items():
        if key in edge_keys:
            ts = to_root(key, [t for t, _ in vals])
            splits.setdefault(find(key), []).extend(
                zip(ts, [s for _, s in vals]))

    # 3. Subdivide the edges; the images of an edge get translated copies of
    # its points, recorded in image_map (lower point -> upper point)
    edge_pts = {}
    image_map = {axis: {kps[a]: kps[b] for a, b in pairs} for axis, pairs in
                 per_pts.items()}
    for key in sorted(edge_keys):
        if key in edge_pts:
            continue
        pt_a, pt_b = p_arr[key[0]], p_arr[key[1]]
        edge_len = np.linalg.norm(pt_b - pt_a)
        min_gap = max(1e-6, 1e-6 * scale / edge_len)
        vals = splits.get(find(key), [])
        t_vals = _merge_params([t for t, _ in vals], [s for _, s in vals],
                               min_gap=min_gap)
        t_vals = sorted(to_root(key, t_vals))
        ids = []
        for t_val in t_vals:
            ids.append(len(pts))
            pts.append((pt_a + t_val * (pt_b - pt_a)).tolist())
        edge_pts[key] = ids

        queue = [key]
        while queue:
            kp_a, kp_b = queue.pop()
            ids = edge_pts[(kp_a, kp_b)]
            for axis in lo_hi:
                for kp_map, sign in ((lo_hi[axis], 1), (hi_lo[axis], -1)):
                    if kp_a not in kp_map or kp_b not in kp_map:
                        continue
                    im_a, im_b = kp_map[kp_a], kp_map[kp_b]
                    im_key = _edge_key(im_a, im_b)
                    if im_key not in edge_keys:
                        continue
                    if im_key not in edge_pts:
                        shift = np.zeros(n_dim)
                        shift[axis] = sign * lengths[axis]
                        im_ids = []
                        for pid in ids:
                            im_ids.append(len(pts))
                            pts.append((np.array(pts[pid]) + shift).tolist())
                        if im_a != im_key[0]:
                            im_ids = im_ids[::-1]
                        edge_pts[im_key] = im_ids
                        queue.append(im_key)
                    im_ids = edge_pts[im_key]
                    if im_a != im_key[0]:
                        im_ids = im_ids[::-1]
                    for pid, im_pid in zip(ids, im_ids):
                        if sign > 0:
                            image_map[axis][pid] = im_pid
                        else:
                            image_map[axis][im_pid] = pid

    # 4. Facet loops with the new points
    loops = {}
    for f_num in facet_nums:
        loop = polymesh.facets[f_num - 1]
        new_loop = []
        for i in range(len(loop)):
            kp_a, kp_b = loop[i], loop[(i + 1) % len(loop)]
            new_loop.append(kps[kp_a])
            key = _edge_key(kp_a, kp_b)
            ids = edge_pts[key]
            new_loop.extend(ids if kp_a == key[0] else ids[::-1])
        loops[f_num - 1] = new_loop

    # 5. Triangulate the facets with their edges fixed; the facets on the
    # upper periodic faces are the images of those on the lower faces. The
    # facets on the periodic faces are refined to the mesh size when one is
    # given (TetGen cannot refine them afterwards), and all the facets to
    # the maximum edge length when it is given; the others are the
    # constrained Delaunay triangulations of their points.
    new_facets = []
    new_nums = []
    for f_num in facet_nums:
        f = f_num - 1
        if f in is_upper:
            continue
        loop_ids = loops[f]
        extra = face_pts.get(f, [])
        quality = f in periodic_facets or np.isfinite(max_edge_length)
        h_val = h_facets[f]
        if quality and np.isfinite(h_val):
            # the area bound is met by equilateral triangles of that edge
            # length; a smaller area keeps the edges of the other triangles
            # at about the maximum edge length, and the elements on the
            # faces, which TetGen may not split, below the maximum volume
            h_val = 0.75 * h_val
        out_pts, tris = _triangle_polygon([pts[k] for k in loop_ids],
                                          h_val, False, extra, quality)
        ids = list(loop_ids)
        new_ids = []
        for pt in out_pts[len(loop_ids):]:
            new_ids.append(len(pts))
            ids.append(len(pts))
            pts.append(pt.tolist())
        tris = [[ids[k] for k in tri] for tri in tris]
        new_facets.extend(tris)
        new_nums.extend([f_num] * len(tris))
        if f not in upper:
            continue

        p_axis, f_hi = upper[f]
        kp_map = image_map[p_axis]
        shift = np.zeros(n_dim)
        shift[p_axis] = lengths[p_axis]
        for pid in new_ids:
            kp_map[pid] = len(pts)
            pts.append((np.array(pts[pid]) + shift).tolist())
        new_facets.extend([[kp_map[k] for k in tri] for tri in tris])
        new_nums.extend([f_hi + 1] * len(tris))
    return pts, new_facets, new_nums


def _merge_face_points(raw, loop_pts, tol_dup):
    """Points inside a facet from the refinement of both periodic faces.

    A point of one side is dropped when a point of another side is kept
    within 0.4 times its distance to the nearest point of its own side (or
    vertex of the facet), so that the facet is as refined as the finest
    side, not the union of the sides. Points closer than ``tol_dup`` are
    merged.

    Args:
        raw (list): Pairs of a point and its side.
        loop_pts (numpy.ndarray): The vertices of the facet.
        tol_dup (float): Distance below which points are the same.

    Returns:
        list: The points kept.

    """
    by_side = {}
    for pt, side in raw:
        by_side.setdefault(side, []).append(pt)
    kept = []
    kept_side = []
    for side in sorted(by_side, key=lambda s: -len(by_side[s])):
        pts_s = np.array(by_side[side])
        tree = cKDTree(np.vstack([pts_s, loop_pts]))
        d_own = tree.query(pts_s, k=2)[0][:, 1]
        for pt, s_own in zip(pts_s, d_own):
            if kept:
                dists = np.linalg.norm(np.array(kept) - pt, axis=1)
                j = int(np.argmin(dists))
                if dists[j] <= tol_dup:
                    continue
                if kept_side[j] != side and dists[j] < 0.4 * s_own:
                    continue
            kept.append(pt.tolist())
            kept_side.append(side)
    return kept


def _collect_facet_points_3d(new_pts, polymesh, edge_t, face_pts):
    """Record the points that TetGen added on the facets of a polymesh.

    A point on an edge of a facet is added to the parameters of that edge
    (``edge_t``), a point inside a facet to the extra points of the facet
    (``face_pts``); points inside the cells are ignored. A point on an
    upper periodic face is moved to the lower face and recorded with the
    facet there, so that the next triangulation of the facets has the
    point, and its images, on both faces. Each point is recorded with its
    side (the periodic faces it was on), and the points of the two faces
    of a pair are merged so that the facets are as refined as the finest
    side (see :func:`_merge_params` and :func:`_merge_face_points`).

    Returns:
        int: The number of points recorded.

    """
    p_arr = np.array(polymesh.points)
    mins = p_arr.min(axis=0)
    lengths = p_arr.max(axis=0) - mins
    maxs = mins + lengths
    scale = lengths.max()
    tol = max(1e-9, 4 * _facet_nonplanarity(polymesh)) * scale
    tol_face = 1e-9 * scale
    tol_dup = 1e-6 * scale
    per_axes = polymesh.periodic_axes
    to_lower = {}
    for axis, f_pairs in (polymesh.periodic_facets or {}).items():
        for f_lo, f_hi in f_pairs:
            to_lower[f_hi] = (axis, f_lo)

    new_pts = np.asarray(new_pts, dtype='float').reshape(-1, 3)
    if len(new_pts) == 0:
        return 0

    # 1. Facets whose plane contains each point, among the facets of the
    # cells that contain it
    cell_geom = _CellGeometry(polymesh, p_arr)
    point_facets = {}
    for r_num, cand, dp in cell_geom.containing(new_pts, tol):
        f_nums = cell_geom.facets(r_num)[0]
        inside = np.all(dp >= -tol, axis=1)
        for i, row in zip(cand[inside], dp[inside]):
            for k in np.nonzero(np.abs(row) <= tol)[0]:
                point_facets.setdefault(i, set()).add(int(f_nums[k]))

    # 2. Record the points on edges and inside facets, with their sides
    n_found = 0
    raw_face = {}
    for i, f_set in point_facets.items():
        pt = np.array(new_pts[i])
        side = tuple([int(bool(per_axes[k]) and
                          abs(pt[k] - maxs[k]) <= tol_face)
                      for k in range(3)])
        f_list = sorted(f_set)
        if len(f_list) > 1:
            loop = polymesh.facets[f_list[0]]
            for k in range(len(loop)):
                key = _edge_key(loop[k - 1], loop[k])
                t_vals = _points_on_segment([pt], p_arr[key[0]],
                                            p_arr[key[1]])
                if t_vals:
                    known = edge_t.setdefault(key, [])
                    edge_len = np.linalg.norm(p_arr[key[1]] - p_arr[key[0]])
                    if all([abs(t_vals[0] - t) * edge_len > tol_dup
                            for t, _ in known]):
                        known.append((t_vals[0], side))
                        n_found += 1
                    break
        else:
            f = f_list[0]
            if f in to_lower:
                axis, f = to_lower[f]
                pt[axis] -= lengths[axis]
            raw_face.setdefault(f, []).append((pt, side))
    for f, raw in raw_face.items():
        merged = _merge_face_points(raw, p_arr[polymesh.facets[f]], tol_dup)
        face_pts.setdefault(f, []).extend(merged)
        n_found += len(merged)
    return n_found


def _tet_info(pts, facets, facet_nums, holes, regions):
    """Build the TetGen input."""
    info = meshpy.tet.MeshInfo()
    info.set_points(pts)
    info.set_facets(facets, facet_nums)
    info.set_holes(holes)
    info.regions.resize(len(regions))
    for i, region in enumerate(regions):
        info.regions[i] = tuple(region)
    return info


def _build_periodic_3d(polymesh, phases, kps, pts, facet_nums, holes, regions,
                       opts, max_volume, max_edge_length):
    """Build a periodic tetrahedral mesh with TetGen, in two passes.

    In the first pass, the facets are triangulated (identically on
    opposite periodic faces) and TetGen meshes the domain as usual, adding
    points on the facets where its quality and size settings require it.
    The facets are then triangulated again with these points, the facets
    on opposite periodic faces getting the points of both, and TetGen
    meshes the domain without changing the facets (-Y): the mesh then has
    matching nodes on opposite faces, facets refined as in the first pass,
    and TetGen still refines the interior of the cells.

    Returns:
        The mesh built by MeshPy.

    """
    pts_1, facets_1, nums_1 = _triangulate_facets_3d(
        polymesh, phases, kps, pts, facet_nums, max_volume, max_edge_length,
        {}, {})
    info = _tet_info(pts_1, facets_1, nums_1, holes, regions)
    tri_mesh = meshpy.tet.build(info, options=opts)
    tri_pts = np.array(tri_mesh.points)
    n_in = len(pts_1)
    if len(tri_pts) < n_in or not np.allclose(tri_pts[:n_in], pts_1):
        raise RuntimeError('TetGen did not keep the input points.')

    edge_t = {}
    face_pts = {}
    _collect_facet_points_3d(tri_pts[n_in:], polymesh, edge_t, face_pts)
    pts_2, facets_2, nums_2 = _triangulate_facets_3d(
        polymesh, phases, kps, pts, facet_nums, max_volume, max_edge_length,
        edge_t, face_pts)
    info = _tet_info(pts_2, facets_2, nums_2, holes, regions)
    opts.nobisect = 1
    return meshpy.tet.build(info, options=opts)


def _build_2d(pts, facets, facet_nums, holes, regions, min_angle,
              allow_boundary_steiner):
    """Build a 2D mesh with Triangle."""
    info = meshpy.triangle.MeshInfo()
    info.set_points(pts)
    info.set_facets(facets, facet_nums)
    info.set_holes(holes)
    info.regions.resize(len(regions))
    for i, region in enumerate(regions):
        info.regions[i] = tuple(region)
    return meshpy.triangle.build(info, attributes=True,
                                 volume_constraints=True, max_volume=None,
                                 min_angle=min_angle, generate_faces=True,
                                 allow_boundary_steiner=allow_boundary_steiner)


def _split_periodic_boundary_2d(tri_pts, pts, facets, facet_nums, polymesh,
                                n_input=None):
    """Add the points that Triangle put on the facets to the facets.

    Triangle refines the segments of a mesh where its quality and size
    settings require it, but not the same way on opposite periodic faces.
    The points it added on a facet of a periodic face and on the image of
    the facet on the opposite face are inserted in both facets, as images
    of each other, so that the next mesh has matching nodes on opposite
    faces. The points of the two facets are merged so that the facets are
    as refined as the finest of the two (see :func:`_merge_params`). The
    points added on the other facets (the other walls, the facets between
    cells, and the facets of the copies of the cells outside the domain)
    are inserted as they are, and the points added inside the cells are
    appended as points of the input, so that the next mesh contains the
    previous one.

    Args:
        tri_pts (numpy.ndarray): Points of the mesh built by Triangle.
        pts (list): Points of the mesher input.
        facets (list): Facets (segments) of the mesher input.
        facet_nums (list): Polymesh facet number + 1 of each facet.
        polymesh (PolyMesh): The periodic polymesh.
        n_input (int): *(optional)* Number of points of the input of the
            mesh that Triangle built; the points after them are the ones
            it added. Defaults to the number of ``pts``.

    Returns:
        tuple: The new points, facets and facet numbers, and the number of
        points that Triangle added on the periodic faces.

    """
    pts = [list(p) for p in pts]
    if n_input is None:
        n_input = len(pts)
    new_pts = np.array(tri_pts)[n_input:]
    p_arr = np.array(polymesh.points)
    mins = p_arr.min(axis=0)
    lengths = p_arr.max(axis=0) - mins
    scale = lengths.max()
    tol = 1e-9 * scale
    per_axes = polymesh.periodic_axes

    # the facets (those of the copied cells too), with the parameters of
    # the new points on them
    seg_t = {}
    ends = {}
    periodic_segs = set()
    on_facets = set()
    for i, f_num in enumerate(facet_nums):
        pt_a, pt_b = np.array(pts[facets[i][0]]), np.array(pts[facets[i][1]])
        seg_t[i], ids = _points_on_segment(new_pts, pt_a, pt_b, True)
        on_facets.update(ids)
        ends[i] = (pt_a, pt_b)
        if f_num <= 0:
            continue  # a facet of a copied cell, outside the domain
        wall = min(polymesh.facet_neighbors[f_num - 1])
        if wall < 0 and per_axes[_misc.wall_axis_side(wall)[0]]:
            periodic_segs.add(i)
    n_new = sum([len(seg_t[i]) for i in periodic_segs])

    # the points that are not on a facet
    free = [k for k in range(len(new_pts)) if k not in on_facets]
    free_pts = new_pts[free].tolist()

    # a facet on a lower periodic face and its image on the upper face,
    # matched by their midpoints
    seg_ids = sorted(periodic_segs)
    mids = np.array([0.5 * (ends[i][0] + ends[i][1]) for i in seg_ids])
    tree = cKDTree(mids) if seg_ids else None
    pair_of = {}
    for i in seg_ids:
        pt_a, pt_b = ends[i]
        for axis, flag in enumerate(per_axes):
            if not flag or not np.allclose([pt_a[axis], pt_b[axis]],
                                           mins[axis], atol=tol):
                continue
            shift = np.zeros(len(mins))
            shift[axis] = lengths[axis]
            dist, k = tree.query(0.5 * (pt_a + pt_b) + shift)
            j = seg_ids[k]
            if dist <= tol and j != i:
                same = np.allclose(ends[j][0], pt_a + shift, atol=tol)
                pair_of[i] = (j, shift, same)
    is_upper = set([j for j, _, _ in pair_of.values()])

    def chain(kp_a, ids, kp_b):
        kp_list = [kp_a] + ids + [kp_b]
        return [[kp_list[k], kp_list[k + 1]] for k in range(len(kp_list) - 1)]

    new_facets = []
    new_nums = []
    for i, (facet, f_num) in enumerate(zip(facets, facet_nums)):
        if i not in seg_t:
            new_facets.append(list(facet))
            new_nums.append(f_num)
            continue
        if i in is_upper:
            continue
        pt_a, pt_b = ends[i]
        t_vals = list(seg_t[i])
        sides = [0] * len(t_vals)
        if i in pair_of:
            j, shift, same = pair_of[i]
            t_vals += [t if same else 1 - t for t in seg_t[j]]
            sides += [1] * len(seg_t[j])
        ids = []
        for t_val in _merge_params(t_vals, sides):
            ids.append(len(pts))
            pts.append((pt_a + t_val * (pt_b - pt_a)).tolist())
        new_facets.extend(chain(facet[0], ids, facet[1]))
        new_nums.extend([f_num] * (len(ids) + 1))
        if i in pair_of:
            j, shift, same = pair_of[i]
            im_ids = []
            for pid in ids:
                im_ids.append(len(pts))
                pts.append((np.array(pts[pid]) + shift).tolist())
            if not same:
                im_ids = im_ids[::-1]
            new_facets.extend(chain(facets[j][0], im_ids, facets[j][1]))
            new_nums.extend([facet_nums[j]] * (len(im_ids) + 1))
    return pts + free_pts, new_facets, new_nums, n_new


def _unmatched_periodic_nodes(pts, polymesh):
    """Nodes on a periodic face without an image on the opposite face.

    Returns:
        list: Tuples of the node number, the axis and the value of the
        coordinate of the opposite face.

    """
    pts = np.asarray(pts, dtype='float')
    p_arr = np.array(polymesh.points)
    mins = p_arr.min(axis=0)
    lengths = p_arr.max(axis=0) - mins
    tol = 1e-9 * lengths.max()
    unmatched = []
    for axis, flag in enumerate(polymesh.periodic_axes):
        if not flag:
            continue
        others = [i for i in range(len(mins)) if i != axis]
        for value, opposite in ((mins[axis], mins[axis] + lengths[axis]),
                                (mins[axis] + lengths[axis], mins[axis])):
            on_face = np.nonzero(np.abs(pts[:, axis] - value) <= tol)[0]
            on_opp = np.nonzero(np.abs(pts[:, axis] - opposite) <= tol)[0]
            if len(on_opp) == 0:
                unmatched.extend([(kp, axis, opposite) for kp in on_face])
                continue
            tree = cKDTree(pts[on_opp][:, others])
            dists, _ = tree.query(pts[on_face][:, others])
            for kp, dist in zip(on_face, dists):
                if dist > tol:
                    unmatched.append((kp, axis, opposite))
    return unmatched


def _mirror_boundary_points_2d(tri_pts, tri_elems, tri_e_atts, polymesh):
    """Give every node on a periodic face an image on the opposite face.

    A node without an image is mirrored by splitting the boundary edge of
    the opposite face, and the triangle behind it, at the image. This is
    only needed for the few points that Triangle keeps adding on the
    periodic faces when their refinement does not converge.

    Returns:
        tuple: The points, elements and element attributes, and the number
        of nodes mirrored.

    """
    pts = [list(p) for p in tri_pts]
    elems = [list(e) for e in tri_elems]
    atts = list(tri_e_atts)
    p_arr = np.array(polymesh.points)
    tol = 1e-9 * (p_arr.max(axis=0) - p_arr.min(axis=0)).max()
    n_mirrored = 0
    for kp, axis, opposite in _unmatched_periodic_nodes(tri_pts, polymesh):
        other = 1 - axis
        image = np.array(pts[kp])
        image[axis] = opposite
        # a node added by the mirroring of another node may be its image
        arr = np.array(pts)
        on_opp = np.abs(arr[:, axis] - opposite) <= tol
        if np.any(np.abs(arr[on_opp, other] - image[other]) <= tol):
            continue
        new_kp = len(pts)
        split = False
        for e_num, elem in enumerate(elems):
            e_pts = np.array([pts[k] for k in elem])
            on_line = np.abs(e_pts[:, axis] - opposite) <= tol
            if np.sum(on_line) != 2:
                continue
            k_a, k_b = [elem[k] for k in np.nonzero(on_line)[0]]
            v_a, v_b = pts[k_a][other], pts[k_b][other]
            if not (min(v_a, v_b) + tol < image[other] <
                    max(v_a, v_b) - tol):
                continue
            pts.append(image.tolist())
            # keep the orientation of the split triangle
            order = list(elem)
            i_a, i_b = order.index(k_a), order.index(k_b)
            tri_1 = list(order)
            tri_1[i_b] = new_kp
            tri_2 = list(order)
            tri_2[i_a] = new_kp
            elems[e_num] = tri_1
            elems.append(tri_2)
            atts.append(atts[e_num])
            split = True
            break
        if not split:
            e_str = 'A node on a periodic face has no image and cannot be '
            e_str += 'mirrored.'
            raise RuntimeError(e_str)
        n_mirrored += 1
    return (np.array(pts), np.array(elems), np.array(atts, dtype='int'),
            n_mirrored)


def _ghost_layer(polymesh, phases, labels, kps, pts, facets, facet_nums,
                 regions, holes, max_volume):
    """Add periodic images of the cells outside the periodic faces.

    The cells that touch a periodic face are copied outside that face,
    translated by the length of the domain along the axis (and, for the
    cells that touch several periodic faces, along each combination of
    the axes). The mesher then sees the same geometry on both sides of a
    periodic face, and around both faces of a pair, and refines them the
    same way. The elements outside the domain are removed after meshing.

    Args:
        polymesh (PolyMesh): The periodic polymesh.
        phases (list): The phases.
        labels (numpy.ndarray): The label of each region of the polymesh.
        kps (dict): Maps polymesh point numbers to mesher point numbers.
        pts (list): Points of the mesher input.
        facets (list): Facets of the mesher input.
        facet_nums (list): Polymesh facet number + 1 of each facet.
        regions (list): Region points of the mesher input.
        holes (list): Hole points of the mesher input.
        max_volume (float): The default maximum volume of the elements.

    Returns:
        tuple: The points, facets, facet numbers (0 for the facets of the
        copies), region points and holes, extended with the copies.

    """
    pts = [list(p) for p in pts]
    facets = [list(f) for f in facets]
    facet_nums = list(facet_nums)
    regions = [list(r) for r in regions]
    holes = [list(h) for h in holes]
    p_arr = np.array(polymesh.points)
    n_dim = p_arr.shape[1]
    lengths = p_arr.max(axis=0) - p_arr.min(axis=0)
    per_axes = polymesh.periodic_axes

    # the periodic faces touched by each cell, and the translations of its
    # copies: +1 moves the cell by the domain length, from the lower face
    touched = {}
    for f_num, neighs in enumerate(polymesh.facet_neighbors):
        wall = min(neighs)
        if wall < 0:
            axis, side = _misc.wall_axis_side(wall)
            if per_axes[axis]:
                sign = 1 if side == 0 else -1
                touched.setdefault(max(neighs), {})[axis] = sign
    copies = {}
    for reg, signs in touched.items():
        axes = sorted(signs)
        copies[reg] = []
        for n_sel in range(1, len(axes) + 1):
            for combo in itertools.combinations(axes, n_sel):
                copies[reg].append(tuple([signs[a] if a in combo else 0
                                          for a in range(n_dim)]))

    # points by location, so that a location reached from a point and from
    # its periodic image, with different translations, is one point
    scale = lengths.max()
    by_location = {}
    for i, pt in enumerate(pts):
        by_location.setdefault(tuple(np.round(np.array(pt) / scale, 9)), i)
    ghost_pts = {}

    def image_id(kp, trans):
        # the mesher point of a polymesh point moved by a translation
        key = (kp, trans)
        if key in ghost_pts:
            return ghost_pts[key]
        shift = np.array([s * lengths[a] for a, s in enumerate(trans)])
        new_pt = p_arr[kp] + shift
        loc = tuple(np.round(new_pt / scale, 9))
        if loc not in by_location:
            by_location[loc] = len(pts)
            pts.append(new_pt.tolist())
        ghost_pts[key] = by_location[loc]
        return ghost_pts[key]

    existing = set([tuple(sorted(f)) for f in facets])
    done = set()
    for reg, trans_list in copies.items():
        phase = phases[polymesh.phase_numbers[reg]]
        mat_type = phase.get('material_type', 'solid')
        reg_kps = set([kp for f in polymesh.regions[reg]
                       for kp in polymesh.facets[f]])
        center = p_arr[sorted(reg_kps)].mean(axis=0)
        for trans in trans_list:
            shift = np.array([s * lengths[a] for a, s in enumerate(trans)])
            cen = (center + shift).tolist()
            if mat_type in _misc.kw_void:
                holes.append(cen)
            else:
                regions.append(cen + [int(labels[reg]),
                                      phase.get('max_volume', max_volume)])
            for f in polymesh.regions[reg]:
                if (f, trans) in done:
                    continue
                done.add((f, trans))
                neighs = polymesh.facet_neighbors[f]
                other = neighs[0] if neighs[1] == reg else neighs[1]
                # a facet removed between merged cells is removed between
                # their copies too, and the copy of a facet on a periodic
                # face, moved along the axis of that face only, is the
                # facet on the opposite face (possibly subdivided), which
                # the input already has; moved along other axes too, it
                # lies outside the domain and closes the copy
                if (other >= 0 and trans in copies.get(other, []) and
                        not facet_check(neighs, polymesh, phases)):
                    continue
                if other < 0:
                    axis = _misc.wall_axis_side(other)[0]
                    if (per_axes[axis] and trans[axis] != 0 and
                            not any([t for a, t in enumerate(trans)
                                     if a != axis])):
                        continue
                ids = [image_id(kp, trans) for kp in polymesh.facets[f]]
                key = tuple(sorted(ids))
                if key in existing:
                    continue
                existing.add(key)
                facets.append(ids)
                facet_nums.append(0)
    return pts, facets, facet_nums, regions, holes


def _build_periodic_2d(polymesh, phases, labels, kps, pts, facets,
                       facet_nums, holes, regions, min_angle, max_volume):
    """Build a periodic triangular mesh with Triangle, in two passes.

    The cells next to the periodic faces are copied outside the faces
    (see :func:`_ghost_layer`) and the mesh is built like a non-periodic
    one, so that Triangle refines both faces of a pair the same way, up
    to the order of its operations. If some nodes on the periodic faces
    have no image on the opposite face, all the points of the mesh become
    the input of the next pass: the points on the facets are put in the
    facets (those on a periodic face and on its image merged and put on
    both, as images of each other) and the others are points of the
    input, so that Triangle only refines the mesh around the points
    brought from the opposite faces, in the same surroundings (the copies)
    that produced them. Passing only the points on the faces back, and
    meshing the cells again from scratch, made Triangle split the narrow
    corners of the cells again at every pass, down to very small
    elements. The elements outside the domain are removed, and the few
    nodes that may remain without an image are mirrored by splitting the
    elements behind them.

    Returns:
        tuple: The points, elements and element attributes.

    """
    pts, facets, facet_nums, regions, holes = _ghost_layer(
        polymesh, phases, labels, kps, pts, facets, facet_nums, regions,
        holes, max_volume)
    tri_mesh = _build_2d(pts, facets, facet_nums, holes, regions, min_angle,
                         True)
    p_arr = np.array(polymesh.points)
    mins = p_arr.min(axis=0)
    maxs = p_arr.max(axis=0)
    tol = 1e-9 * (maxs - mins).max()
    for _ in range(_MAX_PERIODIC_PASSES):
        # the elements inside the domain
        all_pts = np.array(tri_mesh.points)
        tri_elems = np.array(tri_mesh.elements)
        tri_e_atts = np.array(tri_mesh.element_attributes, dtype='int')
        cens = all_pts[tri_elems].mean(axis=1)
        inside = np.all((cens >= mins - tol) & (cens <= maxs + tol), axis=1)
        tri_elems = tri_elems[inside]
        tri_e_atts = tri_e_atts[inside]
        used = np.unique(tri_elems)
        renum = np.full(len(all_pts), -1)
        renum[used] = np.arange(len(used))
        tri_pts = all_pts[used]
        tri_elems = renum[tri_elems]
        if not _unmatched_periodic_nodes(tri_pts, polymesh):
            break

        # all the points of the mesh become the input of the next pass
        n_input = len(pts)
        if (len(all_pts) < n_input or
                not np.allclose(all_pts[:n_input], pts, atol=tol)):
            raise RuntimeError('Triangle did not keep the input points.')
        pts, facets, facet_nums, n_new = _split_periodic_boundary_2d(
            all_pts, pts, facets, facet_nums, polymesh, n_input)
        if n_new == 0:
            break
        tri_mesh = _build_2d(pts, facets, facet_nums, holes, regions,
                             min_angle, True)

    tri_pts, tri_elems, tri_e_atts, _ = _mirror_boundary_points_2d(
        tri_pts, tri_elems, tri_e_atts, polymesh)
    return tri_pts, tri_elems, tri_e_atts


def _facet_nonplanarity(polymesh):
    """Largest distance of a vertex to the plane of its facet.

    The facets of a periodic polymesh are planar only within the tolerance
    of the snapping of the points to the periodic faces. The geometric
    tests of the meshes against the polymesh use this distance as their
    tolerance.

    Returns:
        float: The distance, relative to the size of the domain.

    """
    pts = np.array(polymesh.points)
    scale = np.max(pts.max(axis=0) - pts.min(axis=0))
    max_dev = 0.0
    for facet in polymesh.facets:
        if len(facet) < 4:
            continue
        loop = pts[facet]
        normal = np.zeros(3)
        for i in range(len(loop)):
            normal += np.cross(loop[i - 1], loop[i])
        norm = np.linalg.norm(normal)
        if norm > 0:
            dev = np.abs((loop - loop[0]).dot(normal / norm)).max()
            max_dev = max(max_dev, dev)
    return max_dev / scale


def _attributes_from_polymesh(tri_pts, tri_elems, polymesh, labels):
    """Element attributes and facets of a mesh, from the polymesh geometry.

    Each element belongs to the (convex) cell of the polymesh that contains
    its centroid and its attribute is the label of that cell. The facets of
    the mesh are the faces between elements of cells with different labels
    and the faces on the boundary of the mesh; their attributes are the
    numbers of the polymesh facets they lie on.

    TetGen can leave some sub-faces of a facet unmarked when it may not
    modify the boundary (option -Y, used for periodic meshes). The region
    attributes it assigns then leak between the cells on either side of
    the facet and the facet is incomplete in its output. The geometry of
    the polymesh does not have this problem.

    Args:
        tri_pts (numpy.ndarray): The points of the mesh.
        tri_elems (numpy.ndarray): The elements of the mesh.
        polymesh (PolyMesh): The polygon/polyhedron mesh.
        labels (numpy.ndarray): The label of each region of the polymesh.

    Returns:
        tuple: The element attributes, the facets and the facet attributes.

    Raises:
        RuntimeError: If the mesh does not conform to the polymesh, i.e. an
            element centroid lies outside every cell or a face between two
            cells does not lie on a facet of the polymesh.
    """
    n_dim = tri_pts.shape[1]
    p_pts = np.array(polymesh.points)
    scale = np.max(p_pts.max(axis=0) - p_pts.min(axis=0))
    tol = max(1e-9, 4 * _facet_nonplanarity(polymesh)) * scale
    cell_geom = _CellGeometry(polymesh, p_pts)
    labels = np.array(labels)

    # 1. Cell containing the centroid of each element: the cell in which
    # the centroid is deepest
    cens = tri_pts[tri_elems].mean(axis=1)
    elem_regs = np.full(len(tri_elems), -1)
    depths = np.full(len(tri_elems), -np.inf)
    for r_num, r_i, dp in cell_geom.containing(cens, tol):
        depth = dp.min(axis=1)
        deeper = depth > depths[r_i]
        elem_regs[r_i[deeper]] = r_num
        depths[r_i[deeper]] = depth[deeper]
    n_outside = int(np.sum(depths < -tol))
    if n_outside > 0:
        e_str = 'The mesh does not conform to the polymesh: the centroids '
        e_str += 'of ' + str(n_outside) + ' elements are outside every '
        e_str += 'cell.'
        raise RuntimeError(e_str)
    elem_atts = labels[elem_regs]

    # 2. Faces of the elements, with the cells on either side
    n_elems = len(tri_elems)
    faces = np.concatenate([np.delete(tri_elems, k, axis=1)
                            for k in range(n_dim + 1)])
    faces.sort(axis=1)
    owners = np.tile(np.arange(n_elems), n_dim + 1)
    u_faces, inv, counts = np.unique(faces, axis=0, return_inverse=True,
                                     return_counts=True)
    if np.any(counts > 2):
        e_str = 'The mesh is not a manifold: a face is shared by more '
        e_str += 'than two elements.'
        raise RuntimeError(e_str)
    order = np.argsort(inv.reshape(-1), kind='stable')
    starts = np.cumsum(counts) - counts
    two = counts == 2
    r1 = elem_regs[owners[order[starts]]]
    r2 = np.full(len(u_faces), -1)
    r2[two] = elem_regs[owners[order[starts[two] + 1]]]

    # 3. Facets: faces between cells with different labels and faces on the
    # boundary, numbered by the polymesh facet they lie on. Cells with the
    # same label are not separated by facets of the mesh, so an element can
    # span several of them: the facet is found from the cells that contain
    # the center of the face, which lies on that facet.
    is_facet = np.full(len(u_faces), True)
    is_facet[two] = labels[r1[two]] != labels[r2[two]]
    f_ids = np.nonzero(is_facet)[0]
    f_cens = tri_pts[u_faces[f_ids]].mean(axis=1)
    claims = [[] for _ in f_ids]
    for r_num, c_i, dp in cell_geom.containing(f_cens, tol):
        for j in c_i[np.all(dp >= -tol, axis=1)]:
            claims[j].append(r_num)

    pair_facets = {}
    for f_num, neighs in enumerate(polymesh.facet_neighbors):
        if min(neighs) >= 0:
            pair_facets[(min(neighs), max(neighs))] = f_num
    e_str = 'The mesh does not conform to the polymesh: a face of the mesh '
    e_str += 'is not on a facet of the polymesh.'
    facets = []
    facet_atts = []
    for j, i in enumerate(f_ids):
        face = u_faces[i]
        f_pts = tri_pts[face]
        best_dist = float('inf')
        best_f = None
        for c_1 in [c for c in claims[j] if labels[c] == labels[r1[i]]]:
            f_nums, normals, centers = cell_geom.facets(c_1)
            rel_pos = f_pts[:, np.newaxis, :] - centers
            dists = np.abs(np.einsum('pfd,fd->pf', rel_pos,
                                     normals)).max(axis=0)
            if two[i]:
                cands = [pair_facets.get((min(c_1, c_2), max(c_1, c_2)))
                         for c_2 in claims[j] if labels[c_2] == labels[r2[i]]]
            else:
                # on the boundary of the domain, or of a void cell
                cands = [f for f in f_nums if
                         min(polymesh.facet_neighbors[f]) < 0 or
                         labels[polymesh.facet_neighbors[f][0]] !=
                         labels[polymesh.facet_neighbors[f][1]]]
            for f_num in cands:
                if f_num is None:
                    continue
                k = np.nonzero(f_nums == f_num)[0][0]
                if dists[k] < best_dist:
                    best_dist = dists[k]
                    best_f = int(f_num)
        if best_f is None or best_dist > max(1e-8 * scale, tol):
            raise RuntimeError(e_str)
        facets.append(face)
        facet_atts.append(best_f)
    facets = np.array(facets, dtype='int').reshape(-1, n_dim)
    return elem_atts, facets, np.array(facet_atts, dtype='int')


def _abaqus_periodic_nsets(mesh):
    """Abaqus node sets of the periodic faces of a mesh.

    For each periodic axis, two unsorted node sets are written,
    ``Set-N-Periodic-<axis>-Low`` and ``Set-N-Periodic-<axis>-High``, whose
    n-th entries are periodic images of each other (so that the pairs can
    be tied by equations).

    Args:
        mesh (TriMesh): The mesh.

    Returns:
        str: The ``*Nset`` blocks, or an empty string for a non-periodic
        mesh.

    """
    per_nodes = getattr(mesh, 'periodic_nodes', None)
    if not per_nodes:
        return ''
    abaqus = ''
    n_per = 16
    for axis in sorted(per_nodes):
        pairs = per_nodes[axis]
        if not pairs:
            continue
        axis_name = 'XYZ'[axis]
        for side, kps in (('Low', [lo for lo, _ in pairs]),
                          ('High', [hi for _, hi in pairs])):
            name = 'Set-N-Periodic-' + axis_name + '-' + side
            abaqus += '*Nset, nset=' + name + ', unsorted\n'
            for i in range(0, len(kps), n_per):
                chunk = kps[i:i + n_per]
                abaqus += ', '.join([str(int(kp) + 1) for kp in chunk])
                abaqus += '\n'
    return abaqus


def _abaqus_exterior_unions(polymesh, defined_surfs):
    """Abaqus surfaces that combine the facet surfaces on each domain face.

    Args:
        polymesh (PolyMesh): The polygon mesh, whose facet neighbors
            identify the facets on each face of the domain.
        defined_surfs (set): Facet numbers for which a 'Surface-<number>'
            surface has been written. Facets without elements in the mesh
            (e.g. on the boundary of voids) have no surface and are not
            included in the unions.

    Returns:
        str: The '*Surface, combine=union' blocks.

    """
    abaqus = ''
    poly_neighbors = np.array(polymesh.facet_neighbors)
    poly_mask = np.any(poly_neighbors < 0, axis=1)
    neigh_nums = np.min(poly_neighbors, axis=1)
    u_neighs = np.unique(neigh_nums[poly_mask])
    for neigh_num in u_neighs:
        f_nums = np.nonzero(neigh_nums == neigh_num)[0]
        members = [int(i) for i in f_nums if int(i) in defined_surfs]
        if not members:
            continue
        facet_name = 'Ext-Surface-' + str(-neigh_num)
        abaqus += '*Surface, name=' + facet_name + ', combine=union\n'
        abaqus += ''.join(['Surface-' + str(i) + '\n' for i in members])
    return abaqus


def _vtk_lines(values):
    """Join strings into lines of fewer than 80 characters."""
    lines = []
    line = ''
    for v_str in values:
        if not line:
            line = v_str
        elif len(line) + 1 + len(v_str) < 80:
            line += ' ' + v_str
        else:
            lines.append(line)
            line = v_str
    lines.append(line)
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- #
#                                                                             #
# Raster Mesh Helpers                                                         #
#                                                                             #
# --------------------------------------------------------------------------- #
# Offsets of the corner nodes of a pixel/voxel from its minimum corner, in
# element node order: counter-clockwise in 2D and, in 3D, nodes 1-4 on the
# bottom (-z) face counter-clockwise followed by nodes 5-8 on the top face,
# so that the element is right-handed (Abaqus CPS4 / C3D8 ordering).
_RASTER_CORNERS = {
    2: [(0, 0), (1, 0), (1, 1), (0, 1)],
    3: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
}

# Local node numbers of the faces of a pixel/voxel, keyed by (axis, side),
# where side 0 is the face at the minimum of the axis and side 1 the face
# at its maximum. The node order follows the Abaqus face definitions.
_RASTER_FACES = {
    2: {(0, 0): [3, 0], (0, 1): [1, 2], (1, 0): [0, 1], (1, 1): [2, 3]},
    3: {(0, 0): [3, 7, 4, 0], (0, 1): [1, 5, 6, 2],
        (1, 0): [0, 4, 5, 1], (1, 1): [2, 6, 7, 3],
        (2, 0): [0, 1, 2, 3], (2, 1): [4, 7, 6, 5]},
}

# Abaqus face ids (S1, S2, ...) of the faces above.
# CPS4: S1 = 1-2, S2 = 2-3, S3 = 3-4, S4 = 4-1
# C3D8: S1 = 1-2-3-4, S2 = 5-8-7-6, S3 = 1-5-6-2, S4 = 2-6-7-3,
#       S5 = 3-7-8-4, S6 = 4-8-5-1
_ABAQUS_FACE_IDS = {
    2: {(0, 0): 4, (0, 1): 2, (1, 0): 1, (1, 1): 3},
    3: {(0, 0): 6, (0, 1): 4, (1, 0): 3, (1, 1): 5, (2, 0): 1, (2, 1): 2},
}


class _CellGeometry(object):
    """Geometry of the (convex) cells of a polymesh, computed on demand.

    For each cell, the bounding box and the inward unit normals and centers
    of its facets are cached the first time they are needed.

    Args:
        polymesh (PolyMesh): The polygon/polyhedron mesh.
        p_pts (numpy.ndarray): The points of the polymesh, as an array.

    """
    def __init__(self, polymesh, p_pts):
        self.polymesh = polymesh
        self.p_pts = p_pts
        self._cache = {}

    def _compute(self, cell):
        region = self.polymesh.regions[cell]
        facets = self.polymesh.facets
        r_kps = np.unique([k for f in region for k in facets[f]])
        r_pts = self.p_pts[r_kps]
        r_cen = r_pts.mean(axis=0)

        normals = []
        centers = []
        for f in region:
            u_in, f_cen = _facet_in_normal(self.p_pts[facets[f]], r_cen)
            normals.append(u_in)
            centers.append(f_cen)
        limits = (r_pts.min(axis=0), r_pts.max(axis=0))
        self._cache[cell] = (np.array(region), np.array(normals),
                             np.array(centers), limits)

    def facets(self, cell):
        """Facet numbers, inward unit normals, and facet centers of a cell.
        """
        if cell not in self._cache:
            self._compute(cell)
        return self._cache[cell][:3]

    def limits(self, cell):
        """Bounding box (mins, maxs) of a cell."""
        if cell not in self._cache:
            self._compute(cell)
        return self._cache[cell][3]

    def containing(self, points, tol):
        """Cells whose bounding box contains some of the points.

        Args:
            points (numpy.ndarray): The points.
            tol (float): Tolerance of the bounding box test.

        Yields:
            tuple: The cell number, the indices of the points in its
            bounding box, and the signed distances of those points to the
            planes of the facets of the cell (one row per point, one
            column per facet, positive inside the cell).

        """
        for cell in range(len(self.polymesh.regions)):
            r_mins, r_maxs = self.limits(cell)
            in_box = np.all((points >= r_mins - tol) &
                            (points <= r_maxs + tol), axis=1)
            cand = np.nonzero(in_box)[0]
            if len(cand) == 0:
                continue
            _, normals, centers = self.facets(cell)
            rel_pos = points[cand][:, np.newaxis, :] - centers
            yield cell, cand, np.einsum('efd,fd->ef', rel_pos, normals)

    def exit_facet(self, cell, origin, direction):
        """Facet through which the ray origin + t * direction leaves a cell.

        Returns:
            tuple: The facet number and the value of t at the crossing, or
            (None, None) if the ray does not leave the cell.

        """
        f_nums, normals, centers = self.facets(cell)
        denom = normals.dot(direction)
        exiting = denom < 0
        if not np.any(exiting):
            return None, None

        t_vals = np.full(len(f_nums), float('inf'))
        rel_pos = centers[exiting] - origin
        t_vals[exiting] = np.einsum('ij,ij->i', rel_pos, normals[exiting])
        t_vals[exiting] /= denom[exiting]
        i_min = np.argmin(t_vals)
        return int(f_nums[i_min]), float(t_vals[i_min])


def _raster_facet_number(r1, r2, c1, c2, polymesh, phases, cell_geom,
                         pair_facets):
    """Polymesh facet approximated by the face between two pixels/voxels.

    The first pixel is centered at ``c1``, inside cell ``r1``, and the second
    at ``c2``, inside cell ``r2`` (``r2 < 0`` if it is not in any cell).
    If the cells are neighbors, the facet between them is returned.
    Otherwise, the facets crossed by the segment from ``c1`` to ``c2`` are
    found by walking through the cells of the polymesh, and the one that
    the mesh keeps (see :func:`facet_check`) closest to the face between
    the pixels is returned. Returns None if no facet is found.
    """
    if r2 >= 0:
        key = (min(r1, r2), max(r1, r2))
        if key in pair_facets:
            return pair_facets[key]

    direction = np.asarray(c2, dtype='float') - np.asarray(c1, dtype='float')
    cell = r1
    t_prev = 0
    crossed = []
    for _ in range(len(polymesh.regions)):
        f_num, t = cell_geom.exit_facet(cell, c1, direction)
        if f_num is None or t < t_prev - 1e-9:
            break
        if r2 >= 0 and t > 1 + 1e-6:
            break
        crossed.append((t, f_num))
        t_prev = t

        neighs = polymesh.facet_neighbors[f_num]
        if cell not in neighs:
            break
        nxt = neighs[1] if neighs[0] == cell else neighs[0]
        if nxt < 0 or nxt == r2:
            break
        cell = nxt

    if not crossed:
        return None
    ranked = [(abs(t - 0.5), f) for t, f in crossed if
              facet_check(polymesh.facet_neighbors[f], polymesh, phases)]
    if not ranked:
        ranked = [(abs(t - 0.5), f) for t, f in crossed]
    return min(ranked)[1]


def _raster_facets(polymesh, phases, cell_geom, elems, elem_grid, elem_regs,
                   keep, cens, mesh_size):
    """Facets of a raster mesh, with polymesh facet numbers as attributes.

    A facet is created on the face between two kept pixels of different
    cells when the polymesh facet between them is kept in the mesh (see
    :func:`facet_check`), on the face between a kept pixel and a removed
    one (void, or outside the domain), and on the faces of the kept pixels
    on the boundary of the grid.

    Args:
        polymesh (PolyMesh): The polygon/polyhedron mesh.
        phases (list): Phase dictionaries.
        cell_geom (_CellGeometry): Geometry of the cells of the polymesh.
        elems (numpy.ndarray): Nodes of each element, in face order.
        elem_grid (numpy.ndarray): Element numbers on the pixel grid.
        elem_regs (numpy.ndarray): Polymesh cell of each element (-1 if
            the center is not in any cell).
        keep (numpy.ndarray): Mask of the elements kept in the mesh.
        cens (numpy.ndarray): Centers of the elements.
        mesh_size (float): Side length of the pixels/voxels.

    Returns:
        tuple: Arrays of facets and facet attributes.

    """
    n_dim = elem_grid.ndim
    faces = _RASTER_FACES[n_dim]

    # Polymesh facets between pairs of cells and on the domain boundary
    pair_facets = {}
    bnd_facets = {}
    for f_num, neighs in enumerate(polymesh.facet_neighbors):
        n1, n2 = neighs
        if min(n1, n2) < 0:
            bnd_facets[(max(n1, n2), min(n1, n2))] = f_num
        else:
            pair_facets[(min(n1, n2), max(n1, n2))] = f_num
    args = (polymesh, phases, cell_geom, pair_facets)

    facets = []
    facet_atts = []
    for axis in range(n_dim):
        direction = np.zeros(n_dim)
        direction[axis] = 1

        # Faces between neighboring pixels along this axis
        sl_lo = [slice(None)] * n_dim
        sl_hi = [slice(None)] * n_dim
        sl_lo[axis] = slice(0, -1)
        sl_hi[axis] = slice(1, None)
        e_lo = elem_grid[tuple(sl_lo)].ravel()
        e_hi = elem_grid[tuple(sl_hi)].ravel()
        r_lo = elem_regs[e_lo]
        r_hi = elem_regs[e_hi]
        mask = (keep[e_lo] | keep[e_hi]) & (r_lo != r_hi)
        for e1, e2 in zip(e_lo[mask], e_hi[mask]):
            r1 = elem_regs[e1]
            r2 = elem_regs[e2]
            if keep[e1] and keep[e2]:
                if not facet_check([r1, r2], polymesh, phases):
                    continue
            if keep[e1]:
                f_num = _raster_facet_number(r1, r2, cens[e1], cens[e2],
                                             *args)
                facet = elems[e1][faces[(axis, 1)]]
            else:
                f_num = _raster_facet_number(r2, r1, cens[e2], cens[e1],
                                             *args)
                facet = elems[e2][faces[(axis, 0)]]
            if f_num is not None:
                facets.append(facet)
                facet_atts.append(f_num)

        # Faces on the boundary of the grid: the neighbor id of the domain
        # boundary facets is -1 (-x), -2 (+x), -3 (-y), ..., -6 (+z)
        for side in (0, 1):
            sl_bnd = [slice(None)] * n_dim
            sl_bnd[axis] = -side
            e_bnd = elem_grid[tuple(sl_bnd)].ravel()
            face_id = -(2 * axis + 1 + side)
            sgn = 2 * side - 1
            for e1 in e_bnd[keep[e_bnd]]:
                r1 = elem_regs[e1]
                f_num = bnd_facets.get((r1, face_id))
                if f_num is None:
                    c2 = cens[e1] + sgn * mesh_size * direction
                    f_num = _raster_facet_number(r1, -1, cens[e1], c2, *args)
                if f_num is not None:
                    facets.append(elems[e1][faces[(axis, side)]])
                    facet_atts.append(f_num)

    n_fkp = len(faces[(0, 0)])
    facets = np.array(facets, dtype='int').reshape(-1, n_fkp)
    facet_atts = np.array(facet_atts, dtype='int')
    return facets, facet_atts


def _plot_2d(ax, mesh, index_by, **kwargs):
    simps = np.array(mesh.elements)
    pts = np.array(mesh.points)
    xy = pts[simps, :]

    plt_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, (list, np.ndarray)):
            plt_value = []
            for e_num, e_att in enumerate(mesh.element_attributes):
                if index_by == 'element':
                    ind = e_num
                elif index_by == 'attribute':
                    ind = int(e_att)
                else:
                    e_str = 'Cannot index by {}.'.format(index_by)
                    raise ValueError(e_str)
                try:
                    v = value[ind]
                except IndexError:
                    v = 'none'
                plt_value.append(v)
        else:
            plt_value = value
        plt_kwargs[key] = plt_value

    pc = collections.PolyCollection(xy, **plt_kwargs)
    ax.add_collection(pc)
    ax.autoscale_view()
