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

import meshpy.tet
import meshpy.triangle
import numpy as np
import pygmsh as pg
from matplotlib import collections
from matplotlib import patches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
                along grain boundaries. This option is used  with Triangle
                and gmsh. Defaults to infinity, which turns off this control.
            mesh_size (float): The target size of the mesh elements. This
                option is used with gmsh. Default is infinity, whihch turns off
                this control.

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

        mesh = cls(*tri_args)
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
        pts, per_nodes = _misc.pair_periodic_points(self.points, per_axes,
                                                    dom_lims)
        if self.facets is None:
            per_facets = {axis: [] for axis in per_nodes}
        else:
            per_facets = _misc.pair_periodic_facets(self.facets, per_nodes)

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
    elif periodic:
        # TetGen triangulates polygonal facets itself, so the facets on
        # opposite periodic faces are triangulated here, identically, and
        # passed as triangles
        facets, facet_nums = _triangulate_periodic_facets(polymesh, kps,
                                                          facets, facet_nums)

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

    # build inputs
    if n_dim == 2:
        info = meshpy.triangle.MeshInfo()
    else:
        info = meshpy.tet.MeshInfo()

    info.set_points(pts)
    info.set_facets(facets, facet_nums)
    info.set_holes(holes)

    info.regions.resize(len(regions))
    for i, r in enumerate(regions):
        info.regions[i] = tuple(r)

    # run MeshPy
    # The maximum element volume is set per region above, using the global
    # value as the default for the phases that do not set their own. Only
    # these regional constraints are passed to Triangle/TetGen: a fixed
    # (global) constraint would cap the per-phase values and, in 2D, an
    # infinite one is formatted as 'ainf', which Triangle reads as the
    # switches -a -i -n -f.
    # A periodic mesh must keep the nodes of the boundary facets as they are
    # (Triangle's -Y switch), so that opposite faces have matching nodes.
    if n_dim == 2:
        tri_mesh = meshpy.triangle.build(info,
                                         attributes=True,
                                         volume_constraints=True,
                                         max_volume=None,
                                         min_angle=min_angle,
                                         generate_faces=True,
                                         allow_boundary_steiner=not periodic)
    else:
        opts = meshpy.tet.Options('pq')
        opts.mindihedral = min_angle
        opts.varvolume = 1
        opts.fixedvolume = 0
        opts.regionattrib = 1
        opts.facesout = 1
        if periodic:
            opts.nobisect = 1  # -Y: keep the boundary facets as given
        tri_mesh = meshpy.tet.build(info, options=opts)

    # return mesh
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

    parent = np.arange(len(seed_nums))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        r_i, r_j = find(i), find(j)
        if r_i != r_j:
            parent[max(r_i, r_j)] = min(r_i, r_j)

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
            union(r_a, r_b)

    first_region = {}
    for r, s in enumerate(seed_nums):
        if s in first_region:
            union(first_region[s], r)
        else:
            first_region[s] = r

    roots = np.array([find(r) for r in range(len(seed_nums))])
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


def _triangulate_periodic_facets(polymesh, kps, facets, facet_nums):
    """Triangulate the facets on the periodic faces of a 3D polymesh.

    Each facet on a lower periodic face is split into a fan of triangles
    and its image on the upper face into the corresponding triangles (the
    images of the same points), so that TetGen, which keeps the boundary
    facets as given with the -Y switch, produces matching triangles on
    opposite faces.

    Args:
        polymesh (PolyMesh): The periodic polymesh.
        kps (dict): Maps polymesh point numbers to the point numbers of the
            mesher input.
        facets (list): Facets of the mesher input (lists of point numbers).
        facet_nums (list): Polymesh facet number + 1 of each facet.

    Returns:
        tuple: The new facets and facet numbers.

    """
    f_index = {f_num - 1: i for i, f_num in enumerate(facet_nums)}
    replaced = {}
    for axis, f_pairs in (polymesh.periodic_facets or {}).items():
        kp_map = dict(polymesh.periodic_points[axis])
        for f_lo, f_hi in f_pairs:
            if f_lo not in f_index or f_hi not in f_index:
                continue
            loop_lo = polymesh.facets[f_lo]
            loop_hi = [kp_map[kp] for kp in loop_lo]
            tris_lo = [[kps[loop_lo[0]], kps[loop_lo[k]], kps[loop_lo[k + 1]]]
                       for k in range(1, len(loop_lo) - 1)]
            tris_hi = [[kps[loop_hi[0]], kps[loop_hi[k]], kps[loop_hi[k + 1]]]
                       for k in range(1, len(loop_hi) - 1)]
            replaced[f_index[f_lo]] = tris_lo
            replaced[f_index[f_hi]] = tris_hi

    new_facets = []
    new_nums = []
    for i, (facet, f_num) in enumerate(zip(facets, facet_nums)):
        if i in replaced:
            for tri in replaced[i]:
                new_facets.append(tri)
                new_nums.append(f_num)
        else:
            new_facets.append(facet)
            new_nums.append(f_num)
    return new_facets, new_nums


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
