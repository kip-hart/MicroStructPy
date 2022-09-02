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
                 facet_attributes=None):
        self.points = points
        self.elements = elements
        self.element_attributes = element_attributes
        self.facets = facets
        self.facet_attributes = facet_attributes

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
            for line in file.readlines():
                if 'Mesh Points'.lower() in line.lower():
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
                    else:
                        pass

        # check the inputs
        assert len(pts) == n_pts
        assert len(elems) == n_elems
        assert len(elem_atts) == n_eas
        assert len(facets) == n_facets
        assert len(facet_atts) == n_fas

        return cls(pts, elems, elem_atts, facets, facet_atts)

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
        key = mesher.lower().strip()
        if key in ('triangle/tetgen', 'triangle', 'tetgen'):
            tri_args = _call_meshpy(polymesh, phases, min_angle, max_volume,
                                    max_edge_length)
        elif key == 'gmsh':
            tri_args = _call_gmsh(polymesh, phases, mesh_size, max_edge_length)

        return cls(*tri_args)

    # ----------------------------------------------------------------------- #
    # String and Representation Functions                                     #
    # ----------------------------------------------------------------------- #
    def __str__(self):
        nv = len(self.points)
        nd = len(self.points[0])
        pt_fmt = '\t'
        pt_fmt += ', '.join(['{pt[' + str(i) + ']: e}' for i in range(nd)])

        str_str = 'Mesh Points: ' + str(nv) + '\n'
        str_str += ''.join([pt_fmt.format(pt=p) + '\n' for p in self.points])

        str_str += 'Mesh Elements: ' + str(len(self.elements)) + '\n'
        str_str += '\n'.join(['\t' + str(tuple(e))[1:-1] for e in
                              self.elements])

        try:
            str_str += '\nElement Attributes: '
            str_str += str(len(self.element_attributes)) + '\n'
            str_str += '\n'.join(['\t' + str(a) for a in
                                  self.element_attributes])
        except TypeError:
            pass

        try:
            str_str += '\nFacets: ' + str(len(self.facets)) + '\n'
            str_str += '\n'.join(['\t' + str(tuple(f))[1:-1] for f in
                                  self.facets])
        except TypeError:
            pass

        try:
            str_str += '\nFacet Attributes: '
            str_str += str(len(self.facet_attributes)) + '\n'
            str_str += '\n'.join(['\t' + str(a) for a in
                                  self.facet_attributes])
        except TypeError:
            pass

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

            # Element sets - seed number
            elset_n_per = 16
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
            facets = np.array(self.facets)
            facet_atts = np.array(self.facet_attributes)

            face_ids = {2: [2, 3, 1], 3: [3, 4, 2, 1]}[n_dim]

            for att in np.unique(facet_atts):
                facet_name = 'Surface-' + str(att)
                surf_str = '*Surface, name=' + facet_name + ', type=element\n'

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

            # Surfaces - Exterior
            poly_neighbors = np.array(polymesh.facet_neighbors)
            poly_mask = np.any(poly_neighbors < 0, axis=1)
            neigh_nums = np.min(poly_neighbors, axis=1)
            u_neighs = np.unique(neigh_nums[poly_mask])
            for neigh_num in u_neighs:
                mask = neigh_nums == neigh_num
                facet_name = 'Ext-Surface-' + str(-neigh_num)
                surf_str = '*Surface, name=' + facet_name + ', combine=union\n'
                for i, flag in enumerate(mask):
                    if flag:
                        surf_str += 'Surface-' + str(i) + '\n'
                abaqus += surf_str

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
            # create boundary markers
            bnd_mkrs = np.full(len(self.points), 0, dtype='int')

            facet_arr = np.array(self.facets)
            f_bnd_mkrs = np.full(len(self.facets), 0, dtype='int')
            elem_arr = np.array(self.elements)
            for elem in self.elements:
                for i in range(len(elem)):
                    e_facet = np.delete(elem, i)
                    f_mask = np.full(elem_arr.shape[0], True)
                    for kp in e_facet:
                        f_mask &= np.any(elem_arr == kp, axis=-1)

                    if np.sum(f_mask) == 1:
                        bnd_mkrs[e_facet] = 1

                        f_mask = np.full(facet_arr.shape[0], True)
                        for kp in e_facet:
                            f_mask &= np.any(facet_arr == kp, axis=-1)
                        f_bnd_mkrs[f_mask] = 1

            # write vertices
            n_pts, n_dim = np.array(self.points).shape
            nodes = ' '.join([str(n) for n in (n_pts, n_dim, 0, 1)]) + '\n'
            nodes += ''.join([str(i) + ''.join([' ' + str(x) for x in pt]) +
                              ' ' + str(bnd_mkrs[i]) + '\n' for i, pt in
                              enumerate(self.points)])

            with open(filename + '.node', 'w') as file:
                file.write(nodes)

            # write elements
            n_ele, n_kp = np.array(self.elements).shape
            is_att = self.element_attributes is not None
            n_att = int(is_att)
            eles = ' '.join([str(n) for n in (n_ele, n_kp, n_att)]) + '\n'
            for i, simplex in enumerate(self.elements):
                e_str = ' '.join([str(kp) for kp in simplex])
                if is_att:
                    e_str += ' ' + str(self.element_attributes[i])
                e_str += '\n'
                eles += e_str

            with open(filename + '.ele', 'w') as file:
                file.write(eles)

            # Write edges/faces
            if self.facets is not None:
                ext = {2: '.edge', 3: '.face'}[n_dim]

                n_facet, n_kp = np.array(self.facets).shape
                edge = ' '.join([str(n) for n in (n_facet, n_kp, 1)])
                edge += ''.join([str(i) + ''.join([' ' + str(k) for k in f]) +
                                 ' ' + str(mkr) + '\n' for f, mkr in
                                 zip(self.facets, f_bnd_mkrs)])
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
            try:
                int(self.element_attributes[0])
                att_type = 'int'
            except TypeError:
                att_type = 'float'

            vtk += '\nCELL_DATA ' + str(n_elem) + '\n'
            vtk += 'SCALARS element_attributes ' + att_type + ' 1 \n'
            vtk += 'LOOKUP_TABLE element_attributes\n'
            vtk += ''.join([str(a) + '\n' for a in self.element_attributes])

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
        if n_dim == 2:
            ax = plt.gca()
        else:
            ax = plt.gcf().gca(projection=Axes3D.name)
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
                if type(value) in (list, np.array):
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
                if type(value) not in (list, np.array):
                    for kws in p_kwargs:
                        kws[key] = value

                for i, m in enumerate(material):
                    if type(value) in (list, np.array):
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
# RasterMesh Class                                                               #
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
        # 1. Create node and element grids
        p_pts = np.array(polymesh.points)
        mins = p_pts.min(axis=0)
        maxs = p_pts.max(axis=0)
        lens = (maxs - mins)*(1 + 1e-9)
        sides = [lb + np.arange(0, dlen, mesh_size) for lb, dlen in
                 zip(mins, lens)]
        mgrid = np.meshgrid(*sides)
        nodes = np.array([g.flatten() for g in mgrid]).T
        node_nums = np.arange(mgrid[0].size).reshape(mgrid[0].shape)
        
        n_dim = len(mins)
        if n_dim == 2:
            m, n = node_nums.shape
            kp1 = node_nums[:(m-1), :(n-1)].flatten()
            kp2 = node_nums[1:m, :(n-1)].flatten()
            kp3 = node_nums[1:m, 1:n].flatten()
            kp4 = node_nums[:(m-1), 1:n].flatten()
            elems = np.array([kp1, kp2, kp3, kp4]).T
        elif n_dim == 3:
            m, n, p = node_nums.shape
            kp1 = node_nums[:(m-1), :(n-1), :(p-1)].flatten()
            kp2 = node_nums[1:m, :(n-1), :(p-1)].flatten()
            kp3 = node_nums[1:m, 1:n, :(p-1)].flatten()
            kp4 = node_nums[:(m-1), 1:n, :(p-1)].flatten()
            kp5 = node_nums[:(m-1), :(n-1), 1:p].flatten()
            kp6 = node_nums[1:m, :(n-1), 1:p].flatten()
            kp7 = node_nums[1:m, 1:n, 1:p].flatten()
            kp8 = node_nums[:(m-1), 1:n, 1:p].flatten()
            elems = np.array([kp1, kp2, kp3, kp4, kp5, kp6, kp7, kp8]).T

        else:
            raise NotImplementedError

        # 2. Compute element centers
        cens = nodes[elems[:, 0]] + 0.5 * mesh_size

        # 3. For each region:
        i_remain = np.arange(cens.shape[0])
        elem_regs = np.full(cens.shape[0], -1)
        elem_atts = np.full(cens.shape[0], -1)
        for r_num, region in enumerate(polymesh.regions):
            # A. Create a bounding box
            r_kps = np.unique([k for f in region for k in polymesh.facets[f]])
            r_pts = p_pts[r_kps]
            r_mins = r_pts.min(axis=0)
            r_maxs = r_pts.max(axis=0)

            # B. Isolate element centers with box
            r_i_remain = np.copy(i_remain)
            for i, lb in enumerate(r_mins):
                ub = r_maxs[i]
                x = cens[r_i_remain, i]
                in_range = (x >= lb) & (x <= ub)
                r_i_remain = r_i_remain[in_range]

            # C. For each facet, remove centers on the wrong side
            # note: regions are convex, so mean pt is on correct side of facets
            r_cen = r_pts.mean(axis=0)
            for f in region:
                f_kps = polymesh.facets[f]
                f_pts = p_pts[f_kps]
                u_in, f_cen = _facet_in_normal(f_pts, r_cen)

                rel_pos = cens[r_i_remain] - f_cen
                dp = rel_pos.dot(u_in)
                inside = dp >= 0
                r_i_remain = r_i_remain[inside]
            
            # D. Assign remaining centers to region
            elem_regs[r_i_remain] = r_num
            elem_atts[r_i_remain] = polymesh.seed_numbers[r_num]
            i_remain = np.setdiff1d(i_remain, r_i_remain)

        # 4. Combine regions of the same seed number
        if phases is not None:
            conv_dict = _amorphous_seed_numbers(polymesh, phases)
            elem_atts = np.array([conv_dict.get(s, s) for s in elem_atts])
        
        # 5. Define remaining facets, inherit their attributes
        facets = []
        facet_atts = []
        for f_num, f_neighs in enumerate(polymesh.facet_neighbors):
            n1, n2 = f_neighs
            if n1 >= 0:
                e1 = elems[elem_regs == n1]
                e2 = elems[elem_regs == n2]

                # Shift +x
                e1_s = e1[:, 1]
                e2_s = e2[:, 0]
                mask = np.isin(e1_s, e2_s)
                for elem in e1[mask]:
                    if n_dim == 2:
                        facet = elem[[1, 2]]
                    else:
                        facet = elem[[1, 2, 6, 5]]
                    facets.append(facet)
                    facet_atts.append(f_num)

                # Shift -x
                e1_s = e1[:, 0]
                e2_s = e2[:, 1]
                mask = np.isin(e1_s, e2_s)
                for elem in e1[mask]:
                    if n_dim == 2:
                        facet = elem[[3, 0]]
                    else:
                        facet = elem[[0, 4, 7, 3]]
                    facets.append(facet)
                    facet_atts.append(f_num)

                # Shift +y
                e1_s = e1[:, 3]
                e2_s = e2[:, 0]
                mask = np.isin(e1_s, e2_s)
                for elem in e1[mask]:
                    if n_dim == 2:
                        facet = elem[[2, 3]]
                    else:
                        facet = elem[[2, 3, 7, 6]]
                    facets.append(facet)
                    facet_atts.append(f_num)

                # Shift -y
                e1_s = e1[:, 0]
                e2_s = e2[:, 3]
                mask = np.isin(e1_s, e2_s)
                for elem in e1[mask]:
                    if n_dim == 2:
                        facet = elem[[0, 1]]
                    else:
                        facet = elem[[0, 1, 5, 4]]
                    facets.append(facet)
                    facet_atts.append(f_num)

                if n_dim < 3:
                    continue

                # Shift +z
                e1_s = e1[:, 4]
                e2_s = e1[:, 0]
                mask = np.isin(e1_s, e2_s)
                for elem in e1[mask]:
                    facet = elem[[4, 5, 6, 7]]
                    facets.append(facet)
                    facet_atts.append(f_num)

                # Shift -z
                e1_s = e1[:, 0]
                e2_s = e1[:, 4]
                mask = np.isin(e1_s, e2_s)
                for elem in e1[mask]:
                    facet = elem[[0, 1, 2, 3]]
                    facets.append(facet)
                    facet_atts.append(f_num)

            elif n1 == -1:
                # -x face
                e2 = elems[elem_regs == n2]
                x2 = nodes[e2[:, 0], 0]
                mask = np.isclose(x2, mins[0])
                for elem in e2[mask]:
                    if n_dim == 2:
                        facet = elem[[3, 0]]
                    else:
                        facet = elem[[0, 4, 7, 3]]
                    facets.append(facet)
                    facet_atts.append(f_num)

            elif n1 == -2:
                # +x face
                e2 = elems[elem_regs == n2]
                x2 = nodes[e2[:, 1], 0]
                mask = np.isclose(x2, maxs[0])
                for elem in e2[mask]:
                    if n_dim == 2:
                        facet = elem[[1, 2]]
                    else:
                        facet = elem[[1, 2, 6, 5]]
                    facets.append(facet)
                    facet_atts.append(f_num)

            elif n1 == -3:
                # -y face
                e2 = elems[elem_regs == n2]
                x2 = nodes[e2[:, 0], 1]
                mask = np.isclose(x2, mins[1])
                for elem in e2[mask]:
                    if n_dim == 2:
                        facet = elem[[0, 1]]
                    else:
                        facet = elem[[0, 1, 5, 4]]
                    facets.append(facet)
                    facet_atts.append(f_num)

            elif n1 == -4:
                # +y face
                e2 = elems[elem_regs == n2]
                x2 = nodes[e2[:, 2], 1]
                mask = np.isclose(x2, maxs[1])
                for elem in e2[mask]:
                    if n_dim == 2:
                        facet = elem[[2, 3]]
                    else:
                        facet = elem[[2, 3, 7, 6]]
                    facets.append(facet)
                    facet_atts.append(f_num)

            elif n1 == -5:
                # -z face
                e2 = elems[elem_regs == n2]
                x2 = nodes[e2[:, 0], 2]
                mask = np.isclose(x2, mins[2])
                for elem in e2[mask]:
                    facet = elem[[0, 1, 2, 3]]
                    facets.append(facet)
                    facet_atts.append(f_num)

            elif n1 == -6:
                # +z face
                e2 = elems[elem_regs == n2]
                x2 = nodes[e2[:, 4], 2]
                mask = x2 == maxs[2]
                for elem in e2[mask]:
                    facet = elem[[4, 5, 6, 7]]
                    facets.append(facet)
                    facet_atts.append(f_num)

        # 6. Remove voids and excess cells
        if phases is not None:
            att_rm = [-1]
            for i, phase in enumerate(phases):
                if phase.get('material_type', 'solid') in _misc.kw_void:
                    r_mask = np.array(polymesh.phase_numbers) == i
                    seeds = np.unique(np.array(polymesh.seed_numbers)[r_mask])
                    att_rm.extend(list(seeds))

            # Remove elements
            rm_mask = np.isin(elem_atts, att_rm)
            elems = elems[~rm_mask]
            elem_atts = elem_atts[~rm_mask]

            # Re-number nodes
            nodes_mask = np.isin(np.arange(nodes.shape[0]), elems)
            n_remain = np.sum(nodes_mask)
            node_n_conv = np.arange(nodes.shape[0])
            node_n_conv[nodes_mask] = np.arange(n_remain)

            nodes = nodes[nodes_mask]
            elems = node_n_conv[elems]
            if len(facets) > 0:
                f_keep = np.all(nodes_mask[facets], axis=1)
                facets = node_n_conv[np.array(facets)[f_keep, :]]
                facet_atts = np.array(facet_atts)[f_keep]

        return cls(nodes, elems, elem_atts, facets, facet_atts)

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
            abaqus += ''.join([str(i + 1) + ''.join([', ' + str(kp + 1) for kp
                                                     in elem]) + '\n' for
                               i, elem in enumerate(self.elements)])

            # Element sets - seed number
            elset_n_per = 16
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
            facets = np.array(self.facets)
            facet_atts = np.array(self.facet_attributes)

            face_ids = {2: [2, 3, 1], 3: [3, 4, 2, 1]}[n_dim]

            for att in np.unique(facet_atts):
                facet_name = 'Surface-' + str(att)
                surf_str = '*Surface, name=' + facet_name + ', type=element\n'

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

            # Surfaces - Exterior
            poly_neighbors = np.array(polymesh.facet_neighbors)
            poly_mask = np.any(poly_neighbors < 0, axis=1)
            neigh_nums = np.min(poly_neighbors, axis=1)
            u_neighs = np.unique(neigh_nums[poly_mask])
            for neigh_num in u_neighs:
                mask = neigh_nums == neigh_num
                facet_name = 'Ext-Surface-' + str(-neigh_num)
                surf_str = '*Surface, name=' + facet_name + ', combine=union\n'
                for i, flag in enumerate(mask):
                    if flag:
                        surf_str += 'Surface-' + str(i) + '\n'
                abaqus += surf_str

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
            pt_fmt = '{: f} {: f} {: f}\n'

            # Dimensions
            pts = np.array(self.points)
            coords = [np.unique(ax) for ax in pts.T]
            if len(coords) < 3:
                coords.append([0])  # force z=0 for 2D meshes
            dims = [len(c) for c in coords]
            n_dim = len(dims)


            # write heading
            vtk = '# vtk DataFile Version 2.0\n'
            vtk += '{} mesh\n'.format(mesh_type)
            vtk += 'ASCII\n'
            vtk += 'DATASET RECTILINEAR_GRID\n'
            vtk += 'DIMENSIONS {} {} {}\n'.format(*dims)

            # write points
            for ind, ax in enumerate(['X', 'Y', 'Z']):
                vtk += '{}_COORDINATES {} float\n'.format(ax, dims[ind])
                line = ''
                for x in coords[ind]:
                    x_str = '{:f}'.format(x)
                    if len(line) == 0:
                        line = x_str
                    elif len(line) + len(' ') + len(x_str) < 80:
                        line += ' ' + x_str
                    else:
                        vtk += line + '\n'
                        line = x_str
                vtk += line + '\n'

            # write element attributes
            vtk += 'CELL_DATA {}\n'.format(len(self.element_attributes))
            vtk += 'SCALARS element_attributes float\n'
            vtk += 'LOOKUP_TABLE default\n'
            line = ''
            phase_nums = ''
            phase_line = ''
            pts = np.array(self.points)
            elems = np.sort(self.elements)
            if len(coords[-1]) == 1: # 2D
                for y_ind in range(len(coords[1][:-1])):
                    y_mask_ind = pts[:, 1] == coords[1][y_ind]
                    y_mask_ip1 = pts[:, 1] == coords[1][y_ind]
                    y_mask = y_mask_ind | y_mask_ip1

                    for x_ind in range(len(coords[0][:-1])):
                        # mask self.points
                            x_mask_ind = pts[:, 0] == coords[0][x_ind]
                            x_mask_ip1 = pts[:, 0] == coords[0][x_ind + 1]
                            x_mask = x_mask_ind | x_mask_ip1

                            mask = x_mask & y_mask
                            el = np.where(mask)
                            e_ind = np.where(np.all(elems == el, axis=1))[0][0]

                            # element attribute
                            att = self.element_attributes[e_ind]
                            att_str = '{:f}'.format(att)
                            if len(line) == 0:
                                line += att_str
                            elif len(line) + len(' ') + len(att_str) < 80:
                                line += ' ' + att_str
                            else:
                                vtk += line + '\n'
                                line = att_str

                            # phase number
                            if seeds is not None:
                                phase = seeds[att].phase
                                p_str = str(int(phase))
                                if len(phase_line) == 0:
                                    phase_line = p_str
                                elif len(line) + len(' ') + len(p_str) < 80:
                                    phase_line += ' ' + p_str
                                else:
                                    phase_nums += phase_line + '\n'
                                    phase_line = p_str
                vtk += line + '\n'
                if seeds is not None:
                    vtk += 'SCALARS phase_numbers int\n'
                    vtk += 'LOOKUP_TABLE default\n'
                    vtk += phase_nums + phase_line + '\n'

            else:
                for z_ind in range(len(coords[2][:-1])):
                    z_mask_ind = pts[:, 2] == coords[2][z_ind]
                    z_mask_ip1 = pts[:, 2] == coords[2][z_ind + 1]
                    z_mask = z_mask_ind | z_mask_ip1

                    for y_ind in range(len(coords[1][:-1])):
                        y_mask_ind = pts[:, 1] == coords[1][y_ind]
                        y_mask_ip1 = pts[:, 1] == coords[1][y_ind + 1]
                        y_mask = y_mask_ind | y_mask_ip1

                        for x_ind in range(len(coords[0][:-1])):
                            # mask self.points
                            x_mask_ind = pts[:, 0] == coords[0][x_ind]
                            x_mask_ip1 = pts[:, 0] == coords[0][x_ind + 1]
                            x_mask = x_mask_ind | x_mask_ip1

                            mask = x_mask & y_mask & z_mask
                            el = np.where(mask)
                            e_ind = np.where(np.all(elems == el, axis=1))[0][0]

                            # element attribute
                            att = self.element_attributes[e_ind]
                            att_str = '{:f}'.format(att)
                            if len(line) == 0:
                                line += att_str
                            elif len(line) + len(' ') + len(att_str) < 80:
                                line += ' ' + att_str
                            else:
                                vtk += line + '\n'
                                line = att_str

                            # phase number
                            if seeds is not None:
                                phase = seeds[att].phase
                                p_str = str(int(phase))
                                if len(phase_line) == 0:
                                    phase_line = p_str
                                elif len(line) + len(' ') + len(p_str) < 80:
                                    phase_line += ' ' + p_str
                                else:
                                    phase_nums += phase_line + '\n'
                                    phase_line = p_str
                vtk += line + '\n'
                if seeds is not None:
                    vtk += 'SCALARS phase_numbers int\n'
                    vtk += 'LOOKUP_TABLE default\n'
                    vtk += phase_nums + phase_line + '\n'

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
        arr = np.full(inds_maxs + 1, -1)

        # 3. For each element: populate array with element attributes
        if element_attributes:
            vals = self.element_attributes
        else:
            vals = np.arange(elem_tups.shape[0])
        for t, v in zip(elem_tups, vals):
            arr[tuple(t)] = v

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
        if n_dim == 2:
            ax = plt.gca()
        else:
            ax = plt.gcf().gca(projection=Axes3D.name)
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

            inds = self.as_array(element_attributes=index_by=='attribute')
            plt_kwargs = {}
            for key, value in kwargs.items():
                if type(value) in (list, np.array):
                    plt_value = np.empty(inds.shape, dtype=object)
                    for i, val_i in enumerate(value):
                        plt_value[inds == i] = val_i
                    if 'color' in key:
                        unset_mask = plt_value == None
                        plt_value[unset_mask] = 'k'
                        inds[unset_mask] = -1

                else:
                    plt_value = value
                plt_kwargs[key] = plt_value

            # Scale axes
            pts = np.array(self.points)
            mins = pts.min(axis=0)
            sz = self.mesh_size
            pt_tups = np.round((pts - mins) / sz).astype(int)
            maxs = pt_tups.max(axis=0)
            grids = np.indices(maxs + 1, dtype=float)
            for pt, pt_tup in zip(pts, pt_tups):
                for i, x in enumerate(pt):
                    grids[i][tuple(pt_tup)] = x
            ax.voxels(*grids, inds >= 0, **plt_kwargs)

        # Add legend
        if material and index_by == 'attribute':
            p_kwargs = [{'label': m} for m in material]
            for key, value in kwargs.items():
                if type(value) not in (list, np.array):
                    for kws in p_kwargs:
                        kws[key] = value

                for i, m in enumerate(material):
                    if type(value) in (list, np.array):
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
                 max_edge_length=float('inf')):

    # condition the phases input
    if phases is None:
        default_dict = {'material_type': 'solid',
                        'max_volume': float('inf')}
        n_phases = int(np.max(polymesh.phase_numbers)) + 1
        phases = [default_dict for _ in range(n_phases)]

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
        sub_out = meshpy.triangle.subdivide_facets(n_subs, pts, facets,
                                                   facet_nums)
        pts, facets, facet_nums = sub_out

    # create groups/regions
    pts_arr = np.array(polymesh.points)
    regions = []
    holes = []

    ungrouped = np.full(len(polymesh.regions), True, dtype='?')
    while np.any(ungrouped):
        cell_ind = np.argmax(ungrouped)

        # compute cell center
        facet_list = polymesh.regions[cell_ind]
        cell_kps = {kp for n in facet_list for kp in polymesh.facets[n]}
        cell_cen = pts_arr[list(cell_kps)].mean(axis=0)

        # seed number and phase type
        seed_num = int(polymesh.seed_numbers[cell_ind])
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
    if n_dim == 2:
        tri_mesh = meshpy.triangle.build(info,
                                         attributes=True,
                                         volume_constraints=True,
                                         max_volume=max_volume,
                                         min_angle=min_angle,
                                         generate_faces=True)
    else:
        opts = meshpy.tet.Options('pq')
        opts.mindihedral = min_angle
        opts.maxvolume = float('inf')
        opts.fixedvolume = 1
        opts.regionattrib = 1
        opts.facesout = 1
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
    # ---------------------------------------------------------------------- 
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
                to_add = len(facet_seeds) < 2 or facet_seeds[0] != facet_seeds[1]
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
            raise ValueError('Points cannot have dimension ' + str(n_dim) + '.')

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
    remaining_inds = [i for i in range(1, len(pairs))]
    sorted_inds = [0]
    s_pairs = [pairs[0]]
    while remaining_inds:
        last_kp = s_pairs[-1][-1]
        for ind, i in enumerate(remaining_inds):
            pair = pairs[i]
            if last_kp in pair:
                break
        sorted_inds.append(i)
        del remaining_inds[ind]
        if last_kp == pair[0]:
            s_pairs.append(pair)
        else:
            s_pairs.append(list(reversed(pair)))
    return s_pairs


def _amorphous_seed_numbers(pmesh, phases):
    phase_nums = np.array(pmesh.phase_numbers)
    is_amorph = np.array([p.get('material_type', 'solid') in _misc.kw_amorph
                          for p in phases])
    amorph_mask = is_amorph[phase_nums]

    neighs = np.array(pmesh.facet_neighbors)
    neighs = neighs[np.min(neighs, axis=1) >= 0]
    neighs_mask = phase_nums[neighs[:, 0]] == phase_nums[neighs[:, 1]]
    neighs_mask &= amorph_mask[neighs[:, 0]]
    amorph_neighs = neighs[neighs_mask]

    new_seed_numbers = np.array(pmesh.seed_numbers)
    changes_made = True
    while changes_made:
        changes_made = False
        for pair in amorph_neighs:
            seeds = new_seed_numbers[pair]
            if seeds[0] != seeds[1]:
                changes_made = True
                new_seed_numbers[pair] = np.min(seeds)
    conv_dict = {s1: s2 for s1, s2 in zip(pmesh.seed_numbers, new_seed_numbers)
                 if s1 != s2}
    return conv_dict

def _pt3d(pt):
    pt3d = np.zeros(3)
    pt3d[:len(pt)] = pt
    return pt3d


def _facet_in_normal(pts, cen_pt):
    n_dim = len(cen_pt)
    if n_dim == 2:
        ptA = pts[0]
        ptB = pts[1]
        vt = ptB - ptA
        vn = np.array([-vt[1], vt[0]])
    else:
        ptA = pts[0]
        ptB = pts[1]
        ptC = pts[2]
        v1 = ptB - ptA
        v2 = ptC - ptA
        vn = np.cross(v1, v2)
     
    sgn = vn.dot(cen_pt - ptA)
    vn *= sgn  # flip so center is inward
    un = vn / np.linalg.norm(vn)
    return un, pts.mean(axis=0)


def _plot_2d(ax, mesh, index_by, **kwargs):
    simps = np.array(mesh.elements)
    pts = np.array(mesh.points)
    xy = pts[simps, :]

    plt_kwargs = {}
    for key, value in kwargs.items():
        if type(value) in (list, np.array):
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
