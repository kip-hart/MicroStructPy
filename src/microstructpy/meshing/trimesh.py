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
                        facets.append([int(kp) for kp in line.split(',')])
                    elif stage == 'facet attributes':
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
    def from_polymesh(cls, polymesh, phases=None, min_angle=0,
                      max_volume=float('inf'), max_edge_length=float('inf')):
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
            min_angle (float): The minimum interior angle of an element.
            max_volume (float): The default maximum cell volume, used if one
                is not set for each phase.
            max_edge_length (float): The maximum edge length of elements
                along grain boundaries. Currently only supported in 2D.

        """
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
            cell_kps = set()
            [cell_kps.update(polymesh.facets[n]) for n in facet_list]
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
            opts.maxvolume = max_volume
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

        return cls(tri_pts, tri_elems, tri_e_atts, tri_f, tri_fa)

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
            cell_type =  {3: '5', 4: '10'}[n_kp]
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
            simps = np.array(self.elements)
            pts = np.array(self.points)
            xy = pts[simps, :]

            plt_kwargs = {}
            for key, value in kwargs.items():
                if type(value) in (list, np.array):
                    plt_value = []
                    for e_num, e_att in enumerate(self.element_attributes):
                        if index_by == 'element':
                            ind = e_num
                        elif index_by == 'attribute':
                            ind = int(e_att)
                        else:
                            e_str = 'Cannot index by {}.'.format(index_by)
                            raise ValueError(e_str)
                        v = value[ind]
                        plt_value.append(v)
                else:
                    plt_value = value
                plt_kwargs[key] = plt_value

            pc = collections.PolyCollection(xy, **plt_kwargs)
            ax.add_collection(pc)
            ax.autoscale_view()
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
