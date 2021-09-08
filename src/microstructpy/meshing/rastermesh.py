"""Raster Meshing

This module contains the class definition for the RasterMesh class.

"""
# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #


from __future__ import division
from __future__ import print_function

import numpy as np

from microstructpy.meshing import TriMesh

__all__ = ['RasterMesh']
__author__ = 'Kenneth (Kip) Hart'


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
        # TODO convert from pseudo-code to code
        # Pseudo-Code
        # 1. Create node and element grids
        # 2. Compute element centers
        # 3. Remove elements outside domain
        # 4. For each region:
        #   A. Create a bounding box
        #   B. Isolate element centers with box
        #   C. For each facet, remove centers on the wrong side
        #   D. Assign remaining centers to region
        # 5. Combine regions of the same seed number (remove voids)
        # 6. Define remaining facets, inherit their attributes

        raise NotImplementedError

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
            raise NotImplementedError
        elif fmt in ('str', 'txt'):
            with open(filename, 'w') as file:
                file.write(str(self) + '\n')
        elif fmt == 'vtk':
            raise NotImplementedError

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
        return np.norm(s0)

    def as_array(self, element_attributes=True):
        """numpy.ndarray containing element attributes.

        Array contains -1 where there are no elements (e.g. circular domains).

        Args:
            element_attributes (bool): *(optional)* Flag to return element
            attributes in the array. Set to True return attributes and set to
            False to return element indices. Defaults to True.
        """
        # TODO convert pseudo-code to code
        # 1. Create array full of -1 values
        # 2. Convert 1st node of each element into array indices
        # 3. For each element: populate array with element attributes

        raise NotImplementedError


    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    # Inherited from TriMesh
