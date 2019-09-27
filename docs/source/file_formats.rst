.. _c_file_formats:

Output File Formats
===================

MicroStructPy creates output files for the seed geometries, polygonal meshes,
the unstructured/triangular meshes, and verification data.
Some of these outputs can be written in standard file formats, such as VTK.
Output files with a ``.txt`` extension are custom and explained in the
following sections.

List of Seeds
-------------

The :class:`.SeedList` class can write its contents to a file using the
:meth:`.SeedList.write` method and be read from a file using the
:meth:`.SeedList.from_file` method.
The CLI reads from and writes to ``seeds.txt`` in a run's directory.

This file contains a printed list of all the seeds in the list.
Specifically, the seeds are converted to strings and the strings are written
to the file.

The file that results looks like:

.. code-block:: text

    Geometry: circle
    Radius: 1
    Center: (2, -1)
    Phase: 0
    Breakdown: ((2, -1, 1))
    Position: (2, -1)
    Geometry: ellipse
    a: 3
    b: 1.5
    angle: -15
    center: (-5 3)
    phase: 1
    breakdown: ((-5, 3, 1.5), (-4, 2.5, 1.3), (-6, 3.5, 1.3))
    position: (-5, 3)
    ...
    ...
    ...
    Geometry: <class name from microstructpy.geometry>
    <param1>: <value1>
    <param2>: <value2>
        ...
    <paramN>: <valueN>
    phase: <phase number>
    breakdown: <circular/spherical breakdown of geometry>
    position: <position of seed>

For more information on how each seed listing is converted back into an
instance of the Seed class, see :meth:`.Seed.from_str`.

.. note::

    For geomtries such as the circle and ellipse, it seems redundant to
    specify both the center and the position of the seed.
    The rationale is that some geometries may be specified by some other
    point instead of the center.


.. _s_poly_file_io:

Polygonal Mesh
--------------

The polygonal mesh (or polyhedral mesh in 3D) can be written to and read
from a ``.txt`` file.
It can also be written to ``.poly`` files for 2D meshes, ``.vtk`` files for
3D meshes, and ``.ply`` files for any number of dimensions.

Text File
+++++++++

The text string output file is meant solely for saving the polygon/
polyhedron mesh as an intermediate step in the meshing process.
The format for the text string file is:

.. code-block:: text

    Mesh Points: <numPoints>
        x1, y1(, z1)      <- optional tab at line start
        x2, y2(, z2)
        ...
        xn, yn(, zn)
    Mesh Facets: <numFacets>
        f1_1, f1_2, f1_3, ...
        f2_1, f2_2, f2_3, ...
        ...
        fn_1, fn_2, fn_3, ...
    Mesh Regions: <numRegions>
        r1_1, r1_2, r1_3, ...
        r2_1, r2_2, r2_3, ...
        ...
        rn_1, rn_2, rn_3, ...
    Seed Numbers: <numRegions>
        s1
        s2
        ...
        sn
    Phase Numbers: <numRegions>
        p1
        p2
        ...
        pn

For example:

.. code-block:: text

    Mesh Points: 4
        0.0, 0.0
        1.0, 0.0
        3.0, 2.0
        2.0, 2.0
    Mesh Facets: 5
        0, 1
        1, 2
        2, 3
        3, 0
        1, 3
    Mesh Regions: 2
        0, 4, 3
        1, 2, 4
    Seed Numbers: 2
        0
        1
    Phase Numbers: 2
        0
        0

In this example, the polygon mesh contains a parallelogram
that has been divided into two triangles. In general, the regions do
not need to have the same number of facets.
For 3D meshes, the mesh facets should be an ordered list of point indices
that create the polygonal facet.

.. note::

    Everything is indexed from 0 since this file is produced in Python.
    

Additional Formats
++++++++++++++++++

These additional output file formats are meant for processing and
interpretation by other programs. 

The ``.poly`` POLY file contains a planar straight line graph (PSLG) and
can be read by the Triangle program from J. Shewchuk.
See `.poly files`_ from the Triangle documentation for more details.

The ``.vtk`` VTK legacy file format supports POLYDATA datasets.
The *facets* of a polyhedral mesh are written to the VTK file, but not the
region data, seed numbers, or phase numbers.
See `File Formats for VTK Version 4.2`_ for a guide to the VTK legacy format.

The ``.ply`` polygon file format is intended for 3D scans but can also store
the polygons and polyhedral facets of a polygonal mesh.
See `PLY - Polygon File Format`_ for a description and examples of ply files.


.. _`.poly files`: https://www.cs.cmu.edu/~quake/triangle.poly.html
.. _`File Formats for VTK Version 4.2`: https://www.vtk.org/wp-content/uploads/2015/04/file-formats.pdf
.. _`PLY - Polygon File Format`: http://paulbourke.net/dataformats/ply/


.. _s_tri_file_io:

Triangular Mesh
---------------

The triangular mesh (or tetrahedral mesh in 3D) can be written to and read
from a ``.txt`` file.
It can also be written to ``.inp`` Abaqus input files, ``.vtk`` files for
3D meshes, and ``.node``/``.ele`` files like Triangle and TetGen.

Text File
+++++++++

The organization of the triangular mesh text file is similar to the 
:class:`meshpy.triangle.MeshInfo` and :class:`meshpy.tet.MeshInfo`
classes from `MeshPy`_ .
The format for the text string file is:

.. code-block:: text

    Mesh Points: <numPoints>
        x1, y1(, z1)      <- optional tab at line start
        x2, y2(, z2)
        ...
        xn, yn(, zn)
    Mesh Elements: <numElements>
        e1_1, e1_2, e1_3(, e1_4)
        e2_1, e2_2, e2_3(, e2_4)
        ...
        en_1, en_2, en_3(, en_4)
    Element Attributes: <numElements>
        a1,
        a2,
        ...
        an
    Facets: <numFacets>
        f1_1, f1_2(, f1_3)
        f2_1, f2_2(, f2_3)
        ...
        fn_1, fn_2(, fn_3)
    Facet Attributes: <numFacets>
        a1,
        a2,
        ...
        an

In MicroStructPy, the element attribute is the seed number associated with the
element.
The facet attribute is the facet number from the polygonal mesh, so all of
the triangular mesh facets with the same attribute make up a polygonal mesh
facet.

.. note::

    Everything is indexed from 0 since this file is produced in Python.


Additional Formats
++++++++++++++++++

Triangular and tetrahedral meshes can be output to additional file formats for
processing and vizualization by other programs.
These include Abaqus input files, TetGen/Triangle standard outputs, and
the VTK legacy format.

The Abaqus input file option, ``format='abaqus'`` in :meth:`.TriMesh.write`,
creates an input file for the mesh that defines each grain as its own part.
It also creates surfaces between the grains and on the domain boundary for
applying boundary conditions and loads.

The TetGen/Triangle file option, ``format='tet/tri'``, creates ``.node``,
``.edge`` (or ``.face``), and ``.ele`` files.
See `Triangle`_ and TetGen's `File Formats`_ for more details on
these files and their format. 


.. _`File Formats`: http://wias-berlin.de/software/tetgen/1.5/doc/manual/manual006.html
.. _`MeshPy`: https://documen.tician.de/meshpy/
.. _`Triangle`: https://www.cs.cmu.edu/~quake/triangle.html