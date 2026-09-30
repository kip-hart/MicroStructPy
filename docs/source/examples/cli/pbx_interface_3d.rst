.. _ex_pbx_interface_3d:

===============================
Interface Refinement 3D Example
===============================

XML Input File
==============

The basename for this file is ``pbx_interface_3D.xml``.
The file can be run using this command::

    microstructpy --demo=pbx_interface_3D.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/pbx_interface_3D.xml
    :language: xml


Materials
=========

The materials are those of the :ref:`ex_pbx_3d`: crystalline
inclusions (65% of the volume, spherical seeds with lognormal diameters) in
a binder (a ``matrix`` phase, 35% of the volume, seeded with small spheres).

Domain Geometry
===============

The materials fill a cube of side length 3, periodic in the three
directions.

Settings
========

The mesh is refined on the grain boundaries, which are the interfaces
between the binder and the inclusions (and between neighboring inclusions),
and coarse inside the grains.

In 3D, ``mesh_max_edge_length`` sets the maximum edge length of the
triangles on the facets of the polyhedral mesh, 0.2 here: the facets are
triangulated to that size before TetGen meshes the cells.
``mesh_max_volume`` sets the maximum volume of the tetrahedra, 0.05 here,
which is the volume of a regular tetrahedron with edges about four times
longer, and TetGen grades the element size between the two.

The mesh has a minimum dihedral angle of 15 degrees.

The seeds are placed a margin away from the periodic faces (``periodic_margin``
set to ``auto``: half the target edge length, or an eighth of the smallest
grain if that is smaller) and the edge optimization moves the seeds that
leave a thin piece of a grain on a face, a corner narrower than the minimum
angle of the mesh, or a very short edge, since any of these forces very
small tetrahedra.


Output Files
============

The three plots that this file generates are the seeding, the polyhedral
mesh, and the tetrahedral mesh.
The seeds and the polyhedral mesh are those of the :ref:`ex_pbx_3d`
(:numref:`f_ex_pbx3d_seeds` and :numref:`f_ex_pbx3d_poly`), since the seeds
and the settings of the optimization are the same; only the tetrahedral
mesh differs, shown (its facets) in :numref:`f_ex_pbxint3d_tri`.

.. _f_ex_pbxint3d_tri:
.. figure:: ../../../../src/microstructpy/examples/pbx_interface_3D/trimesh.png
    :alt: Tetrahedral mesh.

    Interface refinement 3D example - tetrahedral mesh.
