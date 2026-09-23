.. _ex_periodic_3d:

===================
Periodic 3D Example
===================

XML Input File
==============

The basename for this file is ``periodic_3D.xml``.
The file can be run using this command::

    microstructpy --demo=periodic_3D.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/periodic_3D.xml
    :language: xml


Materials
=========

The two materials make up half of the volume each.
The first has spherical grains whose diameters are uniformly distributed
between 0.8 and 1.4, the second has spherical grains of diameter 1.2.

Domain Geometry
===============

The materials fill a cube of side length 4, which is periodic in the three
directions: the ``<periodic>`` field of the domain is ``xyz``.
Any subset of the axes can be given, for example ``xz`` to leave the y
direction free.

The polyhedral mesh is periodic across the faces (a grain cut by a face
continues on the opposite face) and the tetrahedral mesh has the same nodes
on opposite faces, stored as pairs in the mesh.

Settings
========

The random number generator seeds make the microstructure repeatable.

The mesh has a minimum dihedral angle of 15 degrees and a maximum element
volume of 0.02.
In 3D, the domain is meshed once as usual, then the facets are
triangulated with the points TetGen added on them, identically on opposite
faces, and the mesh is built again with these facets; the quality and size
settings act as on a non-periodic mesh.

The seeds are placed a margin away from the periodic faces (``periodic_margin``
set to ``auto``: half the target edge length, or an eighth of the smallest
grain if that is smaller) and the edge optimization moves the seeds that
leave a thin piece of a grain on a face, a corner narrower than the minimum
angle of the mesh, or a very short edge, since any of these forces very
small tetrahedra.

The plots are colored by seed number and the line widths are reduced to
make the grains visible.


Output Files
============

The three plots that this file generates are the seeding, the polyhedral
mesh, and the tetrahedral mesh.
These three plots are shown in :numref:`f_ex_per3d_seeds` -
:numref:`f_ex_per3d_tri`.

.. _f_ex_per3d_seeds:
.. figure:: ../../../../src/microstructpy/examples/periodic_3D/seeds.png
    :alt: Seed geometries.

    Periodic 3D example - seed geometries.

.. _f_ex_per3d_poly:
.. figure:: ../../../../src/microstructpy/examples/periodic_3D/polymesh.png
    :alt: Polyhedral mesh.

    Periodic 3D example - polyhedral mesh.

.. _f_ex_per3d_tri:
.. figure:: ../../../../src/microstructpy/examples/periodic_3D/trimesh.png
    :alt: Tetrahedral mesh.

    Periodic 3D example - tetrahedral mesh.
