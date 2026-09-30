.. _ex_periodic_2d:

===================
Periodic 2D Example
===================

XML Input File
==============

The basename for this file is ``periodic_2D.xml``.
The file can be run using this command::

    microstructpy --demo=periodic_2D.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/periodic_2D.xml
    :language: xml


Materials
=========

The first material makes up two thirds of the area, with circular grains
whose diameters are uniformly distributed between 0.2 and 0.4.

The second material makes up the remaining third, with elliptical grains
of aspect ratio 2 and random orientations.

Domain Geometry
===============

The materials fill a square domain of side length 2, which is periodic in
both directions: the ``<periodic>`` field of the domain is ``True``.
The periodicity can also be restricted to some of the axes, for example
``<periodic> x </periodic>``.

Grains that cross a periodic face of the domain continue on the opposite
face, and the meshes have matching nodes on opposite faces.
The seeds are positioned without overlapping the grains on the other side,
the polygonal mesh is periodic across the faces, and the triangular mesh has
the same nodes on opposite faces, so that periodic boundary conditions can
be applied to it directly.
The pairs of periodic nodes are stored in the mesh (in its text file and,
for Abaqus, in node sets named ``Set-N-Periodic-X-Low`` and
``Set-N-Periodic-X-High``).

Settings
========

The random number generator seeds make the microstructure repeatable.

The mesh has a minimum angle of 25 degrees and a maximum element area of
0.004. The quality and size settings act on a periodic mesh as on a
non-periodic one.

A grain that barely crosses a periodic face, or ends just inside it, leaves
a thin piece of itself on the opposite face and very small elements there.
``periodic_margin`` rejects the positions where a seed ends within the
margin of a periodic face or crosses it by less than the margin; ``auto``
sets the margin to half the target edge length of the mesh, or to an eighth
of the smallest grain if that is smaller.
The edge optimization removes the shortest edges of the polygonal mesh, for
the same reason, and with the margin it also thickens or removes the pieces
of the grains at the periodic faces that are thinner than the margin, which
the placement of the seeds alone cannot prevent (a grain extends beyond its
seed, and its corners can cross a face by a small amount), and opens the
corners of the grains at the faces that are narrower than the minimum angle
of the mesh.

The plots are colored by seed number, so that the pieces of a grain on
opposite faces of the domain have the same color.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.
These three plots are shown in :numref:`f_ex_per2d_seeds` -
:numref:`f_ex_per2d_tri`.
The grains cut by the faces of the domain continue on the opposite faces.

.. _f_ex_per2d_seeds:
.. figure:: ../../../../src/microstructpy/examples/periodic_2D/seeds.png
    :alt: Seed geometries.

    Periodic 2D example - seed geometries.

.. _f_ex_per2d_poly:
.. figure:: ../../../../src/microstructpy/examples/periodic_2D/polymesh.png
    :alt: Polygonal mesh.

    Periodic 2D example - polygonal mesh.

.. _f_ex_per2d_tri:
.. figure:: ../../../../src/microstructpy/examples/periodic_2D/trimesh.png
    :alt: Triangular mesh.

    Periodic 2D example - triangular mesh.
