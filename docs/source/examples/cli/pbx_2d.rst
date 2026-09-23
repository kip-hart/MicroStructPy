.. _ex_pbx_2d:

=============================
Binder and Inclusions Example
=============================

XML Input File
==============

The basename for this file is ``pbx_2D.xml``.
The file can be run using this command::

    microstructpy --demo=pbx_2D.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/pbx_2D.xml
    :language: xml


Materials
=========

This microstructure is a particulate composite, such as a plastic-bonded
explosive: crystalline inclusions in a polymer binder.

The first material is the binder, a ``matrix`` phase that makes up 35% of the
area.
Its seeds are small circles: the cells of a matrix phase are merged into one
region, so the seeds only need to fill the space between the inclusions.

The second material is the inclusions, a ``crystalline`` phase that makes up
65% of the area, with circular seeds whose diameters follow a lognormal
distribution.
Each seed of a crystalline phase becomes one polygonal grain of the mesh.

Domain Geometry
===============

The materials fill a square domain of side length 3, periodic in both
directions.

Settings
========

The overlap tolerance ``rtol`` is set to 0.5: with the default fitted value
some of the small binder seeds do not fit between the inclusions and are
rejected.
The random number generator seeds make the microstructure repeatable, and
the edge optimization removes the shortest edges of the polygonal mesh.

The mesh has a minimum angle of 25 degrees and a maximum element area of
0.01.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.
These three plots are shown in :numref:`f_ex_pbx2d_seeds` -
:numref:`f_ex_pbx2d_tri`.
The binder is one region of the mesh; its boundaries with the inclusions are
facets of the mesh.

.. _f_ex_pbx2d_seeds:
.. figure:: ../../../../src/microstructpy/examples/pbx_2D/seeds.png
    :alt: Seed geometries.

    Binder and inclusions example - seed geometries.

.. _f_ex_pbx2d_poly:
.. figure:: ../../../../src/microstructpy/examples/pbx_2D/polymesh.png
    :alt: Polygonal mesh.

    Binder and inclusions example - polygonal mesh.

.. _f_ex_pbx2d_tri:
.. figure:: ../../../../src/microstructpy/examples/pbx_2D/trimesh.png
    :alt: Triangular mesh.

    Binder and inclusions example - triangular mesh.
