.. _ex_pbx_3d:

================================
Binder and Inclusions 3D Example
================================

XML Input File
==============

The basename for this file is ``pbx_3D.xml``.
The file can be run using this command::

    microstructpy --demo=pbx_3D.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/pbx_3D.xml
    :language: xml


Materials
=========

This is the 3D version of the :ref:`ex_pbx_2d` example: crystalline
inclusions (65% of the volume, spherical seeds with lognormal diameters) in
a binder (a ``matrix`` phase, 35% of the volume, seeded with small spheres).

Domain Geometry
===============

The materials fill a cube of side length 3, periodic in the three
directions.

Settings
========

The overlap tolerance ``rtol`` is set to 0.5 so that the small binder seeds
can be placed between the inclusions, and the random number generator seeds
make the microstructure repeatable.

The mesh has a minimum dihedral angle of 15 degrees and a maximum element
volume of 0.02.


Output Files
============

The three plots that this file generates are the seeding, the polyhedral
mesh, and the tetrahedral mesh.
These three plots are shown in :numref:`f_ex_pbx3d_seeds` -
:numref:`f_ex_pbx3d_tri`.

.. _f_ex_pbx3d_seeds:
.. figure:: ../../../../src/microstructpy/examples/pbx_3D/seeds.png
    :alt: Seed geometries.

    Binder and inclusions 3D example - seed geometries.

.. _f_ex_pbx3d_poly:
.. figure:: ../../../../src/microstructpy/examples/pbx_3D/polymesh.png
    :alt: Polyhedral mesh.

    Binder and inclusions 3D example - polyhedral mesh.

.. _f_ex_pbx3d_tri:
.. figure:: ../../../../src/microstructpy/examples/pbx_3D/trimesh.png
    :alt: Tetrahedral mesh.

    Binder and inclusions 3D example - tetrahedral mesh.
