.. _ex_2phase_3d:

====================
Two Phase 3D Example
====================

XML Input File
==============

The basename for this file is ``two_phase_3D.xml``.
The file can be run using this command::

    microstructpy --demo=two_phase_3D.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/two_phase_3D.xml
    :language: xml


Materials
=========

The first material makes up 25% of the volume, with a lognormal grain volume
distribution.

The second material makes up 75% of the volume, with an independent grain
volume distribution.

Domain Geometry
===============

These two materials fill a cube domain of side length 7.


Settings
========

The function will output plots of the microstructure process and those plots
are saved as PNGs.
They are saved in a folder named ``two_phase_3D``, in the current directory
(i.e ``./two_phase_3D``).

The line width of the output plots is reduced to 0.2, to make them more
visible.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.
These three plots are shown in :numref:`f_ex_2p3d_seeds` -
:numref:`f_ex_2p3d_tri`.

.. _f_ex_2p3d_seeds:
.. figure:: ../../../../src/microstructpy/examples/two_phase_3D/seeds.png
    :alt: Seed geometries.

    Two phase 3D example - seed geometries.
    
.. _f_ex_2p3d_poly:
.. figure:: ../../../../src/microstructpy/examples/two_phase_3D/polymesh.png
    :alt: Polygonal mesh.

    Two phase 3D example - polygonal mesh.
    
.. _f_ex_2p3d_tri:
.. figure:: ../../../../src/microstructpy/examples/two_phase_3D/trimesh.png
    :alt: Triangular mesh.

    Two phase 3D example - triangular mesh.
