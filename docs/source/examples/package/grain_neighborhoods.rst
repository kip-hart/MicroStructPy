.. _ex_grain_nbr:

===================
Grain Neighborhoods
===================

Python Script
=============

The basename for this file is ``grain_neighborhoods.py``.
The file can be run using this command::

    microstructpy --demo=grain_neighborhoods.py

The full text of the script is:

.. literalinclude:: ../../../../src/microstructpy/examples/grain_neighborhoods.py
    :language: python

Domain
======

The domain of the microstructure is a
:class:`microstructpy.geometry.Rectangle`, with the bottom left corner at the
origin and side lengths of 8 and 12.

Phases
======

There are initially two phases: a matrix phase and a neighborhood phase.
The neighborhood phase will be broken down into materials later. The matrix
phase occupies two thirds of the domain, while the neighborhoods occupy one
third.

Seeds
=====

The seeds are generated to fill 1.1x the area of the domain, to account for
overlap with the boundaries. They are positioned according to random uniform
distributions.

Neighborhood Replacement
========================

The neighborhood seeds are replaced by a set of three different materials.
One material occupies the center of the neighborhood, while the other two
alternate in a ring around the center.

Polygon and Triangle Meshing
============================

The seeds are converted into a triangular mesh by first constructing a
polygon mesh. Each material is solid, except for the first which is designated
as a matrix phase. Mesh quality controls are specified to prevent high aspect
ratio triangles.

Plotting
========

The triangular mesh is plotted and saved to a file.
Each triangle is colored based on its material phase, using the standard
matplotlib colors: C0, C1, C2, etc.
The output PNG file is shown in :numref:`f_ex_neighs_tri`.

.. _f_ex_neighs_tri:
.. figure:: ../../../../src/microstructpy/examples/grain_neighborhoods/trimesh.png
  :alt: Triangular mesh of microstructure with seed neighborhoods.

  Triangular mesh of microstructure with seed neighborhoods.