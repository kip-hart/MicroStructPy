.. _ex_std_voro:

========================
Standard Voronoi Diagram
========================

Python Script
=============

The basename for this file is ``standard_voronoi.py``.
The file can be run using this command::

    microstructpy --demo=standard_voronoi.py

The full text of the script is:

.. literalinclude:: ../../../../src/microstructpy/examples/standard_voronoi.py
    :language: python

Domain
======

The domain of the microstructure is a :class:`.Square`.
Without arguments, the square's center is (0, 0) and side length is 1.

Seeds
=====

A set of 50 seed circles with small radius is initially created.
Calling the :func:`~microstructpy.seeding.SeedList.position` method
positions the points according to random uniform distributions in the domain.

Polygon Mesh
============

A polygon mesh is created from the list of seed points using the
:func:`~microstructpy.meshing.PolyMesh.from_seeds` class method.
The mesh is plotted and saved into a PNG file in the remaining lines of the
script.

Plotting
========

The output Voronoi diagram is plotted in :numref:`f_ex_voro`.

.. _f_ex_voro:
.. figure:: ../../../../src/microstructpy/examples/standard_voronoi/voronoi_diagram.png

  Standard Voronoi diagram.
