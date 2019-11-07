.. _ex_uni_seed:

===============================
Uniform Seeding Voronoi Diagram
===============================

Python Script
=============

The basename for this file is ``uniform_seeding.py``.
The file can be run using this command::

    microstructpy --demo=uniform_seeding.py

The full text of the script is:

.. literalinclude:: ../../../../src/microstructpy/examples/uniform_seeding.py
    :language: python

Domain
======

The domain of the microstructure is a :class:`.Square`.
Without arguments, the square's center is (0, 0) and side length is 1.

Seeds
=====

A set of 200 seed circles with small radius is initially created.
The positions of the seeds are set with Mitchell's Best Candidate Algorithm
[#f1]_. This algorithm positions seed *i* by sampling *i + 1*
random points and picking the one that is furthest from its nearest neighbor.

Polygon Mesh
============

A polygon mesh is created from the list of seed points using the
:func:`~microstructpy.meshing.PolyMesh.from_seeds` class method.

Plotting
========

The facecolor of each polygon is determined by its area. If it is below the
standard area (domain area / number of cells), then it is shaded blue. If
it is above the standard area, it is shaded red. A custom colorbar is added
to the figure and it is saved as a PNG, shown in :numref:`f_ex_uni_voro`.

.. _f_ex_uni_voro:
.. figure:: ../../../../src/microstructpy/examples/uniform_seeding/voronoi_diagram.png
  :alt: Voronoi diagram with uniformly-spaced seeds, colored by area.
  
  Uniformly seeded Voronoi diagram with cells colored by area.


.. [#f1] Mitchell, T.J., "An Algorithm for the Construction of "D-Optimal"
  Experimental Designs," Technometrics, Vol. 16, No. 2, May 1974, pp. 203-210.
  (https://www.jstor.org/stable/1267940)
