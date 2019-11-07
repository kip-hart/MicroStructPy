.. _ex_2_quality:

================
Quality Controls
================

XML Input File
==============

The basename for this file is ``intro_2_quality.xml``.
The file can be run using this command::

    microstructpy --demo=intro_2_quality.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/intro_2_quality.xml
    :language: xml


Materials
=========

There are two materials, in a 2:1 ratio based on volume.
The first is a matrix, which is represented with small circles.
The second material consists of circular inclusions with diameter 2.

Domain Geometry
===============

These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions to ensure that
microstructure is statistically representative.


Settings
========

The first two settings determine the output directory and whether to run the
program in verbose mode.
The following settings determine the quality of the triangular mesh.

The minimum interior angle of the elements is 25 degrees, ensuring lower
aspect ratios compared to the first example.
The maximum area of the elements is also limited to 1, which populates the
matrix with smaller elements.
Finally, The maximum edge length of elements at interfaces is set to 0.1,
which increasing the mesh density surrounding the inclusions and at the
boundary of the domain.

Note that the edge length control is currently unavailable in 3D.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.
These three plots are shown in :numref:`f_ex_2_quality_seeds` -
:numref:`f_ex_2_quality_tri`.

.. _f_ex_2_quality_seeds:
.. figure:: ../../../../src/microstructpy/examples/intro_2_quality/seeds.png
    :alt: Seed geometries.

    Introduction 2 - seed geometries.
    
.. _f_ex_2_quality_poly:
.. figure:: ../../../../src/microstructpy/examples/intro_2_quality/polymesh.png
    :alt: Polygonal mesh.

    Introduction 2 - polygonal mesh.
    
.. _f_ex_2_quality_tri:
.. figure:: ../../../../src/microstructpy/examples/intro_2_quality/trimesh.png
    :alt: Triangular mesh.

    Introduction 2 - triangular mesh.