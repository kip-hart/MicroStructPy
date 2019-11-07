.. _ex_6_culmin:

===========
Culmination
===========

XML Input File
==============

The basename for this file is ``intro_6_culmination.xml``.
The file can be run using this command::

    microstructpy --demo=intro_6_culmination.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/intro_6_culmination.xml
    :language: xml


Materials
=========

There are two materials, in a 2:1 ratio based on volume.
The first is a pink matrix, which is represented with small circles.

The second material consists of lime green elliptical inclusions with size
ranging from 0 to 2 and aspect ratio ranging from 1 to 3.
Note that the size is defined as the diameter of a circle with equivalent area.
The orientation angle of the inclusions are uniformly distributed between -10
and +10 degrees, relative to the +x axis.

Domain Geometry
===============

These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions.


Settings
========

PNG files of each step in the process will be output, as well as the
intermediate text files.
They are saved in a folder named ``intro_5_plotting``, in the current directory
(i.e ``./intro_5_plotting``).
PDF files of the poly and tri mesh are also generated, plus an EPS file for the
tri mesh.

The seeds are plotted with transparency to show the overlap between them.
The poly mesh is plotted with thick purple edges and the tri mesh is plotted
with thin navy edges.

In all of the plots, the axes are toggles off, creating image files with
minimal borders.

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
These three plots are shown in :numref:`f_ex_6_culmination_seeds` -
:numref:`f_ex_6_culmination_tri`.

.. _f_ex_6_culmination_seeds:
.. figure:: ../../../../src/microstructpy/examples/intro_6_culmination/seeds.png
    :alt: Seed geometries.

    Introduction 6 - seed geometries.
    
.. _f_ex_6_culmination_poly:
.. figure:: ../../../../src/microstructpy/examples/intro_6_culmination/polymesh.png
    :alt: Polygonal mesh.

    Introduction 6 - polygonal mesh.
    
.. _f_ex_6_culmination_tri:
.. figure:: ../../../../src/microstructpy/examples/intro_6_culmination/trimesh.png
    :alt: Triangular mesh.

    Introduction 6 - triangular mesh.