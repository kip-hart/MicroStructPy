.. _ex_3_shape:

============
Size & Shape
============

XML Input File
==============

The basename for this file is ``intro_3_size_shape.xml``.
The file can be run using this command::

    microstructpy --demo=intro_3_size_shape.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/intro_3_size_shape.xml
    :language: xml


Materials
=========

There are two materials, in a 2:1 ratio based on volume.
The first is a matrix, which is represented with small circles.

The second material consists of elliptical inclusions with size ranging from
0 to 2 and aspect ratio ranging from 1 to 3.
Note that the size is defined as the diameter of a circle with equivalent area.
The orientation angle of the inclusions are random, specifically they are
uniformly distributed from 0 to 360 degrees.

Domain Geometry
===============

These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions
to ensure that the microstructure is statistically representative.


Settings
========

Many settings have been left to their defaults, with the exceptions being the
verbose mode and output directory.

By default, MicroStructPy does not print status updates to the command line.
Switching the verbose mode on will regularly print the status of the code.

The output directory is a filepath for writing text and image files.
By default, MicroStructPy outputs texts files containing data on the seeds,
polygon mesh, and triangular mesh as well as the corresponding image files,
saved in PNG format.

.. note::

    The ``<directory>`` field can be an absolute or relative filepath. If it is
    relative, outputs are written relative to the **input file**, not the
    current working directory.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.
These three plots are shown in :numref:`f_ex_3_size_shape_seeds` -
:numref:`f_ex_3_size_shape_tri`.

.. _f_ex_3_size_shape_seeds:
.. figure:: ../../../../src/microstructpy/examples/intro_3_size_shape/seeds.png
    :alt: Seed geometries.

    Introduction 3 - seed geometries.
    
.. _f_ex_3_size_shape_poly:
.. figure:: ../../../../src/microstructpy/examples/intro_3_size_shape/polymesh.png
    :alt: Polygonal mesh.

    Introduction 3 - polygonal mesh.
    
.. _f_ex_3_size_shape_tri:
.. figure:: ../../../../src/microstructpy/examples/intro_3_size_shape/trimesh.png
    :alt: Triangular mesh.

    Introduction 3 - triangular mesh.