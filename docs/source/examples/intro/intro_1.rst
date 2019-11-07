.. _ex_1_basic:

=============
Basic Example
=============

XML Input File
==============

The basename for this file is ``intro_1_basic.xml``.
The file can be run using this command::

    microstructpy --demo=intro_1_basic.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/intro_1_basic.xml
    :language: xml


Materials
=========

There are two materials, in a 2:1 ratio based on volume.
The first is a matrix, which is represented with small circles.
The size and shape of matrix grain particles are not critical, since
the boundaries between them will be removed before triangular meshing.
The second material consists of circular inclusions with diameter 2.

Domain Geometry
===============

The domain of the microstructure is a square with its bottom-left
corner fixed to the origin.
The side length is 20, which is 10x the size of the inclusions to ensure that
the microstructure is statistically representative.


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
These three plots are shown in :numref:`f_ex_1_basic_seeds` -
:numref:`f_ex_1_basic_tri`.

.. _f_ex_1_basic_seeds:
.. figure:: ../../../../src/microstructpy/examples/intro_1_basic/seeds.png
    :alt: Seed geometries.

    Introduction 1 - seed geometries.
    
.. _f_ex_1_basic_poly:
.. figure:: ../../../../src/microstructpy/examples/intro_1_basic/polymesh.png
    :alt: Polygonal mesh.

    Introduction 1 - polygonal mesh.
    
.. _f_ex_1_basic_tri:
.. figure:: ../../../../src/microstructpy/examples/intro_1_basic/trimesh.png
    :alt: Triangular mesh.

    Introduction 1 - triangular mesh.