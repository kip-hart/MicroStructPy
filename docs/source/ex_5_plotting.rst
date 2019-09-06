.. _ex_5_plotting:

=============
Plot Controls
=============

XML Input File
==============

The basename for this file is ``intro_5_plotting.xml``.
The file can be run using this command::

    microstructpy --demo=intro_5_plotting.xml

The full text of the file is given below.

.. literalinclude:: ../../examples/intro_5_plotting.xml
    :language: xml
    :caption: Plot controls example input file


Material 1 - Matrix
-------------------

.. literalinclude:: ../../examples/intro_5_plotting.xml
    :language: xml
    :lines: 3-14
    :dedent: 4
    :caption: Matrix material section of input file

There are two materials, in a 2:1 ratio based on volume.
The first is a pink matrix, which is represented with small circles.

Material 2 - Inclusions
-----------------------

.. literalinclude:: ../../examples/intro_5_plotting.xml
    :language: xml
    :lines: 16-21
    :dedent: 4
    :caption: Inclusions material section of input file

The second material consists of lime green circular inclusions with diameter 2.

Domain Geometry
---------------

.. literalinclude:: ../../examples/intro_5_plotting.xml
    :language: xml
    :lines: 23-27
    :dedent: 4
    :caption: Domain geometry section of input file

These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions.


Settings
--------

.. literalinclude:: ../../examples/intro_5_plotting.xml
    :language: xml
    :lines: 29-57
    :dedent: 4
    :caption: Settings section of input file

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


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.

Seeding Plot
------------

.. image:: ../../examples/intro_5_plotting/seeds.png
   :alt: Seed particles.

Polygon Mesh Plot
-----------------

.. image:: ../../examples/intro_5_plotting/polymesh.png
   :alt: Polygon mesh.

Triangular Mesh Plot
--------------------

.. image:: ../../examples/intro_5_plotting/trimesh.png
   :alt: Triangular mesh.
