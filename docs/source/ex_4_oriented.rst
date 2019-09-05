.. _ex_4_oriented:

=============================
Intro 4 - Oriented Inclusions
=============================

XML Input File
==============

The basename for this file is ``intro_4_oriented.xml``.
The file can be run using this command::

    microstructpy --demo=intro_4_oriented.xml

The full text of the file is:

.. literalinclude:: ../../examples/intro_4_oriented.xml
    :language: xml


Material 1 - Matrix
-------------------

.. literalinclude:: ../../examples/intro_4_oriented.xml
    :language: xml
    :lines: 3-14
    :dedent: 4

There are two materials, in a 2:1 ratio based on volume.
The first is a matrix, which is represented with small circles.

Material 2 - Inclusions
-----------------------

.. literalinclude:: ../../examples/intro_4_oriented.xml
    :language: xml
    :lines: 16-36
    :dedent: 4

The second material consists of elliptical inclusions with size ranging from
0 to 2 and aspect ratio ranging from 1 to 3.
Note that the size is defined as the diameter of a circle with equivalent area.
The orientation angle of the inclusions are uniformly distributed between -10
and +10 degrees, relative to the +x axis.

Domain Geometry
---------------

.. literalinclude:: ../../examples/intro_4_oriented.xml
    :language: xml
    :lines: 38-42
    :dedent: 4

These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions.


Settings
--------

.. literalinclude:: ../../examples/intro_4_oriented.xml
    :language: xml
    :lines: 44-53
    :dedent: 4

PNG files of each step in the process will be output, as well as the
intermediate text files.
They are saved in a folder named ``intro_4_oriented``, in the current directory
(i.e ``./intro_4_oriented``).


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.

Seeding Plot
------------

.. image:: ../../examples/intro_4_oriented/seeds.png
   :alt: Seed particles.

Polygon Mesh Plot
-----------------

.. image:: ../../examples/intro_4_oriented/polymesh.png
   :alt: Polygon mesh.

Triangular Mesh Plot
--------------------

.. image:: ../../examples/intro_4_oriented/trimesh.png
   :alt: Triangular mesh.
