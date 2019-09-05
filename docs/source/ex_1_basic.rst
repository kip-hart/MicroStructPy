.. _ex_1_basic:

=======================
Intro 1 - Basic Example
=======================

XML Input File
==============

The basename for this file is ``intro_1_basic.xml``.
The file can be run using this command::

    microstructpy --demo=intro_1_basic.xml

The full text of the file is:

.. literalinclude:: ../../examples/intro_1_basic.xml
    :language: xml


Material 1 - Matrix
-------------------

.. literalinclude:: ../../examples/intro_1_basic.xml
    :language: xml
    :lines: 3-14
    :dedent: 4

There are two materials, in a 2:1 ratio based on volume.
The first is a matrix, which is represented with small circles.

Material 2 - Inclusions
-----------------------

.. literalinclude:: ../../examples/intro_1_basic.xml
    :language: xml
    :lines: 16-21
    :dedent: 4

The second material consists of circular inclusions with diameter 2.

Domain Geometry
---------------

.. literalinclude:: ../../examples/intro_1_basic.xml
    :language: xml
    :lines: 23-27
    :dedent: 4

These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions.


Settings
--------

.. literalinclude:: ../../examples/intro_1_basic.xml
    :language: xml
    :lines: 29-38
    :dedent: 4

PNG files of each step in the process will be output, as well as the
intermediate text files.
They are saved in a folder named ``intro_1_basic``, in the current directory
(i.e ``./intro_1_basic``).


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.

Seeding Plot
------------

.. image:: ../../examples/intro_1_basic/seeds.png
   :alt: Seed particles.

Polygon Mesh Plot
-----------------

.. image:: ../../examples/intro_1_basic/polymesh.png
   :alt: Polygon mesh.

Triangular Mesh Plot
--------------------

.. image:: ../../examples/intro_1_basic/trimesh.png
   :alt: Triangular mesh.
