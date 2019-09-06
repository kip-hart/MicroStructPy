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

.. literalinclude:: ../../examples/intro_3_size_shape.xml
    :language: xml


Material 1 - Matrix
-------------------

.. literalinclude:: ../../examples/intro_3_size_shape.xml
    :language: xml
    :lines: 3-14
    :dedent: 4

There are two materials, in a 2:1 ratio based on volume.
The first is a matrix, which is represented with small circles.

Material 2 - Inclusions
-----------------------

.. literalinclude:: ../../examples/intro_3_size_shape.xml
    :language: xml
    :lines: 16-32
    :dedent: 4

The second material consists of elliptical inclusions with size ranging from
0 to 2 and aspect ratio ranging from 1 to 3.
Note that the size is defined as the diameter of a circle with equivalent area.
The orientation angle of the inclusions are random, specifically they are
uniformly distributed from 0 to 360 degrees.

Domain Geometry
---------------

.. literalinclude:: ../../examples/intro_3_size_shape.xml
    :language: xml
    :lines: 23-27
    :dedent: 4

These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions.


Settings
--------

.. literalinclude:: ../../examples/intro_3_size_shape.xml
    :language: xml
    :lines: 29-43
    :dedent: 4

PNG files of each step in the process will be output, as well as the
intermediate text files.
They are saved in a folder named ``intro_3_size_shape``, in the current directory
(i.e ``./intro_3_size_shape``).


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.

Seeding Plot
------------

.. image:: ../../examples/intro_3_size_shape/seeds.png
   :alt: Seed particles.

Polygon Mesh Plot
-----------------

.. image:: ../../examples/intro_3_size_shape/polymesh.png
   :alt: Polygon mesh.

Triangular Mesh Plot
--------------------

.. image:: ../../examples/intro_3_size_shape/trimesh.png
   :alt: Triangular mesh.
