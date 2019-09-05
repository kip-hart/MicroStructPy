.. _ex_2_quality:

==========================
Intro 2 - Quality Controls
==========================

XML Input File
==============

The basename for this file is ``intro_2_quality.xml``.
The file can be run using this command::

    microstructpy --demo=intro_2_quality.xml

The full text of the file is:

.. literalinclude:: ../../examples/intro_2_quality.xml
    :language: xml


Material 1 - Matrix
-------------------

.. literalinclude:: ../../examples/intro_2_quality.xml
    :language: xml
    :lines: 3-14
    :dedent: 4

There are two materials, in a 2:1 ratio based on volume.
The first is a matrix, which is represented with small circles.

Material 2 - Inclusions
-----------------------

.. literalinclude:: ../../examples/intro_2_quality.xml
    :language: xml
    :lines: 16-21
    :dedent: 4

The second material consists of circular inclusions with diameter 2.

Domain Geometry
---------------

.. literalinclude:: ../../examples/intro_2_quality.xml
    :language: xml
    :lines: 23-27
    :dedent: 4

These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions.


Settings
--------

.. literalinclude:: ../../examples/intro_2_quality.xml
    :language: xml
    :lines: 29-43
    :dedent: 4

PNG files of each step in the process will be output, as well as the
intermediate text files.
They are saved in a folder named ``intro_2_quality``, in the current directory
(i.e ``./intro_2_quality``).

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

Seeding Plot
------------

.. image:: ../../examples/intro_2_quality/seeds.png
   :alt: Seed particles.

Polygon Mesh Plot
-----------------

.. image:: ../../examples/intro_2_quality/polymesh.png
   :alt: Polygon mesh.

Triangular Mesh Plot
--------------------

.. image:: ../../examples/intro_2_quality/trimesh.png
   :alt: Triangular mesh.
