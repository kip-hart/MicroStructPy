.. _ex_elliptical_grains:

=================
Elliptical Grains
=================

XML Input File
==============

The basename for this file is ``elliptical_grains.xml``.
The file can be run using this command::

    microstructpy --demo=elliptical_grains.xml

The full text of the file is:

.. literalinclude:: ../../examples/elliptical_grains.xml
    :language: xml


Material 1 - Ellipses
---------------------------

.. literalinclude:: ../../examples/elliptical_grains.xml
    :language: xml
    :lines: 3-14
    :dedent: 4

There are two materials, in a 1:2 ratio based on volume.
The first material consists of ellipses and the semi-major axes are
uniformly distributed, :math:`A \sim U(0.05, 0.40)`.
The semi-minor axes are fixed at 0.05, meaning the aspect ratio of these
seeds are 1-8. The orientation angles of the ellipses are uniform random
in distribution.

Material 2 - Circles
-------------------------

.. literalinclude:: ../../examples/elliptical_grains.xml
    :language: xml
    :lines: 16-25
    :dedent: 4

The second material consists of circles, which have a diameter
that is log-normally distributed, :math:`D \sim 0.06 e^{N(0, 0.5)}`.

Domain Geometry
---------------

.. literalinclude:: ../../examples/elliptical_grains.xml
    :language: xml
    :lines: 27-30
    :dedent: 4

These two materials fill a rectangular domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The width of the rectangle is 2 and the height is 3.


Settings
--------

.. literalinclude:: ../../examples/elliptical_grains.xml
    :language: xml
    :lines: 32-58
    :dedent: 4

The aspect ratio of elements in the triangular mesh is controlled
by setting the minimum interior angle for the elements at 20 degrees.

The function will output only plots of the microstructure process
(no text files), and those plots are saved as PNGs.
They are saved in a folder named ``elliptical_grains``, in the current directory
(i.e ``./elliptical_grains``).

The axes are turned off in these plots, creating PNG files with
minimal whitespace.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.

Seeding Plot
------------

.. image:: ../../examples/elliptical_grains/seeds.png
   :alt: Seed particles.

Polygon Mesh Plot
-----------------

.. image:: ../../examples/elliptical_grains/polymesh.png
   :alt: Polygon mesh.

Triangular Mesh Plot
--------------------

.. image:: ../../examples/elliptical_grains/trimesh.png
   :alt: Triangular mesh.
