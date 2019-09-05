.. _ex_minimal:

===============
Minimal Example
===============

XML Input File
==============

The basename for this file is ``minimal_paired.xml``.
The file can be run using this command::

    microstructpy --demo=minimal_paired.xml

The full text of the file is:

.. literalinclude:: ../../examples/minimal_paired.xml
    :language: xml


Material 1
----------

.. literalinclude:: ../../examples/minimal_paired.xml
    :language: xml
    :lines: 3-6
    :dedent: 4

There is only one material, with a constant size of 0.09.

Domain Geometry
---------------

.. literalinclude:: ../../examples/minimal_paired.xml
    :language: xml
    :lines: 8-10
    :dedent: 4

The material fills a square domain.
The default side length is 1, meaning the domain is greater than 10x larger
than the grains.


Settings
--------

.. literalinclude:: ../../examples/minimal_paired.xml
    :language: xml
    :lines: 12-23
    :dedent: 4

The function will output plots of the microstructure process and those plots
are saved as PNGs.
They are saved in a folder named ``minimal``, in the current directory
(i.e ``./minimal``).

The axes are turned off in these plots, creating PNG files with
minimal whitespace.

Finally, the seeds and grains are colored by their seed number, not by
material.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.

Seeding Plot
------------

.. image:: ../../examples/minimal/seeds.png
   :alt: Seed particles.

Polygon Mesh Plot
-----------------

.. image:: ../../examples/minimal/polymesh.png
   :alt: Polygon mesh.

Triangular Mesh Plot
--------------------

.. image:: ../../examples/minimal/trimesh.png
   :alt: Triangular mesh.
