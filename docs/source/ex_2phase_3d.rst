.. _ex_2phase_3d:

====================
Two Phase 3D Example
====================

XML Input File
==============

The basename for this file is ``two_phase_3D.xml``.
The file can be run using this command::

    microstructpy --demo=two_phase_3D.xml

The full text of the file is:

.. literalinclude:: ../../examples/two_phase_3D.xml
    :language: xml


Material 1
----------

.. literalinclude:: ../../examples/two_phase_3D.xml
    :language: xml
    :lines: 3-12
    :dedent: 4

The first material makes up 25% of the volume, with a lognormal grain volume
distribution.

Material 2
----------

.. literalinclude:: ../../examples/two_phase_3D.xml
    :language: xml
    :lines: 14-22
    :dedent: 4

The second material makes up 75% of the volume, with an independent grain
volume distribution.

Domain Geometry
---------------

.. literalinclude:: ../../examples/two_phase_3D.xml
    :language: xml
    :lines: 24-28
    :dedent: 4

These two materials fill a square domain of side length 7.


Settings
--------

.. literalinclude:: ../../examples/two_phase_3D.xml
    :language: xml
    :lines: 30-49
    :dedent: 4

The function will output plots of the microstructure process and those plots
are saved as PNGs.
They are saved in a folder named ``two_phase_3D``, in the current directory
(i.e ``./two_phase_3D``).

The line width of the output plots is reduced to 0.2, to make them more
visible.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.

Seeding Plot
------------

.. image:: ../../examples/two_phase_3D/seeds.png
   :alt: Seed particles.

Polygon Mesh Plot
-----------------

.. image:: ../../examples/two_phase_3D/polymesh.png
   :alt: Polygon mesh.

Triangular Mesh Plot
--------------------

.. image:: ../../examples/two_phase_3D/trimesh.png
   :alt: Triangular mesh.
