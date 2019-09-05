.. _ex_basalt:

===============
Picritic Basalt
===============

XML Input File
==============

The basename for this file is ``basalt_circle.xml``.
The file can be run using this command::

    microstructpy --demo=basalt_circle.xml

The full text of the file is:

.. literalinclude:: ../../examples/basalt_circle.xml
    :language: xml


Material 1 - Plagioclase
------------------------

.. literalinclude:: ../../examples/basalt_circle.xml
    :language: xml
    :lines: 3-15
    :dedent: 4

Plagioclase composes approximately 45% of this picritic basalt sample.
It is an *aphanitic* component, meaning fine-grained, and follows a custom
size distribution.

Material 2 - Olivine
--------------------

.. literalinclude:: ../../examples/basalt_circle.xml
    :language: xml
    :lines: 17-40
    :dedent: 4

Olivine composes approximately 19% of this picritic basalt sample.
There are large *phenocrysts* of olivine in picritic basalt, so the crystals
are generally larger than the other components and have a non-circular shape.
The orientation of the phenocrysts is uniform random, with the aspect ratio
varying from 1 to 3 uniformly.

Materials 3-8
-------------

.. literalinclude:: ../../examples/basalt_circle.xml
    :language: xml
    :lines: 42-125
    :dedent: 4

Diopside, hypersthene, magnetite, chromite, ilmenite, and apatie compose
approximately 36% of this picritic basalt sample.
They are *aphanitic* components, meaning fine-grained, and follow a custom
size distribution.

Domain Geometry
---------------

.. literalinclude:: ../../examples/basalt_circle.xml
    :language: xml
    :lines: 127-130
    :dedent: 4

These materials fill a circular domain with a diameter of 30 mm.


Settings
--------

.. literalinclude:: ../../examples/basalt_circle.xml
    :language: xml
    :lines: 132-158
    :dedent: 4

The function will output plots of the microstructure process and those plots
are saved as PNGs.
They are saved in a folder named ``basalt_circle``, in the current directory
(i.e ``./basalt_circle``).

The axes are turned off in these plots, creating PNG files with
minimal whitespace.

Mesh controls are introducted to increase grid resolution, particularly at the
grain boundaries.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.
Verification plots are also generated, since the ``<verify>`` flag is on.

Seeding Plot
------------

.. image:: ../../examples/basalt_circle/seeds.png
   :alt: Seed particles.

Polygon Mesh Plot
-----------------

.. image:: ../../examples/basalt_circle/polymesh.png
   :alt: Polygon mesh.

Triangular Mesh Plot
--------------------

.. image:: ../../examples/basalt_circle/trimesh.png
   :alt: Triangular mesh.


Crystal Size Distribution Comparison Plot
-----------------------------------------

.. image:: ../../examples/basalt_circle/verification/size_cdf.png
   :alt: Comparing input and output CSDs.

