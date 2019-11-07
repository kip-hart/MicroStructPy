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

.. literalinclude:: ../../../../src/microstructpy/examples/elliptical_grains.xml
    :language: xml


Materials
=========

There are five materials, represented in equal proportions.
The first material consists of ellipses and the semi-major axes are
uniformly distributed, :math:`A \sim U(0.20, 0.75)`.
The semi-minor axes are fixed at 0.05, meaning the aspect ratio of these
seeds are 4-15.
The orientation angles of the ellipses are uniform random in distribution from
0 to 20 degrees counterclockwise from the +x axis.

The remaining four materials are all the same, with lognormal grain area
distributions.
The only difference among these materials is the color, which was done for
visual effect.


Domain Geometry
===============

The domain of the microstructure is a rectangle with side lengths 2.4 in the
x-direction and 1.2 in the y-direction.
The domain is centered on the origin, though the position of the domain is
not relevant considering that the plot axes are switched off.


Settings
========

The aspect ratio of elements in the triangular mesh is controlled
by setting the minimum interior angle for the elements at 20 degrees,
the maximum element volume to 0.001, and the maximum edge length at grain
boundaries to 0.01.

The function will output only plots of the microstructure process
(no text files), and those plots are saved as PNGs.
They are saved in a folder named ``elliptical_grains``, in the current directory
(i.e ``./elliptical_grains``).

The axes are turned off in these plots, creating PNG files with
minimal whitespace.

Finally, the linewiths in the seeds plot, polygonal mesh plot, and the
triangular mesh plot are 0.5, 0.5, 0.1 respectively.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.
These three plots are shown in :numref:`f_ex_ell_seeds` - 
:numref:`f_ex_ell_tri`.

.. _f_ex_ell_seeds:
.. figure:: ../../../../src/microstructpy/examples/elliptical_grains/seeds.png
    :alt: Seed geometries.

    Elliptical grain example - seed geometries.

.. _f_ex_ell_poly:    
.. figure:: ../../../../src/microstructpy/examples/elliptical_grains/polymesh.png
    :alt: Polygonal mesh.

    Elliptical grain example - polygonal mesh.
    
.. _f_ex_ell_tri:
.. figure:: ../../../../src/microstructpy/examples/elliptical_grains/trimesh.png
    :alt: Triangular mesh.

    Elliptical grain example - triangular mesh.

