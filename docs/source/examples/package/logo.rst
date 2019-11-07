.. _ex_logo:

==================
MicroStructPy Logo
==================

Python Script
=============

The basename for this file is ``logo.py``.
The file can be run using this command::

    microstructpy --demo=logo.py

The full text of the script is:

.. literalinclude:: ../../../../src/microstructpy/examples/logo.py
    :language: python

Domain
======

The domain of the microstructure is a :class:`.Circle`.
Without arguments, the circle's center is (0, 0) and side length is 1.

Seeds
=====

The seeds are 14 circles with radii uniformly distributed from 0 to 0.3.
Calling the :func:`~microstructpy.seeding.SeedList.position` method
positions the points according to random uniform distributions in the domain.

Polygon Mesh
============

A polygon mesh is created from the list of seed points using the
:func:`~microstructpy.meshing.PolyMesh.from_seeds` class method.
The mesh is plotted and saved into a PNG file in the remaining lines of the
script.

Plot Logo
=========

The edges in the polygonal mesh are plotted white on a black background.
This image is converted into a mask and the white pixels are converted into
transparent pixels.
The remaining pixels are colored with a linear gradient between two colors.
This image is saved in padded and tight versions, as well as a favicon for the
HTML documentation.
The logo that results is shown in :numref:`f_ex_logo`.

.. _f_ex_logo:
.. figure:: ../../../../src/microstructpy/examples/logo/logo.png
    :alt: MicroStructPy logo.

    MicroStructPy logo.

