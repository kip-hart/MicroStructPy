.. _ex_docs_banner:

===========================
Microstructure Mesh Process
===========================

Python Script
=============

The basename for this file is ``docs_banner.py``.
The file can be run using this command::

    microstructpy --demo=docs_banner.py

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/docs_banner.py
    :language: python

Domain Geometry
===============

The materials fill a rectangular domain with side lengths 20 and 10.
The center of the rectangle defaults to the origin.

Seeds 
=====

The first material is phase 2, which contains a single elliptical seed with
semi-axes 8 and 3.
Next, phases 0 and 1 are created with identical size distributions and
different colors.
The size distributions are uniform random from 0.5 to 1.5.
Seeds of phase 0 and phase 1 are generated to fill the area between the
rectangular domain and the elliptical seed from phase 2.

Next, the phase 2 seed is appended to the list of phase 0 and 1 seeds.
A hold list is then created to indicate to
:func:`~microstructpy.seeding.SeedList.position`
which seeds should have their positions (centers) held.
The default position of a seed is the origin, so by setting the hold flag to
``True`` for the elliptical seed, it will be fixed to the center of the domain
while the remaining seeds will be randomly positioned around it.

Polygonal and Triangular Meshing
================================

Once the seeds are positioned in the domain, a polygonal mesh is created using
:func:`~microstructpy.meshing.PolyMesh.from_seeds`.
The triangular mesh is created using
:func:`~microstructpy.meshing.TriMesh.from_polymesh`,
with the quality control settings ``min_angle``, ``max_edge_length``, and
``max_volume``.

Plot Figure
===========

The figure contains three plots: the seeds, the polygonal mesh, and the
triangular/unstructured mesh.
First, the seeds plot is generated using SeedList
:func:`~microstructpy.seeding.SeedList.plot`
and Rectangle
:func:`~microstructpy.geometry.Rectangle.plot`
to show the boundary of the domain.
The seeds are plotted with some transparency to show overlap.

Next, the polygonal mesh is translated to the right and plotted in such a way
that avoids the internal geometry of the elliptical seed.
This internal geometry is created by the multi-circle approximation used in
polygonal meshing, then removed during the triangular meshing process.
In the interest of clarity, these two steps are combined and the elliptical
grain is plotted without internal geomtry.

Finally, the triangular mesh is translated to the right of the polygonal mesh
and plotted using TriMesh
:func:`~microstructpy.meshing.TriMesh.plot`.

Once all three plots have been added to the figure, the axes and
aspect ratio are adjusted.
This figure is shown in :numref:`f_ex_process`.
The PNG and PDF versions of this plot are saved in a folder named
``docs_banner``, in the current directory (i.e ``./docs_banner``).

.. _f_ex_process:
.. figure:: ../../../../src/microstructpy/examples/docs_banner/banner.png
    :alt: Microstructure meshing process.

    Microstructure meshing process.

The three major steps are:
1) seed the domain with particles,
2) create a Voronoi power diagram, and
3) convert the diagram into an unstructured mesh.