.. _ex_periodic_tiling:

=======================================
Periodic Microstructures and Their Tiles
=======================================

Python Script
=============

The basename for this file is ``periodic_tiling.py``.
The file can be run using this command::

    microstructpy --demo=periodic_tiling.py

The full text of the script is:

.. literalinclude:: ../../../../src/microstructpy/examples/periodic_tiling.py
    :language: python

Periodic Microstructure in 2D
=============================

The domain is a :class:`.Square` of side length 2 and the two phases are
circular grains and elliptical inclusions.
The seeds are created with :func:`~microstructpy.seeding.SeedList.from_info`
and positioned with :func:`~microstructpy.seeding.SeedList.position`, with
``periodic=True``: a seed that crosses a face of the domain is also checked
for overlaps on the opposite face.
The ``periodic_margin`` keeps the seeds from ending within half a target
edge length of a face, or crossing one by less, since such seeds leave thin
pieces of grains on the opposite face and very small triangles there; it is
capped at an eighth of the smallest grain, which needs about four elements
across it.
The polygonal mesh is created with
:func:`~microstructpy.meshing.PolyMesh.from_seeds`, again with
``periodic=True``, and the triangular mesh with
:func:`~microstructpy.meshing.TriMesh.from_polymesh`, which reads the
periodicity from the polygonal mesh.
The edge optimization of the polygonal mesh (``edge_opt``) lengthens its
shortest edges and, with the same margin, thickens or removes the pieces of
the grains at the faces that are thinner than the margin.
The periodicity can be restricted to some axes, for example
``periodic='x'``.

The polygonal mesh and the triangular mesh are then drawn four times, in a
2 x 2 tiling of the domain.
The grains cut by the faces of the domain are colored by seed number and
continue across the faces, and the nodes of the triangular mesh on the face
``x = 0`` (red) have their images on the face ``x = 2`` (blue), listed in
``periodic_nodes[0]`` of the mesh.
The tiling is shown in :numref:`f_ex_tiling_2d`.

.. _f_ex_tiling_2d:
.. figure:: ../../../../src/microstructpy/examples/periodic_tiling/tiled_2D.png
  :alt: Periodic 2D microstructure, tiled 2 x 2.

  Periodic polygonal and triangular meshes, tiled 2 x 2.

Periodic Microstructure in 3D
=============================

The domain is a :class:`.Cube` of side length 4, filled with two phases of
spherical grains, periodic in the three directions.
The script prints the number of pairs of nodes on opposite faces of the
tetrahedral mesh, which are exact images of each other along each axis.

The faces of the polyhedral mesh that are visible from the viewpoint are
drawn for the domain and for a copy of the domain translated along ``x``,
in :numref:`f_ex_tiling_3d`: the grains continue across the periodic
face.

.. _f_ex_tiling_3d:
.. figure:: ../../../../src/microstructpy/examples/periodic_tiling/tiled_3D.png
  :alt: Periodic 3D microstructure, tiled twice along x.

  Periodic polyhedral mesh, tiled twice along x.
