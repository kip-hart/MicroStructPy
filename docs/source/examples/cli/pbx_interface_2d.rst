.. _ex_pbx_interface_2d:

============================
Interface Refinement Example
============================

XML Input File
==============

The basename for this file is ``pbx_interface_2D.xml``.
The file can be run using this command::

    microstructpy --demo=pbx_interface_2D.xml

The full text of the file is:

.. literalinclude:: ../../../../src/microstructpy/examples/pbx_interface_2D.xml
    :language: xml


Materials and Domain
====================

The materials and the domain are those of the :ref:`ex_pbx_2d` example:
crystalline inclusions in a binder, in a square domain periodic in both
directions.

Settings
========

The mesh is refined along the grain boundaries, which are the interfaces
between the binder and the inclusions (and between neighboring inclusions),
and coarse inside the grains.

``mesh_max_edge_length`` sets the maximum length of the element edges along
the grain boundaries, 0.03 here, while ``mesh_max_volume`` sets the maximum
area of the elements, 0.02 here, which is the area of a triangle with edges
about seven times longer.
Triangle grades the element size between the two.

The mesh has a minimum angle of 25 degrees.

The seeds are placed a margin away from the periodic faces (``periodic_margin``
set to ``auto``: half the target edge length, or an eighth of the smallest
grain if that is smaller) and the edge optimization moves the seeds that
leave a thin piece of a grain on a face, or a very short edge, since either
forces very small triangles.


Output Files
============

The three plots that this file generates are the seeding, the polygon mesh,
and the triangular mesh.
The triangular mesh is shown in :numref:`f_ex_pbxint2d_tri`.

.. _f_ex_pbxint2d_tri:
.. figure:: ../../../../src/microstructpy/examples/pbx_interface_2D/trimesh.png
    :alt: Triangular mesh.

    Interface refinement example - triangular mesh.
