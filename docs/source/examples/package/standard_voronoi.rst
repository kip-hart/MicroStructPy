.. _ex_std_voro:

========================
Standard Voronoi Diagram
========================

Python Script
=============

The basename for this file is ``standard_voronoi.py``.
The file can be run using this command::

    microstructpy --demo=standard_voronoi.py

The full text of the script is:

.. literalinclude:: ../../../../examples/standard_voronoi.py
    :language: python

Domain
======

The domain of the microstructure is a :class:`.Square`.
Without arguments, the square's center is (0, 0) and side length is 1.

Seeds
=====

A set of 50 seed circles with small radius is initially created.
Calling the :func:`~microstructpy.seeding.seedlist.SeedList.position` method
positions the points according to random uniform distributions in the domain.

Polygon Mesh
============

A polygon mesh is created from the list of seed points using the
:func:`~microstructpy.meshing.polymesh.PolyMesh.from_seeds` class method.
The mesh is plotted and saved into a PNG file in the remaining lines of the
script.

Plotting
========

.. only:: not latex

  The output Voronoi diagram is plotted below.

  .. figure:: ../../../../examples/standard_voronoi/voronoi_diagram.png

    Standard Voronoi diagram.

.. only:: latex

  .. raw:: latex

    The output Voronoi diagram is plotted in
    Fig.~\ref{fig:ex_standard_voronoi}.

    \begin{figure}[htbp]
        \centering
        \includegraphics[width=0.5\textwidth]{../../examples/standard_voronoi/voronoi_diagram.png}
        \caption{Standard Voronoi diagram.}
        \label{fig:ex_standard_voronoi}
    \end{figure}
