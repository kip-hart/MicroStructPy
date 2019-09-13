.. _ex_2_quality:

================
Quality Controls
================

XML Input File
==============

The basename for this file is ``intro_2_quality.xml``.
The file can be run using this command::

    microstructpy --demo=intro_2_quality.xml

The full text of the file is:

.. literalinclude:: ../../examples/intro_2_quality.xml
    :language: xml


Materials
=========

There are two materials, in a 2:1 ratio based on volume.
The first is a matrix, which is represented with small circles.
The second material consists of circular inclusions with diameter 2.

Domain Geometry
===============

These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions to ensure that
microstructure is statistically representative.


Settings
========

The first two settings determine the output directory and whether to run the
program in verbose mode.
The following settings determine the quality of the triangular mesh.

The minimum interior angle of the elements is 25 degrees, ensuring lower
aspect ratios compared to the first example.
The maximum area of the elements is also limited to 1, which populates the
matrix with smaller elements.
Finally, The maximum edge length of elements at interfaces is set to 0.1,
which increasing the mesh density surrounding the inclusions and at the
boundary of the domain.

Note that the edge length control is currently unavailable in 3D.


Output Files
============

.. only:: not latex

    The three plots that this file generates are the seeding, the polygon mesh,
    and the triangular mesh.
    These three plots are shown below.

    .. figure:: ../../examples/intro_2_quality/seeds.png
        :alt: Seed geometries.

        Seed geometries
        
    .. figure:: ../../examples/intro_2_quality/polymesh.png
        :alt: Polygonal mesh.

        Polygonal mesh
        
    .. figure:: ../../examples/intro_2_quality/trimesh.png
        :alt: Triangular mesh.

        Triangular mesh

.. only:: latex

    .. raw:: latex

        The three plots that this file generates are the seeding, the polygon mesh,
        and the triangular mesh.
        These three plots are shown in Fig.~\ref{fig:ex_2_quality}.

        \begin{figure}[htbp]
            \centering
            \subfloat[Seed geometries]{
                \includegraphics[width=0.3\textwidth]{../../examples/intro_2_quality/seeds.png}
            }
            ~
            \subfloat[Polygonal mesh]{
                \includegraphics[width=0.3\textwidth]{../../examples/intro_2_quality/polymesh.png}
            }
            ~
            \subfloat[Triangular mesh]{
                \includegraphics[width=0.3\textwidth]{../../examples/intro_2_quality/trimesh.png}
            }
            \caption{Output plots for mesh quality example.}
            \label{fig:ex_2_quality}
        \end{figure}
