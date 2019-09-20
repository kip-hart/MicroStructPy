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

.. literalinclude:: ../../../../examples/two_phase_3D.xml
    :language: xml


Materials
=========

The first material makes up 25% of the volume, with a lognormal grain volume
distribution.

The second material makes up 75% of the volume, with an independent grain
volume distribution.

Domain Geometry
===============

These two materials fill a cube domain of side length 7.


Settings
========

The function will output plots of the microstructure process and those plots
are saved as PNGs.
They are saved in a folder named ``two_phase_3D``, in the current directory
(i.e ``./two_phase_3D``).

The line width of the output plots is reduced to 0.2, to make them more
visible.


Output Files
============

.. only:: not latex

    The three plots that this file generates are the seeding, the polygon mesh,
    and the triangular mesh.
    These three plots are shown below.

    .. figure:: ../../../../examples/two_phase_3D/seeds.png
        :alt: Seed geometries.

        Seed geometries
        
    .. figure:: ../../../../examples/two_phase_3D/polymesh.png
        :alt: Polygonal mesh.

        Polygonal mesh
        
    .. figure:: ../../../../examples/two_phase_3D/trimesh.png
        :alt: Triangular mesh.

        Triangular mesh

.. only:: latex

    .. raw:: latex

        The three plots that this file generates are the seeding, the polygon mesh,
        and the triangular mesh.
        These three plots are shown in Fig.~\ref{fig:ex_two_phase_3D}.

        \begin{figure}[htbp]
            \centering
            \subfloat[Seed geometries]{
                \includegraphics[width=0.3\textwidth]{../../examples/two_phase_3D/seeds.png}
            }
            ~
            \subfloat[Polygonal mesh]{
                \includegraphics[width=0.3\textwidth]{../../examples/two_phase_3D/polymesh.png}
            }
            ~
            \subfloat[Triangular mesh]{
                \includegraphics[width=0.3\textwidth]{../../examples/two_phase_3D/trimesh.png}
            }
            \caption{Output plots for example with elliptical grains.}
            \label{fig:ex_two_phase_3D}
        \end{figure}
