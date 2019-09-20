.. _ex_minimal:

===============
Minimal Example
===============

XML Input File
==============

The basename for this file is ``minimal_paired.xml``.
The file can be run using this command::

    microstructpy --demo=minimal_paired.xml

The full text of the file is:

.. literalinclude:: ../../../../examples/minimal_paired.xml
    :language: xml


Material
========

There is only one material, with a constant size of 0.09.

Domain Geometry
===============

The material fills a square domain.
The default side length is 1, meaning the domain is greater than 10x larger
than the grains.


Settings
========

The function will output plots of the microstructure process and those plots
are saved as PNGs.
They are saved in a folder named ``minimal``, in the current directory
(i.e ``./minimal``).

The axes are turned off in these plots, creating PNG files with
minimal whitespace.

Finally, the seeds and grains are colored by their seed number, not by
material.


Output Files
============

.. only:: not latex

    The three plots that this file generates are the seeding, the polygon mesh,
    and the triangular mesh.
    These three plots are shown below.

    .. figure:: ../../../../examples/minimal/seeds.png
        :alt: Seed geometries.

        Seed geometries
        
    .. figure:: ../../../../examples/minimal/polymesh.png
        :alt: Polygonal mesh.

        Polygonal mesh
        
    .. figure:: ../../../../examples/minimal/trimesh.png
        :alt: Triangular mesh.

        Triangular mesh

.. only:: latex

    .. raw:: latex

        The three plots that this file generates are the seeding, the polygon mesh,
        and the triangular mesh.
        These three plots are shown in Fig.~\ref{fig:ex_minimal}.

        \begin{figure}[htbp]
            \centering
            \subfloat[Seed geometries]{
                \includegraphics[width=0.3\textwidth]{../../examples/minimal/seeds.png}
            }
            ~
            \subfloat[Polygonal mesh]{
                \includegraphics[width=0.3\textwidth]{../../examples/minimal/polymesh.png}
            }
            ~
            \subfloat[Triangular mesh]{
                \includegraphics[width=0.3\textwidth]{../../examples/minimal/trimesh.png}
            }
            \caption{Output plots for minimal example with Paired colormap.}
            \label{fig:ex_minimal}
        \end{figure}
