.. _cli_settings:

=========================
``<settings>`` - Settings
=========================

Defaults
++++++++

Settings can be added to the input file to specify file outputs and mesh
quality, among other things. The default settings are:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Default settings -->
    <input>
        <settings>
            <!-- File and Console I/O -->
            <verbose> False </verbose>
            <directory> . </directory>

            <filetypes>
                <seeds> txt </seeds>
                <poly> txt </poly>
                <tri> txt </tri>
                <seeds_plot> png </seeds_plot>
                <poly_plot> png </poly_plot>
                <tri_plot> png </tri_plot>
                <verify_plot> png </verify_plot>
            </filetypes>

            <!-- Run Settings -->
            <restart> True </restart>

            <rng_seeds>
                <position> 0 </position>
                <!-- RNG can be set for grain shape distributions as well. -->
                <!-- For example, <size> 2 </size> seeds the RNG for       -->
                <!-- sampling size distributions with 2.                   --> 
            </rng_seeds>

            <rtol> fit </rtol>

            <edge_opt> False </edge_opt>
            <edge_opt_n_iter> 100 </edge_opt_n_iter>

            <mesher> Triangle/TetGen </mesher>

            <mesh_size> inf </mesh_size> <!-- used with gmsh -->
            <mesh_max_volume> inf </mesh_max_volume> <!-- used with Triangle and TetGen -->
            <mesh_min_angle> 0 </mesh_min_angle> <!-- used with Triangle and TetGen -->
            <mesh_max_edge_length> inf </mesh_max_edge_length> <!-- used with Triangle -->

            <verify> False </verify>

            <!-- Plot Controls -->
            <plot_axes> True </plot_axes>
            
            <color_by> material </color_by>
            <colormap> viridis </colormap>
            
            <seeds_kwargs> </seeds_kwargs>
            <poly_kwargs> </poly_kwargs>
            <tri_kwargs> </tri_kwargs>
        </settings>
    </input>

File and Console I/O
++++++++++++++++++++

verbose
-------

The verbose flag toggles text updates to the console as MicroStructPy runs.
Setting ``<verbose> True </verbose>`` will print updates, while False turns
them off.

directory
---------

The directory field is for the path to the output files.
It can be an absolute file path, or relative to the input file.
For example, if the file is in ``aa/bb/cc/input.xml`` and the directory field
is ``<directory> ../output </directory>``, then MicroStructPy will write
output files to ``aa/bb/output/``.
If the output directory does not exist, MicroStructPy will create it.

filetypes
---------

This field is for specifying output filetypes.
The possible subfields are seeds, seeds_plot, poly, poly_plot, tri, tri_plot,
and verify_plot.
Below is an outline of the possible filetypes for each subfield.

- seeds

    **txt**, **vtk**

    Currently the only options are to output the seed geometries as a
    cache txt file or as a VTK legacy file. The VTK file can be opened
    in ParaView, with the seeds shown as glyphs.

- seeds_plot

    **ps**, **eps**, **pdf**, **pgf**, **png**, **raw**, **rgba**, **svg**,
    **svgz**, **jpg**, **jpeg**, **tif**, **tiff**

    These are the standard matplotlib output filetypes.

- poly

    **txt**, **poly** (2D only), **ply**, **vtk**

    A poly file contains a planar straight line graph (PSLG) and cane be read
    by Triangle.
    More details on poly files can be found on the `.poly files`_ page of the
    Triangle website.
    The ply file contains the surfaces between grains and the boundary of the
    domain.
    VTK legacy files also contain the polygonal grains in 2D and polyhedral
    grains in 3D.

- poly_plot

    **ps**, **eps**, **pdf**, **pgf**, **png**, **raw**, **rgba**, **svg**,
    **svgz**, **jpg**, **jpeg**, **tif**, **tiff**

    These are the standard matplotlib output filetypes.

- tri

    **txt**, **abaqus**, **tet/tri**, **vtk** (3D only)

    The abaqus option will create a part for each grain and assembly the parts.
    The tet/tri option will create .node and .elem files in the same format as
    the output of Triangle or TetGen.
    VTK files are suitable for viewing the mesh interactively in a program such
    as Paraview.

- tri_plot

    **ps**, **eps**, **pdf**, **pgf**, **png**, **raw**, **rgba**, **svg**,
    **svgz**, **jpg**, **jpeg**, **tif**, **tiff**

    These are the standard matplotlib output filetypes.

- verify_plot

    **ps**, **eps**, **pdf**, **pgf**, **png**, **raw**, **rgba**, **svg**,
    **svgz**, **jpg**, **jpeg**, **tif**, **tiff**

    These are the standard matplotlib output filetypes.


For example:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <settings>
            <filetypes>
                <seeds> txt </seeds>
                <seeds_plot> png, pdf </seeds_plot>
                <poly> txt, ply </poly>
                <poly_plot> svg </poly_plot>
                <tri> txt </tri>
                <tri_plot> pdf </tri_plot>
                <verify_plot> pdf </verify_plot>
            </filetypes>
        </settings>
    </input>

If a subfield is not specified, that output is not saved to any file.
The exception is, if ``<restart> True </restart>``, then the seeds, poly mesh,
and tri mesh will all be output to txt files.

Run Settings
++++++++++++

restart
-------

The restart flag will read the intermediate txt output files, if they exist,
instead of duplicating previous work.
Setting ``<restart> True </restart>`` will read the txt files, while False will
ignore the existing txt files.

rng_seeds
---------

The random number generator (RNG) seeds can be included to create multiple,
repeatable realizations of a microstructure.
By default, RNG seeds are all set to 0.
An RNG seed can be specified for any of the distributed parameters in grain
geometry.
For example:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> circle </shape>
            <radius>
                <dist_type> uniform </dist_type>
                <loc> 1 </loc>
                <scale> 2 </scale>
            </radius>
        </material>

        <material>
            <shape> ellipse </shape>
            <axes> 1, 2 </axes>
            <angle_deg>
                <dist_type> norm </dist_type>
                <loc> 0 <loc>
                <scale> 15 </scale>
            </angle_deg>
        </material>

        <settings>
            <rng_seeds>
                <radius> 1 </radius>
                <angle_deg> 0 </angle_deg>
                <position> 3 </position>
            </rng_seeds>
        </settings>
    </input>

In this case, if the position RNG were changed from 3 to 4 and the rest of the
RNG seeds remained the same, MicroStructPy would generate the same set of seed
geometries and arrange them differently in the domain.

rtol
----

The rtol field is for the relative overlap tolerance between seed geometries.
The overlap is relative to the radius of the smaller circle or sphere.
Overlap is acceptable if

.. math::

    \frac{r_1 + r_2 - ||x_1 - x_2||}{min(r_1, r_2)} < rtol


The default value is ``<rtol> fit </rtol>``, which uses a fit curve to
determine an appropriate value of rtol.
This curve considers the coefficient of variation in grain volume and estimates
an rtol value that maximizes the fit between input and output distributions.

Acceptable values of rtol are 0 to 1 inclusive, though rtol below 0.2 will
likely result in long runtimes.

edge_opt
--------

The edge_opt field provides the option to maximize the shortest edge in the
polygonal/polyhedral mesh.
The default is ``<edge_opt> False </edge_opt>``, which skips the optimization
process.
This optimization is performed by making small adjustments to the positions of
seeds surrounding the shortest edge, assessing if the change created an
improvement, then either a) attempting a different change for the same edge if
there was not improvement or b) moving on to the new shortest edge.
The optimization algorithm exits when ``edge_opt_n_iter`` iterations have been
performed on the same edge.

This flag is useful if the polygonal/polyhedral or triangular/tetrahedral are
used in numerical simulations, such as finite element analysis.
A high ratio of longest edge to shortest edge leads to a high ratio in maximum
to minimum eigenvalue in FEA stiffness matrices, which can create problems for
the FEA solver.
Setting ``edge_opt`` to ``True`` will reduce short edges in the polygonal mesh,
which translates into reduced short edges in the triangular mesh.
This optimization process, however, will increase the time to generate a
polygonal mesh.
To track the progress of the optimizer, set ``verbose`` to ``True``.

In a periodic domain (see :ref:`cli_domain`), the cells that cross a periodic
face are split into pieces, and a cell that barely crosses a face leaves a
thin piece on the opposite face, which forces very small elements there.
With ``edge_opt`` and a positive ``periodic_margin``, the optimizer also
treats the thickness of each piece at a periodic face as a feature of the
mesh, like an edge, and moves the seeds of every piece thinner than the
margin (and of its neighbors) normal to the face, until the piece is at least
as thick as the margin or the cell no longer crosses the face.
A corner of a cell at a periodic face that is narrower than ``mesh_min_angle``
is treated the same way: the mesher cannot reach the minimum angle in such a
corner and fills it with shells of very small elements instead, so the
optimizer moves the seeds on both sides of the facet along it to open the
corner.
A change is kept when the shortest feature it modifies gets longer, so the
shorter features of the mesh never get worse.

edge_opt_n_iter
---------------

This field specifies how many times the optimizer should attempt to increase
the length of the shortest edge in the polygonal mesh (or the thickness of a
piece at a periodic face).
The default is ``<edge_opt_n_iter> 100 </edge_opt_n_iter>``, which limits the
optimizer to 100 attempts per edge.
This field is ignored if ``edge_opt`` is set to ``False``.

mesher
------

This field specifies how to mesh the PolyMesh. If set to ``raster``, the output
mesh will contain pixels/voxels of the PolyMesh.
Other options include ``Triangle/Tetgen`` and ``gmsh``.

mesh_size
---------

This field specifies a target element size if using the ``gmsh`` mesher.

mesh_max_volume
---------------

This field defines the maximum volume (or area, in 2D) of any element in the
triangular (unstructured) mesh.
The default is ``<mesh_max_volume> inf </mesh_max_volume>``, which turns off
the volume control.
In this example:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> circle </shape>
            <area> 0.01 </area>
        </material>

        <domain>
            <shape> square </shape>
            <side_length> 1 </side_length>
        </domain>

        <settings>
            <mesh_max_volume> 0.001 </mesh_max_volume>
        </settings>
    </input>

the unstructured mesh will have at least 10 elements per grain and at least
1000 elements overall.

mesh_min_angle
--------------

This field defines the minimum interior angle, measured in degrees, of any
element in the triangular mesh.
For 3D meshes, this is the minimum *dihedral* angle, which is between faces of
the tetrahedron.
This setting controls the aspect ratio of the elements, with angles between
15 and 30 degrees producing good quality meshes.
The default is ``<mesh_min_angle> 0 </mesh_min_angle>``, which effectively
turns off the angle quality control.

periodic_margin
---------------

This field sets the minimum distance between the surface of a seed and a
periodic face of the domain (see :ref:`cli_domain` for periodic domains).
A position where a seed ends within this distance inside a face, or crosses
a face by less than this distance, is rejected and another one is tried:
such seeds leave thin pieces of grains on the faces of the domain, and
elements much smaller than the target size of the mesh there.
The value ``auto`` uses the smaller of two lengths: half the target edge
length of the mesh, taken from ``mesh_max_edge_length`` or, if it is not set,
from ``mesh_max_volume``; and an eighth of the size of the smallest seed (its
smallest diameter or side).
The smallest seeds need about four elements across them, so the mesh cannot
be coarser than a quarter of their size, and a seed cannot satisfy a margin
larger than itself: with such a margin, the positions near the faces would
all be rejected and the seeds dropped.
A large margin makes the seeds harder to place near the faces.
The margin also applies to the cells: with ``edge_opt``, the optimizer
thickens or removes the pieces of the cells at the periodic faces that are
thinner than the margin, which the placement of the seeds alone cannot
prevent (a cell extends beyond its seed, and its corners can cross a face
by a small amount).
The default is ``<periodic_margin> 0 </periodic_margin>``, which turns off
the margin. It has no effect on non-periodic domains.

mesh_max_edge_length
--------------------

This field defines the maximum edge length along a grain boundary: of the
segments of a 2D triangular mesh, and of the triangles on the facets of a 3D
tetrahedral mesh.
A small maximum edge length will increase resolution of the mesh at grain
boundaries, while the size of the elements inside the grains is controlled
by the maximum volume (see the :ref:`ex_pbx_interface_2d` and
:ref:`ex_pbx_interface_3d`).
The default value is ``<mesh_max_edge_length> inf </mesh_max_edge_length>``,
which effectively turns off the edge length quality control.

verify
------

The verify flag will perform mesh verification on the triangular mesh and
report error metrics.
To include mesh verification, include ``<verify> True </verify>`` in the
settings.
The default behavior is to not perform mesh verification.

Plot Controls
+++++++++++++

plot_axes
---------

The plot_axes flag toggles the axes on or off in the output plots.
Setting it to False turns the axes off, producing images with miniminal
borders.
The default setting is ``<plot_axes> True </plot_axes>``, which includes the
coordinate axes in output plots.

color_by
--------

The color_by field defines how the seeds and grains should be colored in the
output plots.
There are three options for this field: "material", "seed number", and
"material number".
The default setting is ``<color_by> material </color_by>``.
Using "material", the output plots will color each seed/grain with the color
of its material.
Using "seed number", the seeds/grains are colored by their seed number, which
is converted into a color using the ``colormap``.
The "material number" option behaves in the same was as "seed number", except
that the material numbers are used instead of seed numbers.

colormap
--------

The colormap field is used when ``color_by`` is set to either "seed number" or
"material number".
This gives the name of the colormap to be used in coloring the seeds/grains.
For a complete list of available colormaps, visit the `Choosing Colormaps in
Matplotlib`_ webpage.

seeds_kwargs
------------

This field contains optional keyword arguments passed to matplotlib when
plotting the seeds.
For example:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <settings>
            <seeds_kwargs>
                <edgecolor> none </edgecolor>
                <alpha> 0.5 </alpha>
            </seeds_kwargs>
        </settings>
    </input>

will plot the seeds with some transparency and no borders.

poly_kwargs
-----------

This field contains optional keyword arguments passed to matplotlib when
plotting the polygonal mesh.
For example:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <settings>
            <poly_kwargs>
                <linewidth> 0.5 </linewidth>
                <edgecolors> blue </edgecolors>
            </poly_kwargs>
        </settings>
    </input>

will plot the mesh with thin, blue lines between the grains.

tri_kwargs
----------

This field contains optional keyword arguments passed to matplotlib when
plotting the triangular mesh.
For example:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <settings>
            <tri_kwargs>
                <linewidth> 0.5 </linewidth>
                <edgecolors> white </edgecolors>
            </tri_kwargs>
        </settings>
    </input>

will plot the mesh with thin, white lines between the elements.


.. _`Choosing Colormaps in Matplotlib`: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
.. _`.poly files`: https://www.cs.cmu.edu/~quake/triangle.poly.html
