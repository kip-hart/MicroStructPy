.. _cli:

==================
Command Line Guide
==================

Basics
------

The command line interface (CLI) for this package is ``microstructpy``.
This command accepts the names of user-generated files and demonstration files.
Multiple filenames can be specified.

For example, to run the XML file that creates the microstructure on the front
page, run::

    microstructpy --demo=docs_banner.xml

This command will copy ``docs_banner.xml`` to the current working directory,
then run that XML file.
For your own input files, run the command::

    microstructpy path/to/my/input_file.xml

Note that relative and absolute file paths are acceptable.

To run multiple input files::

    microstructpy file_1.xml file_2.xml file_3.xml

or::

    microstructpy file_*.xml

To run all of the demonstration files, use the command::

    microstructpy --demo=all

Note that this may take some time.

Command Line Procedure
^^^^^^^^^^^^^^^^^^^^^^

The following tasks are performed by the CLI:

1. Make the output directory, if necessary
2. Create a list of unpositioned seeds
3. Position the seeds in the domain
4. Save the seeds in a text file
5. Save a plot of the seeds to an image file
6. Create a polygon mesh from the seeds
7. Save the mesh to the output directory
8. Save a plot of the mesh to the output directory
9. Create a unstructured (triangular or tetrahedral) mesh
10. Save the unstructured mesh
11. Save a plot of the unstructured mesh
12. (optional) Verify the output mesh against the input file.

Intermediate results are saved in steps 4, 7, and 10 to give the option of
restarting the procedure.
The format of the output files can be specified in the input file
(e.g. png and/or PDF plots).

Minimal Input File
^^^^^^^^^^^^^^^^^^

Input files for MicroStructPy must be in XML format and included a minimum
of 2 pieces of information: the material phases and the domain.
A minimal input file is:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> circle </shape>
            <size> 0.1 </size>
        </material>

        <domain>
            <shape> square </shape>
        </domain>
    </input>


This will create a microstructure with approximately circular grains that
fill a domain that is 10x larger.
MicroStructPy will output three files: ``seeds.txt``, ``polymesh.txt``, and
``trimesh.txt``.
To output plots requires addition settings, described in the :ref:`settings`
section.

The :ref:`materials` section describes more advanced materials specification,
while the :ref:`domain` section discusses specifying the geometry of the
domain.

.. note::

    XML fields that are not recognized by MicroStructPy will be ignored by the
    program. For example, material properties or notes can be included in the
    file without affecting program execution.

Also note that the order of fields in an XML file is not strictly important,
since the file is converted into a dictionary.


.. _materials:

Material Phases
---------------

Multiple Materials
^^^^^^^^^^^^^^^^^^

MicroStructPy supports an arbitrary number of materials within a
microstructure.
For example:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> circle </shape>
            <size> 1 </size>
            <fraction> 0.2 </fraction>
        </material>

        <material>
            <shape> circle </shape>
            <size> 0.5 </size>
            <fraction> 0.3 </fraction>
        </material>

        <material>
            <shape> circle </shape>
            <size> 1.5 </size>
            <fraction> 0.5 </fraction>
        </material>

        <domain>
            <shape> square </shape>
            <side_length> 10 </side_length>
        </domain>
    </input>

Here there are three phases: the first has grain size 1 and makes up 20% of the
area, the second has grain size 0.5 and makes up 30% of the area,
and the third has grain size 1.5 and makes up 50% of the area.
If the fractions are not specified, MicroStructPy assumes the phases have equal
volume fractions.
The fractions can also be given as ratios (e.g. 2, 3, and 5) and
MicroStructPy will normalize them to fractions.

Grain Size Distributions
^^^^^^^^^^^^^^^^^^^^^^^^

Distributed grain sizes, rather than constant sizes, can be specified as
follows:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> circle </shape>
            <size>
                <dist_type> uniform </dist_type>
                <loc> 1 </loc>
                <scale> 1 </scale>
            </size>
        </material>

        <material>
            <shape> circle </shape>
            <size>
                <dist_type> lognorm </dist_type>
                <scale> 0.5 </scale>
                <s> 0.1 </s>
            </size>
        </material>

        <material>
            <shape> circle </shape>
            <size>
                <dist_type> cdf </dist_type>
                <filename> my_empirical_dist.csv </filename>
            </size>
        </material>

        <domain>
            <shape> square </shape>
            <side_length> 10 </side_length>
        </domain>
    </input>

In all three materials, the ``size`` field contains a ``dist_type``.
This type can match the name of one of `SciPy's statistical functions`_, or be
either "pdf" or "cdf".
If it is a SciPy distribution name, then the remaining parameters must match
the inputs for that function.
The first material has size distribution :math:`S\sim U(1, 2)` and the second
has distribution :math:`S\sim 0.5e^{N(0, 0.1)}`. Refer to the SciPy website for
the complete list of available distributions and their input parameters.

In the case that the distribution type is "pdf" then the only other field
should be ``filename``.
For a PDF, the file should contain two lines: the first has the (n+1) bin
locations and the second has the (n) bin heights.
A PDF file could contain, for example::

    1, 2, 2.5
    0.5, 1

For a CDF, the file should have two columns: the first being the size and the
second being the CDF value.
The equivalent CDF file would contain::

    1, 0
    2, 0.5
    2.5, 1

Both PDF and CDF files should be in CSV format.

.. warning::

    Do not use distributions that are equivalent to a deterministic value,
    such as :math:`S\sim N(1, 0)`. The infinite PDF value causes numerical
    issues for SciPy. Instead, replace the distribution with the deterministic
    value or use a small, non-zero variance.

Grain Geometries
^^^^^^^^^^^^^^^^

MicroStructPy supports the following grain geometries:

* Circle (2D)
* Ellipse (2D)
* Ellipsoid (3D)
* Rectangle (2D)
* Sphere (3D)
* Square (2D)

Each geometry can be specified in multiple ways.
For example, the ellipse can be specified in terms of its area and aspect
ratio, or by its semi-major and semi-minor axes.
The 'size' of a grain is defined as the diameter of a circle or sphere with
equivalent area (so for a general ellipse, this would be :math:`2\sqrt{a b}`).
The parameters available for each geometry are described in the lists below.

Circle
++++++

- radius (or r)
- diameter (or d)
- size (same as d)
- area

Ellipse
+++++++

- a
- b
- size
- aspect_ratio
- angle_deg
- angle_rad
- angle (same as angle_deg)
- axes (equivalent to [a, b])
- matrix
- orientation (same as matrix)

Ellipsoid
+++++++++

- a
- b
- c
- size
- ratio_ab or ratio_ba
- ratio_ac or ratio_ca
- ratio_bc or ratio_cb
- rot_seq_deg
- rot_seq_rad
- rot_seq (same as rot_seq_deg)
- axes (equivalent to [a, b, c])
- matrix
- orientation (same as matrix)

Rectangle
+++++++++

- length
- width
- side_lengths (equivalent to [length, width])
- angle_deg
- angle_rad
- angle (same as angle_deg)
- matrix

Sphere
++++++

- radius (or r)
- diameter (or d)
- size (same as d)
- volume

Square
++++++

- side_length
- angle_deg
- angle_rad
- angle (same as angle_deg)
- matrix

.. note::
    Over-parameterizing grain geometries will cause unexpected behavior.

For parameters such as "side_lengths" and "axes", the input is expected to be
a list, e.g. ``<axes> 1, 2 </axes>`` or ``<axes> (1, 2) </axes>``.
For matrices, such as "orientation", the input is expected to be a list of
lists, e.g. ``<orientation> [[0, -1], [1, 0]] </orientation>``.

Each of the scalar arguments can be either a constant value or a distribution.
For uniform random distribution of ellipse and ellipsoid axes, used the
parameter ``<orientation> random </orientation>``.
The default orientation is axes-aligned.

Here is an example input file with non-circular grains:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> ellipse </shape>
            <size>
                <dist_type> uniform </dist_type>
                <loc> 1 </loc>
                <scale> 1 </scale>
            </size>
            <aspect_ratio> 3 </aspect_ratio>
            <orientation> random </orientation>
        </material>

        <material>
            <shape> square </shape>
            <side_length>
                <dist_type> lognorm </dist_type>
                <scale> 0.5 </scale>
                <s> 0.1 </s>
            </side_length>
        </material>

        <material>
            <shape> rectangle </shape>
            <length>
                <dist_type> cdf </dist_type>
                <filename> my_empirical_dist.csv </filename>
            </length>
            <width> 0.2 </width>
            <angle_deg>
                <dist_type> uniform <dist_type>
                <loc> -30 </loc>
                <scale> 60 </scale>
            </angle_deg>
        </material>

        <domain>
            <shape> square </shape>
            <side_length> 10 </side_length>
        </domain>
    </input>


Material Type
^^^^^^^^^^^^^

There are three types of materials supported by MicroStructPy: crystalline,
amorphous, and void.
For amorphous phases, the facets between cells of the same material type are
removed before unstructured meshing.
Several aliases are available for each type, given in the list below.

* crystalline
    + solid
    + granular
* amorphous
    + matrix
    + glass
* void
    + crack
    + hole

The default material type is crystalline.
An example input file with material types is:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> circle </shape>
            <size>
                <dist_type> uniform </dist_type>
                <loc> 0 </loc>
                <scale> 1 </scale>
            </size>
            <material_type> matrix </material_type>
        </material>

        <material>
            <shape> square </shape>
            <side_length> 0.5 </side_length>
            <material_type> void </material_type>
        </material>

        <domain>
            <shape> square </shape>
            <side_length> 10 </side_length>
        </domain>
    </input>


Here, the first phase is an amorphous (matrix) phase and the second phase
contains square voids of constant size.

Multiple amorphous and void phases can be present in the material.


Grain Position Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default position distribution for grains is random uniform throughout the
domain.
Grains can be non-uniformly distributed by adding a position distribution.
The x, y, and z can be independently distributed or coupled.
The coupled distributions can be any of the multivariate distributions listed
on `SciPy's statistical functions`_ page.


In the example below, the first material has independently distributed
coordinates while the second has a coupled distribution.

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> circle </shape>
            <size>
                <dist_type> uniform </dist_type>
                <loc> 0 </loc>
                <scale> 1 </scale>
            </size>
            <position>  <!-- x -->
                <dist_type> binom </dist>
                <loc> 0.5 </loc>
                <n> 9 </n>
                <p> 0.5 </p>
            </position>
            <position> <!-- y -->
                <dist_type> uniform </dist>
                <loc> 0 </loc>
                <scale> 10 </scale>
            </position>
        </material>

        <material>
            <shape> square </shape>
            <side_length> 0.5 </side_length>
            <position>
                <dist_type> multivariate_normal </dist_type>
                <mean> [2, 3] </mean>
                <cov> [[4, -1], [-1, 3]] </cov>
            </position>
        </material>

        <domain>
            <shape> square </shape>
            <side_length> 10 </side_length>
            <corner> 0, 0 </corner>
        </domain>
    </input>

Position distributions should be used with care, as seeds may not fill the
entire domain.


Other Material Settings
^^^^^^^^^^^^^^^^^^^^^^^

**Name** The name of each material can be specified by adding a "name" field.
The default name is "Material N" where N is the order of the material in
the XML file, starting from 0.

**Color** The color of each material in output plots can be specified by adding
a "color" field.
The default color is "CN" where N is the order of the material in the XML file,
starting from 0.
For more information about color specification, visit the Matplotlib
`Specifying Colors`_ page.

For example:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <name> Aluminum </name>
            <color> silver </color>
            <shape> circle </shape>
            <size> 1 </size>
        </material>

        <domain>
            <shape> square </shape>
            <side_length> 10 </side_length>
        </domain>
    </input>


.. _domain:

Domain
------

MicroStructPy supports the following domain geometries:

* Box (3D)
* Circle (2D)
* Cube (3D)
* Ellipse (2D)
* Rectangle (2D)
* Square (2D)

Each geometry can be defined several ways, such as a center and edge lengths
for the rectangle or two bounding points.
Note that over-parameterizing the domain geometry will cause unexpected
behavior.

Box
^^^

The parameters available for defining a 3D box domain are:

- side_lengths
- center
- corner (i.e. :math:`(x, y, z)_{min}`)
- limits (i.e. :math:`[[x_{min}, x_{max}], [y_{min}, y_{max}], [z_{min}, z_{max}]]`)
- bounds (same as limits)

Below are some example box domain definitions.

.. code-block:: XML

    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Example box domains -->
    <input>
        <domain>
            <shape> box </shape>
            <!-- default side length is 1 -->
            <!-- default center is the origin -->
        </domain>

        <domain>
            <shape> box </shape>
            <side_lengths> 2, 1, 6 </side_lengths>
            <corner> 0, 0, 0 </corner>
        </domain>

        <domain>
            <shape> BOX </shape>
            <limits> 0, 2 </limits>   <!-- x -->
            <limits> -2, 1 </limits>  <!-- y -->
            <limits> -3, 0 </limits>  <!-- z -->
        </domain>

        <domain>
            <shape> boX </shape>  <!-- case insensitive -->
            <bounds> [[0, 2], [-2, 1], [-3, 0]] </bounds>
        </domain>
    </input>

Circle
^^^^^^

The parameters available for defining a 2D circle domain are:

- radius (or r)
- diameter (or d)
- size (same as diameter)
- area
- center

Below are some example circle domain definitions.

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Example circle domains -->
    <input>
        <domain>
            <shape> circle </shape>
            <!-- default radius is 1 -->
            <!-- default center is the origin -->
        </domain>

        <domain>
            <shape> circle </shape>
            <diameter> 3 </diameter>
        </domain>

        <domain>
            <shape> circle </shape>
            <radius> 10 </radius>
            <center> 0, 10 <center>
        </domain>
    </input>


Cube
^^^^

The parameters available for defining a 3D cube domain are:

- side_length
- center
- corner (i.e. :math:`(x, y, z)_{min}`)

Below are some example cube domain definitions.

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Example cube domains -->
    <input>
        <domain>
            <shape> cube </shape>
            <!-- default side length is 1 -->
            <!-- default center is the origin -->
        </domain>

        <domain>
            <shape> cube </shape>
            <side_length> 10 </side_length>
            <corner> (0, 0, 0) </corner>
        </domain>

        <domain>
            <shape> cube </shape>
            <corner> 0, 0, 0 </corner>
        </domain>
    </input>


Ellipse
^^^^^^^

The parameters available for defining a 2D ellipse domain are:

- a
- b
- axes
- size
- aspect_ratio
- angle_deg
- angle_rad
- angle (same as angle_deg)
- matrix
- orientation (same as matrix)
- center

Below are some example ellipse domain definitions.

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Example ellipse domains -->
    <input>
        <domain>
            <shape> ellipse </shape>
            <!-- default is a unit circle centered at the origin -->
        </domain>

        <domain>
            <shape> ellipse </shape>
            <a> 10 </a>
            <b>  4 </b>
            <angle> 30 </angle>
            <center> 2, -1 </center>
        </domain>

        <domain>
            <shape> ellipse </shape>
            <axes> 5, 3 </axes>
        </domain>

        <domain>
            <shape> ellipse </shape>
            <size> 10 </size>
            <aspect_ratio> 5 </aspect_ratio>
            <angle_deg> -45 </angle_deg>
        </domain>
    </input>


Rectangle
^^^^^^^^^

The parameters available to define a 2D rectangle domain are:

- length
- width
- side_lengths
- center
- corner (i.e. :math:`(x, y)_{min}`)
- limits (i.e. :math:`[[x_{min}, x_{max}], [y_{min}, y_{max}]]`)
- bounds (same as limits)

Below are some example rectangle domain definitions.

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Example rectangle domains -->
    <input>
        <domain>
            <shape> rectangle </shape>
            <!-- default side length is 1 -->
            <!-- default center is the origin -->
        </domain>

        <domain>
            <shape> rectangle </shape>
            <side_lengths> 2, 1 </side_lengths>
            <corner> 0, 0 </corner>
        </domain>

        <domain>
            <shape> rectangle </shape>
            <limits> 0, 2 </limits>   <!-- x -->
            <limits> -2, 1 </limits>  <!-- y -->
        </domain>

        <domain>
            <shape> rectangle </shape>
            <bounds> [[0, 2], [-2, 1]] </bounds>
        </domain>
    </input>


Square
^^^^^^

The parameters available to define a 2D square domain are:

- side_length
- center
- corner (i.e. :math:`(x, y)_{min}`)

Below are some example square domain definitions.

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Example square domains -->
    <input>
        <domain>
            <shape> square </shape>
            <!-- default side length is 1 -->
            <!-- default center is the origin -->
        </domain>

        <domain>
            <shape> square </shape>
            <side_length> 2 </side_length>
            <corner> 0, 0 </corner>
        </domain>

        <domain>
            <shape> square </shape>
            <corner> 0, 0 </corner>
        </domain>

        <domain>
            <shape> square </shape>
            <side_length> 10 </side_length>
            <center> 5, 0 </center>
        </domain>
    </input>


.. _settings:

Settings
--------

Settings can be added to the input file to specify file outputs and mesh
quality, among other things. The default settings are:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Default settings -->
    <input>
        <settings>
            <verbose> False </verbose>
            <restart> True </restart>
            <directory> . </directory>

            <filetypes>
                <seeds> txt </seeds>
                <poly> txt </poly>
                <tri> txt </tri>
            </filetypes>

            <rng_seeds>
                <position> 0 </position>
            </rng_seeds>

            <rtol> fit </rtol>
            <mesh_max_volume> inf </mesh_max_volume>
            <mesh_min_angle> 0 </mesh_min_angle>
            <mesh_max_edge_length> inf </mesh_max_edge_length>

            <verify> True </verify>

            <plot_axes> True </plot_axes>
            <color_by> material </color_by>
            <colormap> viridis </colormap>
            <seeds_kwargs> </seeds_kwargs>
            <poly_kwargs> </poly_kwargs>
            <tri_kwargs> </tri_kwargs>
        </settings>
    </input>

verbose
^^^^^^^

The verbose flag toggles text updates to the console as MicroStructPy runs.
Setting ``<verbose> True </verbose>`` will print updates, while False turns
them off.

restart
^^^^^^^

The restart flag will read the intermediate txt output files, if they exist,
instead of duplicating previous work.
Setting ``<restart> True </restart>`` will read the txt files, while False will
ignore the existing txt files.

directory
^^^^^^^^^

The directory field is for the path to the output files.
It can be an absolute file path, or relative to the input file.
For example, if the file is in ``aa/bb/cc/input.xml`` and the directory field
is ``<directory> ../output </directory>``, then MicroStructPy will write
output files to ``aa/bb/output/``.
If the output directory does not exist, MicroStructPy will create it.

filetypes
^^^^^^^^^

This field is for specifying output filetypes.
The possible subfields are seeds, seeds_plot, poly, poly_plot, tri, tri_plot,
and verify_plot.
Below is an outline of the possible filetypes for each subfield.

- seeds

    **txt**

    Currently the only option is to output the seed geometries as a
    cache txt file.

- seeds_plot

    **ps**, **eps**, **pdf**, **pgf**, **png**, **raw**, **rgba**, **svg**,
    **svgz**, **jpg**, **jpeg**, **tif**, **tiff**

    These are the standard matplotlib output filetypes.

- poly

    **txt**, **poly** (2D only), **ply**, **vtk** (3D only)

    A poly file contains a planar straight line graph (PSLG) and cane be read
    by Triangle.
    More details on poly files can be found on the `.poly files`_ page of the
    Triangle website.
    The ply file contains the surfaces between grains and the boundary of the
    domain.
    VTK legacy files also contain the polygonal surfaces between grains.

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

If a subfield is not specified, the default behavior is not to save that
output.
The exception is, if ``<restart> True </restart>``, then the seeds, poly mesh,
and tri mesh will all be output to txt files.
The subsections below describe the options for each subfield.

rng_seeds
^^^^^^^^^

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
^^^^

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

mesh_max_volume
^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^

This field defines the minimum interior angle, measured in degrees, of any
element in the triangular mesh.
For 3D meshes, this is the minimum *dihedral* angle, which is between faces of
the tetrahedron.
This setting controls the aspect ratio of the elements, with angles between
15 and 30 degrees producing good quality meshes.
The default is ``<mesh_min_angle> 0 </mesh_min_angle>``, which effectively
turns off the angle quality control.

mesh_max_edge_length
^^^^^^^^^^^^^^^^^^^^

This field defines the maximum edge length along a grain boundary in a 2D
triangular mesh.
A small maximum edge length will increase resolution of the mesh at grain
boundaries.
Currently this feature has no equivalent in 3D.
The default value is ``<mesh_max_edge_length> inf </mesh_max_edge_length>``,
which effectively turns off the edge length quality control.

verify
^^^^^^

The verify flag will perform mesh verification on the triangular mesh and
report error metrics.
To include mesh verification, include ``<verify> True </verify>`` in the
settings.
The default behavior is to not perform mesh verification.

plot_axes
^^^^^^^^^

The plot_axes flag toggles the axes on or off in the output plots.
Setting it to False turns the axes off, producing images with miniminal
borders.
The default setting is ``<plot_axes> True </plot_axes>``, which includes the
coordinate axes in output plots.

color_by
^^^^^^^^

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
^^^^^^^^

The colormap field is used when ``color_by`` is set to either "seed number" or
"material number".
This gives the name of the colormap to be used in coloring the seeds/grains.
For a complete list of available colormaps, visit the `Choosing Colormaps in
Matplotlib`_ page.

seeds_kwargs
^^^^^^^^^^^^

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
^^^^^^^^^^^

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
^^^^^^^^^^

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
.. _`SciPy's statistical functions`: https://docs.scipy.org/doc/scipy/reference/stats.html
.. _`Specifying Colors`: https://matplotlib.org/3.1.0/tutorials/colors/colors.html
