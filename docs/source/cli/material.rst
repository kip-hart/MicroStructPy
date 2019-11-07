================================
``<material>`` - Material Phases
================================

Single Material
---------------

MicroStructPy supports an arbitrary number of materials, including just one.
For example:

.. literalinclude:: ../../../src/microstructpy/examples/minimal.xml
    :language: xml

This input file will produce a microstructure with nearly circular grains
of size 0.15 (within a domain with side length 1).


Multiple Materials
------------------

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

Volume fractions can also be distributed quantities, rather than fixed values.
This is useful if measured volume fractions have some uncertainty.
For example:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> circle </shape>
            <size> 1 </size>
            <fraction>
                <dist_type> norm </dist_type>
                <loc> 0.7 </loc>
                <scale> 0.03 </scale>
            </fraction>
        </material>

        <material>
            <shape> circle </shape>
            <size> 0.5 </size>
            <fraction>
                <dist_type> norm </dist_type>
                <loc> 0.3 </loc>
                <scale> 0.03 </scale>
            </fraction>
        </material>
    </input>

Here the standard deviation on the volume fractions is 0.03, meaning that
volume fractions are accurate to within 6 percentage points at 95%
confidence.

.. note::

    If the volume fraction distribution has negative numbers in the support,
    MicroStructPy will re-sample the distribution until a non-negative volume
    fraction is sampled.

Grain Size Distributions
------------------------

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
This type can match the name of a statistical distribution in the SciPy
:mod:`scipy.stats` module, or be either "pdf" or "cdf".
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
----------------

MicroStructPy supports several grain geometries and each can be specified in
multiple ways.
For example, the ellipse can be specified in terms of its area and aspect
ratio, or by its semi-major and semi-minor axes.
The 'size' of a grain is defined as the diameter of a circle or sphere with
equivalent area (so for a general ellipse, this would be :math:`2\sqrt{a b}`).
The parameters available for each geometry are described in the lists below.

Circle
++++++

**Class**: :class:`microstructpy.geometry.Circle`

**Parameters**

- ``area`` - the area of the circle
- ``d`` - alias for ``diameter``
- ``diameter`` - the diameter of the circle
- ``r`` - alias for ``radius``
- ``radius`` - the radius of the circle
- ``size`` - same as ``diameter``

Only one of the above is necessary to define the circle geometry.
If no parameters are specified, the default is a unit circle.

Ellipse
+++++++

**Class**: :class:`microstructpy.geometry.Ellipse`

**Parameters**

- ``a`` - the semi-major axis of the ellipse
- ``angle`` - alias for ``angle_deg``
- ``angle_deg`` - the counterclockwise positive angle between the semi-major
  axis and the +x axis, measured in degrees
- ``angle_rad`` - the counterclockwise positive angle between the semi-major
  axis and the +x axis, measured in radians
- ``aspect_ratio`` - the ratio a/b
- ``axes`` - semi-axes of ellipse, equivalent to [a, b]
- ``b`` - the semi-minor axis of the ellipse
- ``matrix`` - orientation matrix for the ellipse
- ``orientation`` - alias for ``matrix``
- ``size`` - the diameter of a circle with the same area as the ellipse

Two shape parameters and one orientation parameter are necessary to fully
define the ellipse geometry.
If less than two shape parameters are given, the default is a unit circle.
If an orientation parameter is not given, the default is aligned with the
coordinate axes.

.. note::

    The default orientation of an ellipse is aligned with the coordinate
    axes.
    Uniform random orientation can be achieved by setting
    ``<orientation> random </orientation>`` in the input file.
    

Ellipsoid
+++++++++

**Class**: :class:`microstructpy.geometry.Ellipsoid`

**Parameters**

- ``a`` - first semi-axis of the ellipsoid
- ``axes`` - semi-axes of the ellipsoids, equivalent to [a, b, c]
- ``b`` - second semi-axis of the ellipsoid
- ``c`` - third semi-axis of the ellipsoid
- ``matrix`` - orientation matrix for the ellipsoid
- ``orientation`` - alias for ``matrix``
- ``ratio_ab`` - the ratio a/b
- ``ratio_ac`` - the ratio a/c
- ``ratio_ba`` - the ratio b/a
- ``ratio_bc`` - the ratio b/c
- ``ratio_ca`` - the ratio c/a
- ``ratio_cb`` - the ratio c/b
- ``rot_seq`` - alias for ``rot_set_seq``
- ``rot_seq_deg`` - a rotation sequence, with angles in degrees, to define
  the orientation of the ellipsoid. See below for details.
- ``rot_seq_rad`` - a rotation sequence, with angles in radians, to define
  the orientation of the ellipsoid. See below for details.
- ``size`` - the diameter of a sphere with the same volume as the ellipsoid

Three shape parameters and one orientation parameter are necessary to fully
define the ellipsoid geometry.
If the length of a semi-axis cannot be determined from the input parameters,
it defaults to unit length.
If an orientation parameter is not given, the default is aligned with the
coordinate axes.

A rotation sequence is a list of axes and angles to rotate the ellipsoid.
The order of the rotations is as such: supposing after a z-rotation that
the new x and y axes are x' and y', a subsequent y-rotation would be about the
y' axis.
An example rotation sequence is:

.. code-block:: xml

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> ellipsoid </shape>
            <axes> 5, 3, 1 </axes>

            <rot_seq_deg>
                <axis> z </axis>
                <angle> 10 </angle>
            </rot_seq_deg>

            <rot_seq_deg>
                <axis> x </axis>
                <angle>
                    <dist_type> uniform </dist_type>
                    <loc> 30 </loc>
                    <scale> 30 </scale>
                </angle>
            </rot_seq_deg>

            <rot_seq_deg>
                <axis> y </axis>
                <angle>
                    <dist_type> norm </dist_type>
                    <loc> 0 </loc>
                    <scale> 30 </scale>
                </angle>
            </rot_seq_deg>
        </material>
    </input>

This represents first a z-rotation of 10 degrees, then an x-rotation of 30-60
degrees, then finally a y-rotation of :math:`N(0, 30)` degrees.

.. note::

    Ellipsoids with uniform random distribution will be generated using
    ``<orientation> random </orientation>``.
    Positions on the unit 4-sphere are generated with a uniform random
    distribution, then converted into a quaternion and finally into a rotation
    matrix.

Rectangle
+++++++++

**Class**: :class:`microstructpy.geometry.Ellipsoid`

**Parameters**

- ``angle``- alias for ``angle_deg``
- ``angle_deg`` - rotation angle, in degrees, measured counterclockwise from
  the +x axis
- ``angle_rad`` - rotation angle, in radians, measured counterclockwise from
  the +x axis
- ``length`` - the x-direction side length of the rectangle
- ``matrix`` - the orientation matrix of the rectangle
- ``side_lengths`` - equivalent to [length, width]
- ``width`` - the y-direction side length of the rectangle

Both the length and the width of the rectangle must be specified.
If either is not specified, the default rectangle is a sqaure with unit
side length.
If an orientation is not specified, the default is aligned with the coordinate
axes.

Sphere
++++++

**Class**: :class:`microstructpy.geometry.Sphere`

**Parameters**

- ``d`` - alias for ``diameter``
- ``diameter`` - the diameter of the sphere
- ``r`` - alias for ``radius``
- ``radius`` - the radius of the sphere
- ``size`` - alias for ``diameter``
- ``volume`` - volume of the sphere

Only one of the above is necessary to define the sphere geometry.
If no parameters are specified, the default is a unit sphere.

Square
++++++

**Class**: :class:`microstructpy.geometry.Square`

**Parameters**

- ``angle`` - alias for ``angle_deg``
- ``angle_deg`` - the rotation angle, in degrees, of the square measured
  counterclockwise from the +x axis
- ``angle_rad`` - the rotation angle, in radians, of the square measured
  counterclockwise from the +x axis
- ``matrix`` - the orientation matrix of the square
- ``side_length`` - the side lnegth of the square

If the side length of the square is not specified, the default is 1.
If an orientation parameter is not specified, the default orientation is
aligned with the coordinate axes.

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
-------------

There are three types of materials supported by MicroStructPy: crystalline,
amorphous, and void.
For crystalline phases, facets between cells of the same grain are removed
before unstructured meshing.
For amorphous phases, facets between cells of the same phase are removed
before meshing.
Finall, void phases produce empty spaces in the unstructured mesh.
There are several synonyms for these material types, including:

* **crystalline**: granular, solid
* **amorphous**: glass, matrix
* **void**: crack, hole

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
A material can contain multiple amorphous and void phases.

.. note::
    
    Void phases may cause parts of the mesh to become disconnected.
    MicroStructPy does not check for or remove disconnected regions from the
    mesh.


Grain Position Distribution
---------------------------

The default position distribution for grains is random uniform throughout the
domain.
Grains can be non-uniformly distributed by adding a position distribution.
The x, y, and z can be independently distributed or coupled.
The coupled distributions can be any of the multivariate distributions listed
in the SciPy :mod:`scipy.stats` module.


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
-----------------------

**Name** The name of each material can be specified by adding a "name" field.
The default name is "Material N" where N is the order of the material in
the XML file, starting from 0.

**Color** The color of each material in output plots can be specified by adding
a "color" field.
The default color is "CN" where N is the order of the material in the XML file,
starting from 0.
For more information about color specification, visit the Matplotlib
`Specifying Colors`_ webpage.

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


.. _`Specifying Colors`: https://matplotlib.org/3.1.0/tutorials/colors/colors.html
