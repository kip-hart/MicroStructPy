====================================
``<domain>`` - Microstructure Domain
====================================

MicroStructPy supports the following domain geometries:

* **2D**: circle, ellipse, rectangle, square
* **3D**: box, cube

Each geometry can be defined several ways, such as a center and edge lengths
for the rectangle or two bounding points.
Note that over-parameterizing the domain geometry will cause unexpected
behavior.

Box
^^^

**Class**: :class:`microstructpy.geometry.Box`

**Parameters** 

- ``bounds`` - alias for ``limits``
- ``center`` - the center of the box
- ``corner`` - the bottom-most corner of the box (i.e. :math:`(x, y, z)_{min}`)
- ``limits`` - the x, y, z upper and lower bounds of the box
  (i.e. :math:`[[x_{min}, x_{max}], [y_{min}, y_{max}], [z_{min}, z_{max}]]`)
- ``side_lengths`` - the x, y, and z side lengths of the box

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

**Class**: :class:`microstructpy.geometry.Circle`

**Parameters**

- ``area`` - the area of the circle
- ``center`` - the center of the circle
- ``d`` - alias for ``diameter``
- ``diameter`` - the diameter of the circle
- ``r`` - alias for ``radius``
- ``radius`` - the radius of the circle
- ``size`` - same as ``diameter``

The default radius of a circle is 1, while the default center is (0, 0).

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

**Class**: :class:`microstructpy.geometry.Cube`

**Parameters**

- ``center`` - the center of the cube
- ``corner`` - the bottom-most corner of the cube
  (i.e. :math:`(x, y, z)_{min}`)
- ``side_length`` - the side length of the cube

The defaultt side length of the cube is 1, while the default center is
(0, 0).

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
- ``center`` - the center of the ellipse
- ``matrix`` - orientation matrix for the ellipse
- ``orientation`` - alias for ``matrix``
- ``size`` - the diameter of a circle with the same area as the ellipse

The default value for the semi-axes of the ellipse is 1.
The default orientation of the ellipse is aligned with the coordinate axes.
Finally, the default position of the ellipse is centered at (0, 0).

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

**Class**: :class:`microstructpy.geometry.Rectangle`

**Parameters**

- ``bounds`` - alias for ``limits``
- ``center`` - the center of the rectangle
- ``corner`` - the bottom-most corner of the rectangle
  (i.e. :math:`(x, y)_{min}`)
- ``length`` - the x-direction side length of the rectangle
- ``limits`` - the x and y upper and lower bounds of the rectangle
  (i.e. :math:`[[x_{min}, x_{max}], [y_{min}, y_{max}]]`)
- ``side_lengths`` - equivalent to [length, width]
- ``width`` - the y-direction side length of the rectangle

The default side lengths of the rectangle are 1, while the default position is
centered at the origin.

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

**Class**: :class:`microstructpy.geometry.Square`

**Parameters**

- ``side_length`` - the side length of the square
- ``center`` - the position of the center of the square
- ``corner`` - the bottom-most corner of the square
  (i.e. :math:`(x, y)_{min}`)

The default side length of a square is 1, while the default center position is
(0, 0).

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

Periodicity
^^^^^^^^^^^

Rectangular domains (rectangle, square, box, cube) can be made periodic with
the ``<periodic>`` field, which lists the periodic axes. Seeds that cross a
periodic face are placed so that they do not overlap seeds on the opposite
side, the tessellation is periodic across those faces (a grain that crosses a
face continues on the opposite side), and the triangular mesh has matching
nodes on opposite faces. The pairs of periodic nodes are written with the
meshes, and the Abaqus output contains a node set per periodic face, in
matching order.

.. code-block:: XML

    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Example periodic domains -->
    <input>
        <domain>
            <shape> square </shape>
            <side_length> 10 </side_length>
            <!-- periodic in both directions -->
            <periodic> True </periodic>
        </domain>

        <domain>
            <shape> rectangle </shape>
            <side_lengths> 10, 5 </side_lengths>
            <!-- periodic in x only -->
            <periodic> x </periodic>
        </domain>

        <domain>
            <shape> cube </shape>
            <side_length> 10 </side_length>
            <!-- periodic in x and z, free in y -->
            <periodic> xz </periodic>
        </domain>
    </input>

The mesher must be Triangle/TetGen for periodic microstructures (with gmsh the
nodes on opposite faces are not guaranteed to match), and the mesh size of a
raster mesh must divide the domain length along the periodic axes.
The mesh quality and size settings (``mesh_min_angle``, ``mesh_max_volume``,
the ``max_volume`` of each phase and ``mesh_max_edge_length``) apply to
periodic meshes as to non-periodic ones. In 2D, the cells next to the
periodic faces are copied outside the faces while meshing, so that Triangle
refines both faces of a pair the same way, and the mesh is built again with
the points it added on the faces when some nodes have no image. In 3D, the
mesh is built once as usual, then the facets are triangulated with the points
TetGen added on them (the facets on opposite periodic faces with the points
of both) and the mesh is built again with the facets fixed; the elements next
to the periodic faces are slightly more numerous and of slightly lower
quality than in a non-periodic mesh.
