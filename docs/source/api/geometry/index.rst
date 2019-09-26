microstructpy.geometry
======================

.. automodule:: microstructpy.geometry

The geometry module contains classes for several 2D and 3D geometries.
The module also contains some N-D geometries, which are inherited by the 
2D and 3D geometries.

**2D Geometries**

* :ref:`api_geometry_circle` † ‡
* :ref:`api_geometry_ellipse` † ‡
* :ref:`api_geometry_rectangle` † ‡
* :ref:`api_geometry_square` † ‡

**3D Geometries**

* :ref:`api_geometry_box` ‡
* :ref:`api_geometry_cube` ‡
* :ref:`api_geometry_ellipsoid` †
* :ref:`api_geometry_sphere` †

**ND Geometries**

* :ref:`api_geometry_n_box`
* :ref:`api_geometry_n_sphere`

†: These classes may be used to define seed particles.

‡: These classes may be used to define the microstructure domain.

To assist with creating geometries, a factory method is included in the module:

* :ref:`api_geometry_factory`

.. only:: html

    **Module Contents**

.. toctree::

    box
    circle
    cube
    ellipse
    ellipsoid
    n_box
    n_sphere
    rectangle
    sphere
    square
    factory
