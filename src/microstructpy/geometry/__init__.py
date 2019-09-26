from microstructpy.geometry.box import Box
from microstructpy.geometry.box import Cube
from microstructpy.geometry.circle import Circle
from microstructpy.geometry.ellipse import Ellipse
from microstructpy.geometry.ellipsoid import Ellipsoid
from microstructpy.geometry.rectangle import Rectangle
from microstructpy.geometry.rectangle import Square
from microstructpy.geometry.sphere import Sphere

__all__ = ['Box', 'Cube', 'Circle', 'Ellipse', 'Ellipsoid',
           'Rectangle', 'Square', 'Sphere']


def factory(name, **kwargs):
    """Factory method for geometries.

    This function returns a geometry based on a string containing the
    name of the geometry and keyword arguments defining the geometry.

    .. note::

        The function call is ``factory(name, **kwargs)``. Sphinx autodocs
        has dropped the first parameter.

    Args:
        name (str): {'box' | 'cube' | 'ellipse' | 'ellipsoid' | 'circle' | 
            'rectangle' | 'square' | 'sphere'} Name of geometry.
        **kwargs (dict): Arguments defining the geometry.

    """
    geom = name.strip().lower()
    if geom in ('box', 'rectangular prism', 'cuboid'):
        return Box(**kwargs)
    elif geom == 'cube':
        return Cube(**kwargs)
    elif geom == 'ellipse':
        return Ellipse(**kwargs)
    elif geom == 'ellipsoid':
        return Ellipsoid(**kwargs)
    elif geom == 'circle':
        return Circle(**kwargs)
    elif geom == 'rectangle':
        return Rectangle(**kwargs)
    elif geom == 'square':
        return Square(**kwargs)
    elif geom == 'sphere':
        return Sphere(**kwargs)
    else:
        e_str = 'Cannot recognize geometry name: ' + geom
        raise ValueError(e_str)
