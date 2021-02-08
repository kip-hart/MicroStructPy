"""Seed

This module contains the class definition for the Seed class.
"""

# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #

from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import collections
from matplotlib import patches
from matplotlib import pyplot as plt

from microstructpy import _misc
from microstructpy import geometry

__all__ = ['Seed']
__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# Seed Class                                                                  #
#                                                                             #
# --------------------------------------------------------------------------- #
class Seed(object):
    """Seed particle

    The Seed class contains the information about a single seed in the mesh.
    These seeds have a geometry (from :mod:`microstructpy.geometry`),
    a phase number, a breakdown, and a position.

    Args:
        seed_geometry (from :mod:`microstructpy.geometry`) : The geometry of
            the seed.
        phase (int) : *(optional)* The phase number of the seed.
            Defaults to 0.
        breakdown (list or numpy.ndarray) : *(optional)* The circle/sphere
            approximation of this grain. The format for this input is::

                #                 x   y  r
                breakdown_2D = [( 2,  3, 1),
                                ( 0,  0, 4),
                                (-2,  4, 8)]

                #                 x   y   z  r
                breakdown_3D = [( 3, -1,  2, 1),
                                ( 0,  2, -1, 1)]

            The default behavior is to call the ``approximate()`` function
            of the geometry.
        position (list or numpy.ndarray) : *(optional)* The coordinates of the
            seed. See :attr:`position` for more details.
            Defaults to the origin.
    """
    # ----------------------------------------------------------------------- #
    # Initializer                                                             #
    # ----------------------------------------------------------------------- #
    def __init__(self, seed_geometry, phase=0, breakdown=None, position=None):
        self.geometry = seed_geometry
        self.phase = phase

        if position is None and self.geometry is not None:
            self.position = [0 for _ in range(self.geometry.n_dim)]
        else:
            self.position = position

        if breakdown is None:
            self.breakdown = seed_geometry.approximate()
        else:
            self.breakdown = breakdown

    # ----------------------------------------------------------------------- #
    # Factory Method                                                          #
    # ----------------------------------------------------------------------- #
    @classmethod
    def factory(cls, seed_type, phase=0, breakdown=None, position=None,
                **kwargs):
        """Factory method for seeds

        This function returns a seed based on the seed type and keyword
        arguments associated with that type. The currently supported types
        are:

            * circle
            * ellipse
            * ellipsoid
            * rectangle
            * sphere
            * square

        If the seed_type is not on this list, an error is thrown.

        Args:
            seed_type (str): type of seed, from list above.
            phase (int): *(optional)* Material phase number of seed.
                Defaults to 0.
            breakdown (list or numpy.ndarray): *(optional)* List of circles or
                spheres that approximate the geometry. The list should be
                formatted as follows::

                    breakdown = [(x1, y1, z1, r1),
                                 (x2, y2, z2, r2),
                                 ...]

                The breakdown will be automatically generated if not provided.
            position (list or numpy.ndarray): *(optional)* The coordinates of
                the seed. Default is the origin.
            **kwargs: Keyword arguments that define the size, shape, etc of the
                seed geometry.

        Returns:
            Seed: An instance of the class.
        """
        assert type(seed_type) is str
        seed_type = seed_type.strip().lower()

        if seed_type == 'nonetype':
            n_dim = 0
        else:
            n_dim = geometry.factory(seed_type).n_dim
        if 'volume' in kwargs:
            if n_dim == 2:
                size = 2 * np.sqrt(kwargs['volume'] / np.pi)
            else:
                size = 2 * np.cbrt(3 * kwargs['volume'] / (4 * np.pi))
            kwargs['size'] = size

        # Catch NoneType geometries
        if seed_type == 'nonetype':
            geom = None
        else:
            geom = geometry.factory(seed_type, **kwargs)

        if breakdown is None:
            if seed_type in ('circle', 'sphere'):
                breakdown = np.append(geom.center, geom.r).reshape(1, -1)
            else:
                breakdown = geom.approximate()

        if position is None:
            position = [0 for _ in range(geom.n_dim)]

        return cls(geom, phase, breakdown, position)

    # ----------------------------------------------------------------------- #
    # Read from String                                                        #
    # ----------------------------------------------------------------------- #
    @classmethod
    def from_str(cls, seed_str):
        """Create seed from a string.

        This method creates a seed particle from a string representation.
        This is used when reading in seeds from a file.

        Args:
            seed_str (str): String representation of the seed.

        Returns:
            Seed: An instance of a Seed derived class.

        """
        # Convert to dictionary
        str_dict = {}
        for line in seed_str.strip().split('\n'):
            try:
                k_str, v_str = line.split(':')
            except ValueError:
                continue
            else:
                k = k_str.strip().lower().replace(' ', '_')
                v = _misc.from_str(v_str)
                str_dict[k] = v

        # Extract seed type, phase, and breakdown
        seed_type = str_dict['geometry']
        del str_dict['geometry']

        if 'phase' in str_dict:
            phase = str_dict['phase']
            del str_dict['phase']
        else:
            phase = 0

        if 'breakdown' in str_dict:
            breakdown = str_dict['breakdown']
            if not isinstance(breakdown[0], tuple):
                breakdown = (breakdown,)
            del str_dict['breakdown']
        else:
            breakdown = None

        if 'position' in str_dict:
            position = str_dict['position']
            del str_dict['position']
        else:
            position = None

        return cls.factory(seed_type, phase, breakdown, position, **str_dict)

    # ----------------------------------------------------------------------- #
    # String and Representation Functions                                     #
    # ----------------------------------------------------------------------- #
    def __str__(self):
        geom_name = type(self.geometry).__name__.lower()
        str_str = 'Geometry: ' + geom_name + '\n'
        str_str += str(self.geometry) + '\n'
        str_str += 'Phase: ' + str(self.phase) + '\n'
        bkdwn_str = ', '.join([str(tuple(b)) for b in self.breakdown])
        if len(self.breakdown) == 1:
            bkdwn_str += ','  # breakdowns will be a tuple of length 1
        str_str += 'Breakdown: (' + bkdwn_str + ')\n'
        str_str += 'Position: (' + ', '.join([str(x) for x in self.position])
        str_str += ')'
        return str_str

    def __repr__(self):
        repr_str = 'Seed('
        repr_str += repr(self.geometry) + ', '
        repr_str += 'phase=' + repr(self.phase) + ', '
        bkdwn_str = ', '.join([repr(tuple(b)) for b in self.breakdown])
        repr_str += 'breakdown=(' + bkdwn_str + '), '
        repr_str += 'position=(' + ', '.join([repr(x) for x in self.position])
        repr_str += ')'
        repr_str += ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Comparison Functions                                                    #
    # ----------------------------------------------------------------------- #
    def __lt__(self, seed):
        if self.geometry is None:
            return True
        if seed.geometry is None:
            return False

        a_str = 'Seeds are not the same dimension.'
        assert self.geometry.n_dim == seed.geometry.n_dim, a_str
        if self.geometry.n_dim == 2:
            v_self = self.geometry.area
            v_seed = seed.geometry.area
        else:
            v_self = self.geometry.volume
            v_seed = seed.geometry.volume

        if v_self == v_seed:
            return self.phase < seed.phase
        else:
            return v_self < v_seed

    def __eq__(self, seed):
        if not isinstance(seed, Seed):
            return False

        if seed.phase != self.phase:
            return False

        if not np.all(np.isclose(seed.breakdown, self.breakdown)):
            return False

        if seed.geometry != self.geometry:
            return False

        if not np.all(np.isclose(seed.position, self.position)):
            return False

        return True

    # ----------------------------------------------------------------------- #
    # Position Getter/Setter                                                  #
    # ----------------------------------------------------------------------- #
    @property
    def position(self):
        """Position of the seed

        This is the location of the seed center.

        Note:
            If the breakdown of the seed has been populated, the setter
            function will update the position of the center and translate
            the breakdown circles/spheres.

        """
        return self._position

    @position.setter
    def position(self, pos):
        try:
            old_pos = np.array(self.position)
        except AttributeError:
            old_pos = np.zeros(len(pos))

        try:
            displace = np.array(pos) - old_pos
            for i, bkdwn in enumerate(self.breakdown):
                coords = bkdwn[:-1]
                rad = bkdwn[-1]
                new_coords = [x + d for x, d in zip(coords, displace)]
                new_bkdwn = new_coords + [rad]
                self.breakdown[i] = new_bkdwn
        except AttributeError:
            pass

        try:
            self.geometry.center = pos
        except AttributeError:
            pass

        self._position = pos

    # ----------------------------------------------------------------------- #
    # Volume                                                                  #
    # ----------------------------------------------------------------------- #
    @property
    def volume(self):
        """float: The area (2D) or volume (3D) of the seed"""
        if self.geometry is None:
            return 0
        if self.geometry.n_dim == 2:
            return self.geometry.area
        else:
            return self.geometry.volume

    # ----------------------------------------------------------------------- #
    # Limits                                                                  #
    # ----------------------------------------------------------------------- #
    @property
    def limits(self):
        """list: The (lower, upper) bounds of the seed"""
        if self.geometry is None:
            return []
        return self.geometry.limits

    # ----------------------------------------------------------------------- #
    # Plot                                                                    #
    # ----------------------------------------------------------------------- #
    def plot(self, **kwargs):
        """Plot the seed

        This function plots the geometry of the seed. The keyword arguments
        are passed through to matplotlib.
        See the plot methods in :mod:`microstructpy.geometry` for more
        details.

        Args:
            **kwargs: Plotting keyword arguments.

        """
        if self.geometry is not None:
            self.geometry.plot(**kwargs)

    # ----------------------------------------------------------------------- #
    # Plot Breakdown                                                          #
    # ----------------------------------------------------------------------- #
    def plot_breakdown(self, **kwargs):
        """Plot breakdown of seed

        This function plots the circle/sphere breakdown of the seed. In 2D,
        this adds a :class:`matplotlib.collections.PatchCollection`
        to the current axes.

        Args:
            **kwargs: Matplotlib keyword arguments.

        """

        n = len(self.breakdown[0]) - 1
        if n == 2:
            pc = [patches.Circle([x, y], r) for x, y, r in self.breakdown]
            coll = collections.PatchCollection(pc, **kwargs)
            plt.gca().add_collection(coll)

        else:
            [geometry.Sphere(r=r, center=(x, y, z)).plot(**kwargs)
             for x, y, z, r in self.breakdown]
