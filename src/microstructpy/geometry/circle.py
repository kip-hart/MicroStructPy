# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #
from __future__ import division

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from microstructpy.geometry.n_sphere import NSphere

__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# Circle Class                                                                #
#                                                                             #
# --------------------------------------------------------------------------- #
class Circle(NSphere):
    """A 2D circle.

    This class represents a two-dimensional circle. It is defined by
    a center point and size parameter, which can be either radius or diameter.

    Without parameters, this returns a unit circle centered on the origin.

    Args:
        r (float): *(optional)* The radius of the circle. Defaults to 1.
        center (list): *(optional)* The coordinates of the center.
            Defaults to (0, 0).
        diameter : *(optional)* Alias for ``2*r``.
        radius : *(optional)* Alias for ``r``.
        d : *(optional)* Alias for ``2*r``.
        size : *(optional)* Alias for ``2*r``.
        position : *(optional)* Alias for ``center``.
    """
    # ----------------------------------------------------------------------- #
    # Constructor                                                             #
    # ----------------------------------------------------------------------- #
    def __init__(self, **kwargs):
        if 'area' in kwargs:
            a = kwargs['area']
            r = np.sqrt(a / np.pi)
            kwargs['r'] = r

        NSphere.__init__(self, **kwargs)
        if len(self.center) == 0:
            self.center = tuple(self.n_dim * [0])

    # ----------------------------------------------------------------------- #
    # Representation Function                                                 #
    # ----------------------------------------------------------------------- #
    def __repr__(self):
        repr_str = 'Circle('
        repr_str += 'center=' + repr(tuple(self.center))
        repr_str += ', radius=' + repr(self.r)
        repr_str += ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Number of Dimensions                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def n_dim(self):
        """int: number of dimensions, 2"""
        return 2

    # ----------------------------------------------------------------------- #
    # Area                                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def area(self):
        r"""float: area of cirle, :math:`A=\pi r^2`"""
        return np.pi * self.r * self.r

    @property
    def volume(self):
        """float: alias for area"""
        return self.area

    # ----------------------------------------------------------------------- #
    # Expected Area                                                           #
    # ----------------------------------------------------------------------- #
    @classmethod
    def area_expectation(cls, **kwargs):
        r"""Expected value of area.

        This function computes the expected value for the area of a circle.
        The keyword arguments are the same as the class parameters.
        The values can be constants (ints or floats), or a distribution from
        the SciPy :mod:`scipy.stats` module.

        The expected value is computed by the following formula:

        .. math::

            \mathbb{E}[A] = \pi \mathbb{E}[R^2] = \pi (\mu_R^2 + \sigma_R^2)


        For example::

            >>> from microstructpy.geometry import Circle
            >>> Circle.area_expectation(r=1)
            3.141592653589793
            >>> from scipy.stats import norm
            >>> Circle.area_expectation(r=norm(1, 1))
            6.283185307179586

        Args:
            **kwargs: Keyword arguments, see :class:`.Circle`.

        Returns:
            float: Expected value of the area of the circle.

        """  # NOQA: E501
        # Check for radius distribution
        r_dist = None
        if 'radius' in kwargs:
            r_dist = kwargs['radius']
        elif 'r' in kwargs:
            r_dist = kwargs['r']

        try:
            return np.pi * r_dist.moment(2)
        except AttributeError:
            try:
                return np.pi * r_dist * r_dist
            except TypeError:
                pass

        # Check for diameter distribution
        d_dist = None
        for d_kw in ('d', 'diameter', 'size'):
            if d_kw in kwargs:
                d_dist = kwargs[d_kw]
                break

        try:
            return 0.25 * np.pi * d_dist.moment(2)
        except AttributeError:
            try:
                return 0.25 * np.pi * d_dist * d_dist
            except TypeError:
                pass

        if 'area' in kwargs:
            a_dist = kwargs['area']
            try:
                a_exp = a_dist.moment(1)
            except AttributeError:
                a_exp = a_dist
            return a_exp

        # Raise error
        e_str = 'Could not find one of the following keywords in the inputs: '
        e_str += 'r, radius, d, diameter, size.'
        raise KeyError(e_str)

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, **kwargs):
        """Plot the circle.

        This function adds a :class:`matplotlib.patches.Circle` to the
        current axes. The keyword arguments are passed through to the
        circle patch.

        Args:
            **kwargs (dict): Keyword arguments for matplotlib.

        """  # NOQA: E501
        c = patches.Circle(self.center, self.r, **kwargs)
        plt.gca().add_patch(c)
