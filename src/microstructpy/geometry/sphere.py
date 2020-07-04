"""Sphere

This module contains the Sphere class.
"""
# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #

from __future__ import division

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from microstructpy.geometry.n_sphere import NSphere

__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# Sphere Class                                                                #
#                                                                             #
# --------------------------------------------------------------------------- #
class Sphere(NSphere):
    """A 3D sphere.

    This class represents a three-dimensional circle. It is defined by
    a center point and size parameter, which can be either radius or diameter.

    Without input parameters, this defaults to a unit sphere centered at
    the origin.

    Args:
        r (float): *(optional)* The radius of the sphere.
            Defaults to 1.
        radius (float): *(optional)* Same as ``r``.
        d (float): *(optional)* Alias for ``2*r``.
        diameter (float): *(optional)* Alias for ``2*r``.
        size (float): *(optional)* Alias for ``2*r``.
        center (list, float, numpy.ndarray): *(optional)* The coordinates of
            the center. Defaults to ``[0, 0, 0]``.
        position (list, float, numpy.ndarray): *(optional)*
            Alias for ``center``.
    """
    # ----------------------------------------------------------------------- #
    # Constructor                                                             #
    # ----------------------------------------------------------------------- #
    def __init__(self, **kwargs):
        if 'volume' in kwargs:
            v = kwargs['volume']
            r = np.cbrt(3 * v / (4 * np.pi))
            kwargs['r'] = r

        NSphere.__init__(self, **kwargs)
        if len(self.center) == 0:
            self.center = tuple(self.n_dim * [0])

    # ----------------------------------------------------------------------- #
    # Representation Function                                                 #
    # ----------------------------------------------------------------------- #
    def __repr__(self):
        repr_str = 'Sphere('
        repr_str += 'center=' + repr(tuple(self.center))
        repr_str += ', radius=' + repr(self.r)
        repr_str += ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Number of Dimensions                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def n_dim(self):
        """int: number of dimensions, 3"""
        return 3

    # ----------------------------------------------------------------------- #
    # Volume                                                                  #
    # ----------------------------------------------------------------------- #
    @property
    def volume(self):
        """float: volume of sphere"""
        return 4 * np.pi * self.r * self.r * self.r / 3

    @classmethod
    def volume_expectation(cls, **kwargs):
        r"""Expected value of volume.

        This function computes the expected value for the volume of a sphere.
        The keyword arguments are identical to the :class:`.Sphere` function.
        The values for these keywords can be either constants or
        :mod:`scipy.stats` distributions.

        The expected value is computed by the following formula:

        .. math::

            \mathbb{E}[V] &= \mathbb{E}[\frac{4}{3}\pi R^3] \\
                            &= \frac{4}{3}\pi \mathbb{E}[R^3] \\
                            &= \frac{4}{3}\pi (\mu_R^3 + 3 \mu_R \sigma_R^2 + \gamma_{1, R} \sigma_R^3)

        Args:
            **kwargs: Keyword arguments, see :class:`microstructpy.geometry.Sphere`.

        Returns:
            float: Expected value of the volume of the sphere.

        """  # NOQA: E501
        # Check for radius distribution
        r_dist = None
        if 'radius' in kwargs:
            r_dist = kwargs['radius']
        elif 'r' in kwargs:
            r_dist = kwargs['r']

        if type(r_dist) in (float, int):
            return 4 * np.pi * r_dist * r_dist * r_dist / 3
        elif r_dist is not None:
            return 4 * np.pi * r_dist.moment(3) / 3

        # Check for diameter distribution
        d_dist = None
        for d_kw in ('d', 'diameter', 'size'):
            if d_kw in kwargs:
                d_dist = kwargs[d_kw]
                break

        if type(d_dist) in (float, int):
            return 0.5 * np.pi * d_dist * d_dist * d_dist / 3
        elif d_dist is not None:
            return 0.5 * np.pi * d_dist.moment(3) / 3

        if 'volume' in kwargs:
            v_dist = kwargs['volume']
            try:
                v_exp = v_dist.moment(1)
            except AttributeError:
                v_exp = v_dist

            return v_exp

        # Raise error
        e_str = 'Could not find one of the following keywords in the inputs: '
        e_str += 'r, radius, d, diameter, volume.'
        raise KeyError(e_str)

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, **kwargs):
        """Plot the sphere.

        This function uses the :meth:`mpl_toolkits.mplot3d.Axes3D.plot_surface`
        method to add the sphere to the current axes. The keyword arguments
        are passed through to plot_surface.

        Args:
            **kwargs (dict): Keyword arguments for plot_surface.

        """  # NOQA: E501
        if len(plt.gcf().axes) == 0:
            ax = plt.axes(projection=Axes3D.name)
        else:
            ax = plt.gca()

        u = np.linspace(0, 2 * np.pi, 11)
        cv = np.linspace(-1, 1, 12)
        uu, cvv = np.meshgrid(u, cv)
        svv = np.sin(np.arccos(cvv))

        xc, yc, zc = self.center
        r = self.r

        xx = xc + r * np.cos(uu) * svv
        yy = yc + r * np.sin(uu) * svv
        zz = zc + r * cvv

        mod_kwargs = {}
        for key, val in kwargs.items():
            if key == 'facecolors' and type(val) != list:
                mod_kwargs['color'] = val
            else:
                mod_kwargs[key] = val
        ax.plot_surface(xx, yy, zz, **mod_kwargs)
