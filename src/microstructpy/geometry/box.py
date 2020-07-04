"""Box

This module contains the Box class.

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from microstructpy.geometry.n_box import NBox

__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# Box Class                                                                   #
#                                                                             #
# --------------------------------------------------------------------------- #
class Box(NBox):
    """Box

    This class contains a generic, 3D box. The position and dimensions of the
    box can be specified using any of the parameters below.

    Without any parameters, this is a unit cube centered on the origin.

    Args:
        side_lengths (list): *(optional)* Side lengths.
        center (list): *(optional)* Center of box.
        corner (list): *(optional)* Bottom-left corner.
        limits (list): *(optional)* Bounds of box.
        bounds (list): *(optional)* Alias for ``limits``.

    """
    def __init__(self, **kwargs):
        NBox.__init__(self, **kwargs)

        try:
            self.center
        except AttributeError:
            self.center = [0, 0, 0]

        try:
            self.side_lengths
        except AttributeError:
            self.side_lengths = [1, 1, 1]

    # ----------------------------------------------------------------------- #
    # Representation Function                                                 #
    # ----------------------------------------------------------------------- #
    def __repr__(self):
        repr_str = 'Box('
        repr_str += 'center=' + repr(tuple(self.center)) + ', '
        repr_str += 'side_lengths=' + repr(tuple(self.side_lengths)) + ')'
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
        """float: volume of box, :math:`V=l_1 l_2 l_3`"""
        return self.n_vol

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, **kwargs):
        """Plot the box.

        This function adds an
        :class:`mpl_toolkits.mplot3d.art3d.Poly3DCollection` to the current
        axes. The keyword arguments are passed through to the Poly3DCollection.

        Args:
            **kwargs (dict): Keyword arguments for Poly3DCollection.

        """  # NOQA: E501
        if len(plt.gcf().axes) == 0:
            ax = plt.axes(projection=Axes3D.name)
        else:
            ax = plt.gca()

        xlim, ylim, zlim = self.limits

        inds = [(0, 0), (0, 1), (1, 1), (1, 0)]

        # x faces
        f1 = np.array([(xlim[0], ylim[i], zlim[j]) for i, j in inds])
        f2 = np.array([(xlim[1], ylim[i], zlim[j]) for i, j in inds])

        # y faces
        f3 = np.array([(xlim[i], ylim[0], zlim[j]) for i, j in inds])
        f4 = np.array([(xlim[i], ylim[1], zlim[j]) for i, j in inds])

        # z faces
        f5 = np.array([(xlim[i], ylim[j], zlim[0]) for i, j in inds])
        f6 = np.array([(xlim[i], ylim[j], zlim[1]) for i, j in inds])

        # plot
        xy = [f1, f2, f3, f4, f5, f6]
        pc = Poly3DCollection(xy, **kwargs)
        ax.add_collection(pc)


# --------------------------------------------------------------------------- #
#                                                                             #
# Cube Class                                                                  #
#                                                                             #
# --------------------------------------------------------------------------- #
class Cube(Box):
    """A cube.

    This class contains a generic, 3D cube. It is derived from the
    :class:`.Box` and contains the ``side_length`` property, rather than
    multiple side lengths.

    Without any parameters, this is a unit cube centered on the origin.

    Args:
        side_length (float): *(optional)* Side length.
        center (list, tuple, numpy.ndarray): *(optional)* Center of box.
        corner (list, tuple, numpy.ndarray): *(optional)* Bottom-left corner.
    """
    def __init__(self, **kwargs):
        if 'side_length' in kwargs:
            kwargs['side_lengths'] = 3 * [kwargs['side_length']]

        Box.__init__(self, **kwargs)

    # ----------------------------------------------------------------------- #
    # Side Length Property                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def side_length(self):
        """float: length of the side of the cube."""
        return self.side_lengths[0]
