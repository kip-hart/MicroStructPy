"""N-Dimensional Box

This module contains the NBox class.

"""
# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #

from __future__ import division

import numpy as np

__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# NBox Class                                                                  #
#                                                                             #
# --------------------------------------------------------------------------- #
class NBox(object):
    """N-dimensional box

    This class contains a generic, n-dimensinoal box.

    Args:
        side_lengths (list): *(optional)* Side lengths.
        center (list): *(optional)* Center of box.
        corner (list): *(optional)* Bottom-left corner.
        bounds (list): *(optional)* Bounds of box. Expected in the form
            [(xmin, xmax), (ymin, ymax), ...].
        limits : Alias for *bounds*.
        matrix (list, numpy.ndarray): *(optional)* Rotation matrix, nxn
    """

    def __init__(self, **kwargs):
        if 'bounds' in kwargs:
            lims = kwargs['bounds']
        elif 'limits' in kwargs:
            lims = kwargs['limits']

        if ('bounds' in kwargs) or ('limits' in kwargs):
            self.center = [0.5 * (lb + ub) for lb, ub in lims]
            self.side_lengths = [ub - lb for lb, ub in lims]

        elif ('side_lengths' in kwargs) and ('center' in kwargs):
            self.center = kwargs['center']
            self.side_lengths = kwargs['side_lengths']

        elif ('side_lengths' in kwargs) and ('corner' in kwargs):
            corner = kwargs['corner']
            side_lens = kwargs['side_lengths']
            cen = [xc + 0.5 * sl for xc, sl in zip(corner, side_lens)]
            self.center = cen
            self.side_lengths = side_lens

        elif ('center' in kwargs) and ('corner' in kwargs):
            corner = kwargs['corner']
            center = kwargs['center']
            side_lens = [2 * abs(x1 - x2) for x1, x2 in zip(center, corner)]
            self.center = center
            self.side_lengths = side_lens

        elif 'center' in kwargs:
            self.center = kwargs['center']
            self.side_lengths = [1 for _ in self.center]

        elif 'side_lengths' in kwargs:
            self.side_lengths = kwargs['side_lengths']
            self.center = [0 for _ in self.side_lengths]

        elif 'corner' in kwargs:
            self.side_lengths = [1 for _ in kwargs['corner']]
            self.center = [xc + 0.5 for xc in kwargs['corner']]

        if 'matrix' in kwargs:
            self.matrix = kwargs['matrix']
        else:
            try:
                self.matrix = np.eye(self.n_dim)
            except AttributeError:
                self.matrix = np.eye(len(self.side_lengths))

    # ----------------------------------------------------------------------- #
    # String and Representation Functions                                     #
    # ----------------------------------------------------------------------- #
    def __str__(self):
        cen = np.array(self.center)
        sides = np.array(self.side_lengths)
        cen_str = np.array2string(cen, separator=', ')
        sides_str = np.array2string(sides, separator=', ')

        str_str = 'Center: ' + cen_str + '\n'
        str_str += 'Side Lengths: ' + sides_str + '\n'
        str_str += 'Matrix: ('
        for row in self.matrix:
            str_str += '('
            str_str += ', '.join([str(val) for val in row])
            str_str += '),'
        str_str = str_str[:-1] + ')'
        return str_str

    def __repr__(self):
        repr_str = 'NBox('
        repr_str += 'center=' + repr(tuple(self.center)) + ', '
        repr_str += 'side_lengths=' + repr(tuple(self.side_lengths)) + ', '
        repr_str += 'matrix='
        repr_str += repr(tuple([tuple(r) for r in self.matrix])) + ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Dimension Getters                                                       #
    # ----------------------------------------------------------------------- #
    @property
    def corner(self):
        """list: bottom-left corner"""
        c_rel = -0.5 * np.array(self.matrix).dot(self.side_lengths)
        return np.array(self.center) + c_rel

    @property
    def limits(self):
        """list: (lower, upper) bounds of the box"""
        cen = self.center
        side_lens = self.side_lengths

        n_dim = len(cen)
        powers = np.power(2, np.arange(n_dim))
        n_pts = np.power(2, n_dim)
        pts = np.zeros((n_pts, n_dim))
        for i in range(n_pts):
            for j in range(n_dim):
                ind = (i // powers[j]) % 2
                pts[i, j] = (ind - 0.5) * side_lens[j]

        pts = np.array(cen) + pts.dot(np.array(self.matrix).T)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        return np.array([mins, maxs]).T

    @property
    def bounds(self):
        """list: (lower, upper) bounds of the box"""
        return self.limits

    # ----------------------------------------------------------------------- #
    # Sample Limits                                                           #
    # ----------------------------------------------------------------------- #
    @property
    def sample_limits(self):
        """ list: (lower, upper) bounds of the sampling region of the box"""
        cen = self.center
        lens = self.side_lengths
        tol = 1e-4 * max(lens)
        return [(x - 0.5 * s + tol, x + 0.5 * s - tol) for x, s in
                zip(cen, lens)]

    # ----------------------------------------------------------------------- #
    # Area/Volume                                                             #
    # ----------------------------------------------------------------------- #
    @property
    def n_vol(self):
        """float: area, volume of n-box"""
        return np.prod(self.side_lengths)

    # ----------------------------------------------------------------------- #
    # Within Test                                                             #
    # ----------------------------------------------------------------------- #
    def within(self, points):
        """Test if points are within n-box.

        This function tests whether a point or set of points are within the
        n-box. For the set of points, a list of booleans is returned to
        indicate which points are within the n-box.

        Args:
            points (list or numpy.ndarray): Point or list of points.

        Returns:
            bool or numpy.ndarray: Flags set to True for points in geometry.
        """
        pts = np.array(points)
        single_pt = pts.ndim == 1
        if single_pt:
            pts = pts.reshape(1, -1)

        rel_pos = pts - np.array(self.center)
        min_dist = 0.5 * np.array(self.side_lengths)

        mask = np.all(np.abs(rel_pos) <= min_dist, axis=-1)
        if single_pt:
            return mask[0]
        else:
            return mask
