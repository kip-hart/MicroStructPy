# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from microstructpy.geometry.n_box import NBox

__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# Rectangle Class                                                             #
#                                                                             #
# --------------------------------------------------------------------------- #
class Rectangle(NBox):
    """Rectangle

    This class contains a generic, 2D rectangle. The position and dimensions
    of the box can be specified using any of the parameters below.

    Args:
        length (float, optional): Length of the rectangle.
        width (float, optional): Width of the rectangle.
        side_lengths (list, optional): Side lengths. Defaults to (1, 1).
        center (list, optional): Center of rectangle. Defaults to (0, 0).
        corner (list, optional): bottom-left corner.
        bounds (list, optional): Bounds of rectangle. Expected to be in the
            format [(xmin, xmax), (ymin, ymax)].
        limits : Alias for *bounds*.
        angle (float, optional): The rotation angle, in degrees.
        angle_deg (float, optional): The rotation angle, in degrees.
        angle_rad (float, optional): The rotation angle, in radians.
        matrix (2x2 array, optional): The rotation matrix.
    """

    def __init__(self, **kwargs):
        if 'length' in kwargs and 'width' in kwargs:
            kwargs['side_lengths'] = [kwargs['length'], kwargs['width']]

        if 'angle' in kwargs:
            cp = np.cos(np.radians(kwargs['angle']))
            sp = np.sin(np.radians(kwargs['angle']))
            kwargs['matrix'] = np.array([[cp, -sp], [sp, cp]])
        elif 'angle_deg' in kwargs:
            cp = np.cos(np.radians(kwargs['angle_deg']))
            sp = np.sin(np.radians(kwargs['angle_deg']))
            kwargs['matrix'] = np.array([[cp, -sp], [sp, cp]])
        elif 'angle_rad' in kwargs:
            cp = np.cos(kwargs['angle_rad'])
            sp = np.sin(kwargs['angle_rad'])
            kwargs['matrix'] = np.array([[cp, -sp], [sp, cp]])

        NBox.__init__(self, **kwargs)
        try:
            self.center
        except AttributeError:
            self.center = [0, 0]

        try:
            self.side_lengths
        except AttributeError:
            self.side_lengths = [1, 1]

    # ----------------------------------------------------------------------- #
    # Best Fit Function                                                       #
    # ----------------------------------------------------------------------- #
    def best_fit(self, points):
        """Find rectangle of best fit for points

        Args:
            points (list): List of points to fit.

        Returns:
            :class:`microstructpy.geometry.Rectangle`: an instance of the class
                that best fits the points.
        """
        # Unpack the input points
        pts = np.array(points, dtype='float')
        x, y = pts.T

        # Find the most likely orientation for the rectangle
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        theta = np.arctan(m)

        # Rotate the points to an axis-aligned frame
        s = np.sin(theta)
        c = np.cos(theta)
        rot = np.array([[c, -s], [s, c]])
        pts_prin = pts.dot(rot)

        # Translate points to center of bounding box
        mins = pts_prin.min(axis=0)
        maxs = pts_prin.max(axis=0)
        bb_cen = 0.5 * (mins + maxs)
        pts_bb = pts_prin - bb_cen

        x_min, y_min = pts_bb.min(axis=0)
        x_max, y_max = pts_bb.max(axis=0)

        # Find nearest edge
        x1_dist = np.abs(pts_bb[:, 0] - x_min)
        x2_dist = np.abs(pts_bb[:, 0] - x_max)
        y1_dist = np.abs(pts_bb[:, 1] - y_min)
        y2_dist = np.abs(pts_bb[:, 1] - y_max)

        dists = np.array([x1_dist, x2_dist, y1_dist, y2_dist]).T
        min_dist = dists.min(axis=1)

        mask_x1 = np.isclose(x1_dist, min_dist)
        mask_x2 = np.isclose(x2_dist, min_dist)
        mask_y1 = np.isclose(y1_dist, min_dist)
        mask_y2 = np.isclose(y2_dist, min_dist)

        # Group points by edge
        x_x1 = pts_bb[mask_x1, 0].T
        y_y1 = pts_bb[mask_y1, 1].T
        x_x2 = pts_bb[mask_x2, 0].T
        y_y2 = pts_bb[mask_y2, 1].T

        # Get lines of best fit for each edge
        x1_fit = np.mean(x_x1)
        y1_fit = np.mean(y_y1)
        x2_fit = np.mean(x_x2)
        y2_fit = np.mean(y_y2)

        x_cen = 0.5 * (x1_fit + x2_fit)
        y_cen = 0.5 * (y1_fit + y2_fit)
        fit_cen_bb = np.array([x_cen, y_cen])
        x_len = x2_fit - x1_fit
        y_len = y2_fit - y1_fit

        # Translate center to rotated frame
        fit_cen_prin = fit_cen_bb + bb_cen
        fit_cen = rot.dot(fit_cen_prin.reshape(-1, 1)).flatten()

        # Disambiguate the orientation and axes
        x_ax_seed = np.array(self.matrix)[:, 0]
        x_dot, y_dot = rot.T.dot(x_ax_seed)

        if np.abs(x_dot) > np.abs(y_dot):
            if x_dot > 0:
                x_ax_fit = rot[:, 0]
            else:
                x_ax_fit = - rot[:, 0]

            length = x_len
            width = y_len
        else:
            if y_dot > 0:
                x_ax_fit = rot[:, 1]
            else:
                x_ax_fit = - rot[:, 1]

            length = y_len
            width = x_len

        ang_diff = np.arcsin(np.cross(x_ax_seed, x_ax_fit))
        ang_rad = self.angle_rad + ang_diff

        rect_fit = Rectangle(center=fit_cen, length=length, width=width,
                             angle_rad=ang_rad)

        return rect_fit

    # ----------------------------------------------------------------------- #
    # Representation Function                                                 #
    # ----------------------------------------------------------------------- #
    def __repr__(self):
        repr_str = 'Rectangle('
        repr_str += 'center=' + repr(tuple(self.center)) + ', '
        repr_str += 'side_lengths=' + repr(tuple(self.side_lengths)) + ', '
        repr_str += 'angle=' + repr(self.angle) + ')'
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
        """float: area of rectangle"""
        return self.n_vol

    @classmethod
    def area_expectation(cls, **kwargs):
        if 'length' in kwargs or 'width' in kwargs:
            len_dist = kwargs.get('length', 1)
            width_dist = kwargs.get('width', 1)
            return _prod_exp(*[len_dist, width_dist])

        if 'side_lengths' in kwargs:
            return _prod_exp(*kwargs['side_lengths'])

        if 'bounds' in kwargs:
            x_bnds, y_bnds = kwargs['bounds']
            x_lb, x_ub = x_bnds
            y_lb, y_ub = y_bnds

            p1 = _prod_exp(*[x_ub, y_ub])
            p2 = _prod_exp(*[x_ub, y_lb])
            p3 = _prod_exp(*[x_lb, y_ub])
            p4 = _prod_exp(*[x_lb, y_lb])

            return p1 - p2 - p3 + p4

        raise ValueError('Cannot find the expected area of rectangle')

    # ----------------------------------------------------------------------- #
    # Properties                                                              #
    # ----------------------------------------------------------------------- #
    @property
    def length(self):
        return self.side_lengths[0]

    @property
    def width(self):
        return self.side_lengths[1]

    @property
    def angle(self):
        return self.angle_deg

    @property
    def angle_deg(self):
        return 180 * self.angle_rad / np.pi

    @property
    def angle_rad(self):
        return np.arctan2(self.matrix[1][0], self.matrix[0][0])

    # ----------------------------------------------------------------------- #
    # Circle Approximation                                                    #
    # ----------------------------------------------------------------------- #
    def approximate(self, x1=None):
        if x1 is None:
            x1 = 0.25 * min(self.side_lengths)

        if self.side_lengths[0] >= self.side_lengths[1]:
            length = self.side_lengths[0]
            width = self.side_lengths[1]
            inds = [0, 1]
        else:
            length = self.side_lengths[1]
            width = self.side_lengths[0]
            inds = [1, 0]

        # Centerline circles
        xc = 0
        half_len = 0.5 * length
        r = 0.5 * width
        circs = []
        while xc < half_len:
            circ = [xc, 0, r]
            circs.append(circ)

            if np.isclose(xc + r, half_len):
                xc = half_len
            elif xc + x1 > half_len - r:
                xc = half_len - r
            else:
                xc = xc + x1

        # Corner circle
        r -= x1
        while r > 0:
            circs.append([half_len - r, 0.5 * width - r, r])
            r -= x1

        # Reflect circles
        circs = np.array(circs)
        for dim in range(self.n_dim):
            mask = circs[:, dim] > 0
            new_circs = np.copy(circs[mask])
            new_circs[:, dim] *= -1
            circs = np.concatenate((circs, new_circs))

        circs[:, inds] = circs[:, [0, 1]]

        # Rotate and translate circles
        pts = circs[:, :-1]
        circs[:, :-1] = pts.dot(np.array(self.matrix).T)
        circs[:, :-1] += self.center
        return circs

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, **kwargs):
        """Plot the rectangle.

        This function adds a `matplotlib.patches.Rectangle`_ patch to the
        current axes. The keyword arguments are passed through to the patch.

        Args:
            **kwargs (dict): Keyword arguments for the patch.

        .. _`matplotlib.patches.Rectangle` : https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html
        """  # NOQA: E501
        pt = self.corner
        w = self.length
        h = self.width
        ang = self.angle
        c = patches.Rectangle(pt, w, h, ang, **kwargs)
        plt.gca().add_patch(c)


def _prod_exp(*args):
    prod = 1
    for arg in args:
        try:
            arg_mu = arg.moment(1)
        except AttributeError:
            arg_mu = arg
        prod *= arg_mu
    return prod


# --------------------------------------------------------------------------- #
#                                                                             #
# Square Class                                                                #
#                                                                             #
# --------------------------------------------------------------------------- #
class Square(Rectangle):
    """A square.

    This class contains a generic, 2D square. It is derived from the
    :class:`microstructpy.geometry.Rectangle` class and contains the
    ``side_length`` property, rather than multiple side lengths.

    Args:
        side_length (float, optional): Side length. Defaults to 1.
        center (list, optional): Center of rectangle. Defaults to (0, 0).
        corner (list, optional): bottom-left corner.
    """

    def __init__(self, **kwargs):
        if 'side_length' in kwargs:
            kwargs['side_lengths'] = 2 * [kwargs['side_length']]

        Rectangle.__init__(self, **kwargs)

    # ----------------------------------------------------------------------- #
    # Side Length Property                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def side_length(self):
        """float: length of the side of the square."""
        return self.side_lengths[0]

    # ----------------------------------------------------------------------- #
    # Area                                                                    #
    # ----------------------------------------------------------------------- #
    @classmethod
    def area_expectation(cls, **kwargs):
        if 'side_length' in kwargs:
            len_dist = kwargs['side_length']

            try:
                area_exp = len_dist.moment(2)
            except AttributeError:
                area_exp = len_dist * len_dist
            return area_exp

        Rectangle.area_expectation(**kwargs)
