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
# NSphere Class                                                               #
#                                                                             #
# --------------------------------------------------------------------------- #
class NSphere(object):
    """An N-dimensional sphere.

    This class represents a generic, n-dimensional sphere. It is defined by
    a center point and size parameter, which can be either radius or diameter.

    If multiple size or position keywords are given, there is no guarantee
    whhich keywords are used to create the geometry.

    Args:
        r (float): *(optional)* The radius of the n-sphere.
            Defaults to 1.
        center (list): *(optional)* The coordinates of the center.
            Defaults to [].
        radius : Alias for ``r``.
        d : Alias for ``2*r```.
        diameter : Alias for ``2*r``.
        size : Alias for ``2*r``.
        position : Alias for ``center``.

    """
    # ----------------------------------------------------------------------- #
    # Constructor                                                             #
    # ----------------------------------------------------------------------- #
    def __init__(self, **kwargs):
        if 'r' in kwargs:
            self.r = kwargs['r']
        elif 'radius' in kwargs:
            self.r = kwargs['radius']
        elif 'd' in kwargs:
            self.r = 0.5 * kwargs['d']
        elif 'diameter' in kwargs:
            self.r = 0.5 * kwargs['diameter']
        elif 'size' in kwargs:
            self.r = 0.5 * kwargs['size']
        else:
            self.r = 1

        if 'center' in kwargs:
            self.center = kwargs['center']
        elif 'position' in kwargs:
            self.center = kwargs['position']
        else:
            self.center = []

    @classmethod
    def best_fit(cls, points):
        """Find n-sphere of best fit for set of points.

        This function takes a list of points and computes an n-sphere of
        best fit, in an algebraic sense. This method was developed using the
        a published writeup, which was extended from 2D to ND. [#bullock]_

        Args:
            points (list, numpy.ndarray): List of points to fit.

        Returns:
            NSphere: An instance of the class that fits the points.

        .. [#bullock] Circle fitting writup by Randy Bullock,
          https://dtcenter.org/met/users/docs/write_ups/circle_fit.pdf
        """  # NOQA: E501
        # convert points to numpy array
        pts = np.array(points)
        n_pts, n_dim = pts.shape
        if n_pts <= n_dim:
            mid = pts.mean(axis=0)
            rel_pos = pts - mid
            dist = np.linalg.norm(rel_pos, axis=1).mean()
            return cls(center=mid, radius=dist)

        # translate points to average position
        bcenter = pts.mean(axis=0)
        pts -= bcenter

        # Assemble matrix and vector of sums
        mat = np.zeros((n_dim, n_dim))
        vec = np.zeros(n_dim)

        for i in range(n_dim):
            for j in range(n_dim):
                mat[i, j] = np.sum(pts[:, i] * pts[:, j])
                vec[i] += np.sum(pts[:, i] * pts[:, j] * pts[:, j])
        vec *= 0.5

        # Solve linear system for the center
        try:
            cen_b = np.linalg.solve(mat, vec)
        except np.linalg.linalg.LinAlgError:
            cen_b = pts.mean(axis=0)
        cen = cen_b + bcenter

        # Calculate the radius
        alpha = np.sum(cen_b * cen_b) + np.trace(mat) / n_pts
        R = np.sqrt(alpha)

        # Create the instance
        return cls(center=cen, radius=R)

    # ----------------------------------------------------------------------- #
    # String and Representation Functions                                     #
    # ----------------------------------------------------------------------- #
    def __str__(self):
        str_str = 'Radius: ' + str(self.r) + '\n'
        str_str += 'Center: ' + str(tuple(self.center))
        return str_str

    def __repr__(self):
        repr_str = 'NSphere('
        repr_str += 'r=' + repr(self.r) + ', '
        repr_str += 'center=' + repr(tuple(self.center)) + ')'
        return repr_str

    # ----------------------------------------------------------------------- #
    # Equality                                                                #
    # ----------------------------------------------------------------------- #
    def __eq__(self, nsphere):
        if not hasattr(nsphere, 'r'):
            return False

        if not np.isclose(self.r, nsphere.r):
            return False

        if not hasattr(nsphere, 'center'):
            return False

        c1 = np.array(self.center)
        c2 = np.array(nsphere.center)

        if c1.shape != c2.shape:
            return False

        dx = np.array(self.center) - np.array(nsphere.center)
        if not np.all(np.isclose(dx, 0)):
            return False
        return True

    def __neq__(self, nsphere):
        return not self.__eq__(nsphere)

    # ----------------------------------------------------------------------- #
    # Size Setters/Getters                                                    #
    # ----------------------------------------------------------------------- #
    @property
    def radius(self):
        """float: radius of n-sphere."""
        return self.r

    @property
    def d(self):
        """float: diameter of n-sphere."""
        return 2 * self.r

    @property
    def diameter(self):
        """float: diameter of n-sphere."""
        return 2 * self.r

    @property
    def size(self):
        """float: size (diameter) of n-sphere."""
        return 2 * self.r

    @property
    def position(self):
        """list: position of n-sphere."""
        return self.center

    # ----------------------------------------------------------------------- #
    # Bounding N-Spheres                                                      #
    # ----------------------------------------------------------------------- #
    @property
    def bound_max(self):
        """tuple: maximum bounding n-sphere"""
        return tuple(list(self.center) + [self.r])

    @property
    def bound_min(self):
        """tuple: minimum interior n-sphere"""
        return self.bound_max

    # ----------------------------------------------------------------------- #
    # Limits                                                                  #
    # ----------------------------------------------------------------------- #
    @property
    def limits(self):
        """list: list of (lower, upper) bounds for the bounding box"""
        return [(x - self.r, x + self.r) for x in self.center]

    @property
    def sample_limits(self):
        """list: list of (lower, upper) bounds for the sampling region"""
        return self.limits

    # ----------------------------------------------------------------------- #
    # Approximate                                                             #
    # ----------------------------------------------------------------------- #
    def approximate(self):
        """Approximate the n-sphere with itself

        Other geometries can be approximated by a set of circles or spheres.
        For the n-sphere, this approximation is exact.

        Returns:
            list: A list containing [(x, y, z, ..., r)]
        """
        return [tuple(list(self.center) + [self.r])]

    # ----------------------------------------------------------------------- #
    # Within Test                                                             #
    # ----------------------------------------------------------------------- #
    def within(self, points):
        """Test if points are within n-sphere.

        This function tests whether a point or set of points are within the
        n-sphere. For the set of points, a list of booleans is returned to
        indicate which points are within the n-sphere.

        Args:
            points (list or numpy.ndarray): Point or list of points.

        Returns:
            bool or numpy.ndarray: Set to True for points in geometry.
        """
        pts = np.array(points)
        single_pt = pts.ndim == 1
        if single_pt:
            pts = pts.reshape(1, -1)

        rel_pos = pts - np.array(self.center)
        sq_dist = np.sum(rel_pos * rel_pos, axis=-1)

        mask = sq_dist <= self.r * self.r
        if single_pt:
            return mask[0]
        else:
            return mask

    # ----------------------------------------------------------------------- #
    # Reflect                                                                 #
    # ----------------------------------------------------------------------- #
    def reflect(self, points):
        """Reflect points across surface.

        This function reflects a point or set of points across the surface
        of the n-sphere. Points at the center of the n-sphere are not
        reflected.

        Args:
            points (list or numpy.ndarray): Points to reflect.

        Returns:
            numpy.ndarray: Reflected points.
        """
        pts = np.array(points)
        single_pt = pts.ndim == 1
        if single_pt:
            pts = pts.reshape(1, -1)

        rel_pos = pts - np.array(self.center)
        cen_dist = np.sqrt(np.sum(rel_pos * rel_pos, axis=-1))
        mask = cen_dist > 0
        new_dist = 2 * self.r - cen_dist[mask]
        scl = new_dist / cen_dist[mask]

        new_rel_pos = np.zeros(pts.shape)
        new_rel_pos[mask] = scl.reshape(-1, 1) * rel_pos[mask]
        new_rel_pos[~mask] = 0
        new_rel_pos[~mask, 0] = 2 * self.r

        new_pts = new_rel_pos + np.array(self.center)
        if single_pt:
            return new_pts[0]
        else:
            return new_pts
