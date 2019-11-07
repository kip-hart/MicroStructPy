"""Seed List

This module contains the class definition for the SeedList class.
"""
# --------------------------------------------------------------------------- #
#                                                                             #
# Import Modules                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #

from __future__ import division
from __future__ import print_function

import warnings

import aabbtree
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib import collections
from matplotlib.patches import Rectangle
from pyquaternion import Quaternion
from scipy.spatial import distance

from microstructpy import _misc
from microstructpy import geometry
from microstructpy.seeding import seed as _seed

__all__ = ['SeedList']
__author__ = 'Kenneth (Kip) Hart'


# --------------------------------------------------------------------------- #
#                                                                             #
# SeedList Class                                                              #
#                                                                             #
# --------------------------------------------------------------------------- #
class SeedList(object):
    """List of seed geometries.

    The SeedList is similar to a standard Python list, but contains instances
    of the :class:`.Seed` class. It can be generated from a list of Seeds,
    by creating enough seeds to fill a given volume, or by reading the content
    of a cache text file.

    Args:
        seeds (list): *(optional)* List of :class:`.Seed` instances.

    """
    # ----------------------------------------------------------------------- #
    # Constructors                                                            #
    # ----------------------------------------------------------------------- #
    def __init__(self, seeds=[]):
        self.seeds = seeds

    @classmethod
    def from_file(cls, filename):
        """Create seed list from file containing list of seeds

        This function creates a seed list from a file containing a list of
        seeds. This file should contain the string representations of seeds,
        separated by a newline character (which is the behavior of
        :meth:`write`).

        Args:
            filename (str): File containing the seed list.

        Returns:
            SeedList: Instance of class.
        """
        with open(filename, 'r') as file:
            file_str = file.read()

        beg = 'Geometry:'
        rem = file_str.split(beg)[1:]

        return cls([_seed.Seed.from_str(beg + s) for s in rem])

    @classmethod
    def from_info(cls, phases, volume, rng_seeds={}):
        """Create seed list from microstructure information

        This function creates a seed list from information about the
        microstruture. The "phases" input should be a list of material
        phase dictionaries, formatted according to the :ref:`phase_dict_guide`
        guide.

        The "volume" input is the minimum volume of the list of seeds. Seeds
        will be added to the list until this volume threshold is crossed.

        Finally, the "rng_seeds" input is a dictionary of random number
        generator (RNG) seeds for each parameter of the seed geometries.
        For example, if one of the phases uses "size" to define the seeds,
        then "size" could be a keyword of the "rng_seeds" input. The value
        should be a non-negative integer, to seed the RNG for size.
        The default RNG seed is 0.

        Note:
            If two or more parameters have the same RNG seed and the same
            kernel of the distribution, those parameters will **not** be
            correlated. This method updates RNG seeds based on the order that
            distributions are sampled to avoid correlation between independent
            random variables.

        Args:
            phases (dict): Dictionary of phase information, see
                :ref:`phase_dict_guide` for a guide.
            volume (float): The total area/volume of the seeds in the list.
            rng_seeds (dict): *(optional)* Dictionary of RNG seeds for each
                step in the seeding process. The dictionary keys should match
                shape parameters in ``phases``. For example::

                    rng_seeds = {
                        'size': 0,
                        'angle': 3,
                    }

        Returns:
            SeedList: An instance of the class containing seeds prescribed by
            the phase information and filling the given volume.

        """

        # determine dimensionality, set default shape
        default_shapes = {2: 'circle', 3: 'sphere'}
        n_dim = None
        if isinstance(phases, dict):
            phases = [phases]

        for phase in phases:
            if 'shape' in phase:
                n_dim = geometry.factory(phase['shape']).n_dim
        if n_dim is None:
            e_str = 'Number of dimensions could not be determined from phase '
            e_str += 'shapes. Consider setting the shape of a phase, or'
            e_str += ' specifying the number of dimensions.'
            raise ValueError(e_str)

        for phase in phases:
            if 'shape' in phase:
                assert geometry.factory(phase['shape']).n_dim == n_dim
            else:
                phase['shape'] = default_shapes[n_dim]

        # compute volume of each phase
        vol_rng = rng_seeds.get('fraction', 0)
        np.random.seed(vol_rng)

        n_phases = len(phases)
        rel_vols = np.ones(n_phases)
        for i, phase in enumerate(phases):
            vol = phase.get('fraction', 1)
            try:
                v_sample = -1
                while v_sample < 0:
                    v_sample = vol.rvs()
                rel_vols[i] = v_sample
            except AttributeError:
                rel_vols[i] = vol
        vol_fracs = rel_vols / sum(rel_vols)
        phase_vols = volume * vol_fracs

        # compute number of seeds for each phase
        if n_dim == 2:
            avg_vols = [geometry.factory(p['shape']).area_expectation(**p)
                        for p in phases]
        else:
            avg_vols = [geometry.factory(p['shape']).volume_expectation(**p)
                        for p in phases]
        weights = phase_vols / np.array(avg_vols)
        pop_fracs = weights / sum(weights)

        seed_vol = 0
        seeds = []
        max_int = np.iinfo(np.int32).max
        while seed_vol < volume:
            # Pick the phase
            rng_seed = rng_seeds.get('phase', 0)
            np.random.seed(rng_seed)
            phase_num = np.random.choice(n_phases, p=pop_fracs)
            phase = phases[phase_num]
            rng_seeds['phase'] = np.random.randint(max_int)

            # Create the seed
            seed_shape = phase['shape']
            seed_args = {'phase': phase_num}
            kw_n = 0
            for kw in set(phase) - set(_misc.gen_kws):
                # set the RNG seed
                rng_seed = rng_seeds.get(kw, 0)
                np.random.seed(rng_seed)

                # Sample, with special cases for orientation
                if kw not in _misc.ori_kws:
                    try:
                        val = phase[kw].rvs(random_state=rng_seed)
                    except AttributeError:
                        val = phase[kw]
                    seed_args[kw] = val
                elif (phase[kw] == 'random') and (n_dim == 2):
                    np.random.seed(rng_seed)
                    seed_args['angle_deg'] = 360 * np.random.rand()
                elif phase[kw] == 'random':
                    np.random.seed(rng_seed)
                    elems = np.random.normal(size=4)
                    mag = np.linalg.norm(elems)
                    elems /= mag
                    val = Quaternion(elems).rotation_matrix
                    seed_args[kw] = val
                elif kw in ['rot_seq', 'rot_seq_deg', 'rot_seq_rad']:
                    seq = []
                    val = phase[kw]
                    if not isinstance(val, list):
                        val = [val]
                    for rotation in val:
                        rot_dict = {str(kw): rotation[kw] for kw in rotation}
                        ax = rot_dict.get('axis', 'x')
                        ang_dist = rot_dict.get('angle', 0)
                        try:
                            ang = ang_dist.rvs(random_state=rng_seed)
                        except AttributeError:
                            ang = ang_dist
                        seq.append((ax, ang))
                    seed_args[kw] = seq
                else:
                    try:
                        val = phase[kw].rvs(random_state=rng_seed)
                    except AttributeError:
                        val = phase[kw]
                    seed_args[kw] = val

                # Update the RNG seed
                np.random.seed(rng_seed + kw_n)
                rng_seeds[kw] = np.random.randint(max_int - kw_n)
                kw_n += 1

            # Add seed to list
            seed = _seed.Seed.factory(seed_shape, **seed_args)
            seeds.append(seed)
            seed_vol += seed.volume

        return cls(seeds)

    # ----------------------------------------------------------------------- #
    # Representation and String Functions                                     #
    # ----------------------------------------------------------------------- #
    def __repr__(self):
        repr_str = 'SeedList('
        repr_str += repr(self.seeds)
        repr_str += ')'
        return repr_str

    def __str__(self):
        str_str = '\n'.join([str(s) for s in self.seeds])
        return str_str

    # ----------------------------------------------------------------------- #
    # List Methods                                                            #
    # ----------------------------------------------------------------------- #
    def __getitem__(self, i):
        if isinstance(i, slice):
            return SeedList(self.seeds[i])
        elif np.issubdtype(type(i), np.integer):
            return self.seeds[i]
        elif all([np.issubdtype(type(b), np.bool_) for b in i]):
            return SeedList([s for s, b in zip(self.seeds, i) if b])
        elif all([np.issubdtype(type(k), np.integer) for k in i]):
            return SeedList([self.seeds[k] for k in i])
        else:
            print(i.dtype)
            raise ValueError('Cannot index with type ' + str(type(i)))

    def __setitem__(self, i, s):
        try:
            self.seeds[int(i)] = s
        except TypeError:
            n_added = 0
            for ind, i_val in enumerate(i):
                if str(i_val) == 'True':
                    self.seeds[ind] = s[n_added]
                    n_added += 1
                elif str(i_val) != 'False':
                    self.seeds[int(i_val)] = s[ind]

    def __add__(self, seedlist):
        """Add seed lists together

        This function overloads the + operator, similar to
        :meth:`list.__add__`.

        Args:
            seedlist (SeedList, list): List of seeds to add.

        Returns:
            SeedList: Instance that joins the two seed lists.

        .. versionadded:: 1.1
        """
        if type(self) == type(seedlist):
            return SeedList(self.seeds + seedlist.seeds)
        else:
            return SeedList(self.seeds + seedlist)

    def __len__(self):
        return len(self.seeds)

    def append(self, seed):
        """Append seed

        This function appends a seed to the list.

        Args:
            seed (Seed): The seed to append to the list

        """
        self.seeds.append(seed)

    def extend(self, seeds):
        """Extend seed list

        This function adds a list of seeds to the end of the seed list.

        Args:
            seeds (list or SeedList): List of seeds

        """
        if isinstance(seeds, SeedList):
            self.seeds.extend(seeds.seeds)
        else:
            self.seeds.extend(seeds)

    # ----------------------------------------------------------------------- #
    # Comparison Methods                                                      #
    # ----------------------------------------------------------------------- #
    def __eq__(self, other_list):
        same = len(self) == len(other_list)
        if same:
            for s1, s2 in zip(self, other_list):
                same &= s1 == s2
        return same

    # ----------------------------------------------------------------------- #
    # Write Function                                                          #
    # ----------------------------------------------------------------------- #
    def write(self, filename):
        """Write seed list to a text file

        This function writes out the seed list to a file. The content of this
        file is human-readable and can be read by the
        :func:`SeedList.from_file` method.

        Args:
            filename (str): File to write the seed list.
        """

        with open(filename, 'w') as file:
            file.write(str(self) + '\n')

    # ----------------------------------------------------------------------- #
    # Plot Function                                                           #
    # ----------------------------------------------------------------------- #
    def plot(self, **kwargs):
        """Plot the seeds in the seed list.

        This function plots the seeds contained in the seed list.
        In 2D, the seeds are grouped into matplotlib collections to reduce
        the computational load. In 3D, matplotlib does not have patches, so
        each seed is rendered as its own surface.

        Additional keyword arguments can be specified and passed through to
        matplotlib. These arguments should be either single values
        (e.g. ``edgecolors='k'``), or lists of values that have the same
        length as the seed list.

        Args:
            **kwargs: Keyword arguments to pass to matplotlib

        """
        seed_args = [{} for seed in self]
        for key, val in kwargs.items():
            if type(val) in (list, np.array):
                for args, elem in zip(seed_args, val):
                    args[key] = elem
            else:
                for args in seed_args:
                    args[key] = val

        n = self[0].geometry.n_dim

        if n == 3:
            for seed, args in zip(self, seed_args):
                seed.plot(**args)

        elif n == 2:
            ellipse_data = {'w': [], 'h': [], 'a': [], 'xy': []}
            ec_kwargs = {}

            rect_data = {'xy': [], 'w': [], 'h': [], 'angle': []}
            rect_kwargs = {}

            pc_verts = []
            pc_kwargs = {}
            for seed, args in zip(self, seed_args):
                geom_name = type(seed.geometry).__name__.lower().strip()
                if geom_name == 'ellipse':
                    a, b = seed.geometry.axes
                    cen = np.array(seed.position)
                    t = seed.geometry.angle_deg

                    ellipse_data['w'].append(2 * a)
                    ellipse_data['h'].append(2 * b)
                    ellipse_data['a'].append(t)
                    ellipse_data['xy'].append(cen)

                    for key, val in args.items():
                        val_list = ec_kwargs.get(key, [])
                        val_list.append(val)
                        ec_kwargs[key] = val_list

                elif geom_name == 'circle':
                    diam = seed.geometry.diameter
                    cen = np.array(seed.position)

                    ellipse_data['w'].append(diam)
                    ellipse_data['h'].append(diam)
                    ellipse_data['a'].append(0)
                    ellipse_data['xy'].append(cen)

                    for key, val in args.items():
                        val_list = ec_kwargs.get(key, [])
                        val_list.append(val)
                        ec_kwargs[key] = val_list

                elif geom_name in ['rectangle', 'square']:
                    w, h = seed.geometry.side_lengths
                    corner = seed.geometry.corner
                    t = seed.geometry.angle_deg

                    rect_data['w'].append(w)
                    rect_data['h'].append(h)
                    rect_data['angle'].append(t)
                    rect_data['xy'].append(corner)

                    for key, val in args.items():
                        val_list = rect_kwargs.get(key, [])
                        val_list.append(val)
                        rect_kwargs[key] = val_list

                elif geom_name == 'curl':
                    xy = seed.geometry.plot_xy()
                    pc_verts.append(xy)
                    for key, val in args.items():
                        val_list = pc_kwargs.get(key, [])
                        val_list.append(val)
                        pc_kwargs[key] = val_list

                elif geom_name == 'nonetype':
                    pass

                else:
                    e_str = 'Cannot plot groups of ' + geom_name
                    e_str += ' yet.'
                    raise NotImplementedError(e_str)

            # abbreviate kwargs if all the same
            for key, val in ec_kwargs.items():
                v1 = val[0]
                same = True
                for v in val:
                    same &= v == v1
                if same:
                    ec_kwargs[key] = v1

            for key, val in pc_kwargs.items():
                v1 = val[0]
                same = True
                for v in val:
                    same &= v == v1
                if same:
                    pc_kwargs[key] = v1

            # Plot Circles and Ellipses
            ax = plt.gca()

            w = np.array(ellipse_data['w'])
            h = np.array(ellipse_data['h'])
            a = np.array(ellipse_data['a'])
            xy = np.array(ellipse_data['xy'])
            ec = collections.EllipseCollection(w, h, a, units='x', offsets=xy,
                                               transOffset=ax.transData,
                                               **ec_kwargs)
            ax.add_collection(ec)

            # Plot Rectangles
            rects = [Rectangle(xy=xyi, width=wi, height=hi, angle=ai) for
                     xyi, wi, hi, ai in zip(rect_data['xy'], rect_data['w'],
                     rect_data['h'], rect_data['angle'])]
            rc = collections.PatchCollection(rects, False, **rect_kwargs)
            ax.add_collection(rc)

            # Plot Polygons
            pc = collections.PolyCollection(pc_verts, **pc_kwargs)
            ax.add_collection(pc)

            ax.autoscale_view()

    def plot_breakdown(self, **kwargs):
        """Plot the breakdowns of the seeds in seed list.

        This function plots the breakdowns of seeds contained in the seed list.
        In 2D, the breakdowns are grouped into matplotlib collections to reduce
        the computational load. In 3D, matplotlib does not have patches, so
        each breakdown is rendered as its own surface.

        Additional keyword arguments can be specified and passed through to
        matplotlib. These arguments should be either single values
        (e.g. ``edgecolors='k'``), or lists of values that have the same
        length as the seed list.

        Args:
            **kwargs: Keyword arguments to pass to matplotlib

        """
        seed_args = [{} for seed in self]
        for key, val in kwargs.items():
            if type(val) in (list, np.array):
                for args, elem in zip(seed_args, val):
                    args[key] = elem
            else:
                for args in seed_args:
                    args[key] = val

        n = self[0].geometry.n_dim

        if n == 3:
            for seed, args in zip(self, seed_args):
                seed.plot_breakdown(**args)

        elif n == 2:
            breakdowns = np.zeros((0, 3))
            ec_kwargs = {}
            for seed, args in zip(self, seed_args):
                breakdowns = np.concatenate((breakdowns, seed.breakdown))
                n_c = len(seed.breakdown)
                for key, val in args.items():
                    val_list = ec_kwargs.get(key, [])
                    val_list.extend(n_c * [val])
                    ec_kwargs[key] = val_list
            d = 2 * breakdowns[:, -1]
            xy = breakdowns[:, :-1]
            a = np.full(len(breakdowns), 0)

            # abbreviate kwargs if all the same
            for key, val in ec_kwargs.items():
                v1 = val[0]
                same = True
                for v in val:
                    same &= v == v1
                if same:
                    ec_kwargs[key] = v1

            ax = plt.gca()
            ec = collections.EllipseCollection(d, d, a, units='x', offsets=xy,
                                               transOffset=ax.transData,
                                               **ec_kwargs)
            ax.add_collection(ec)
            ax.autoscale_view()

    # ----------------------------------------------------------------------- #
    # Position Function                                                       #
    # ----------------------------------------------------------------------- #
    def position(self, domain, pos_dists={}, rng_seed=0, hold=[],
                 max_attempts=10000, rtol='fit', verbose=False):
        """Position seeds in a domain

        This method positions the seeds within a domain. The "domain" should be
        a geometry instance from the :mod:`microstructpy.geometry` module.

        The "pos_dist" input is for phases with custom position distributions,
        the default being a uniform random distribution.
        For example:

        .. code-block:: python

            import scipy.stats
            mu = [0.5, -0.2]
            sigma = [[2.0, 0.3], [0.3, 0.5]]
            pos_dists = {2: scipy.stats.multivariate_normal(mu, sigma),
                         3: ['random',
                             scipy.stats.norm(0, 1)]
                         }

        Here, phases 0 and 1 have the default distribution, phase 2 has a
        bivariate normal position distribution, and phase 3 is uniform in the
        x and normally distributed in the y. Multivariate distributions are
        described in the multivariate section of the :mod:`scipy.stats`
        documentation.

        The position of certain seeds can be held fixed during the positioning
        process using the "hold" input. This should be a list of booleans,
        where False indicates a seed should not be held fixed and True
        indicates that it should be held fixed. The default behavior is to not
        hold any seeds fixed.

        The "rtol" parameter governs the relative overlap tolerable between
        seeds. Setting rtol to 0 means that there is no overlap, while a value
        of 1 means that one seed's center is on the edge of another seed.
        The default value is 'fit', which determines a tolerance between 0 and
        1 based on the ratio of standard deviation to mean in grain volumes.

        Args:
            domain (from :mod:`microstructpy.geometry`): The domain of the
                microstructure.
            pos_dists (dict): *(optional)* Position distributions for each
                phase, formatted like the example above.
                Defaults to uniform random throughout the domain.
            rng_seed (int): *(optional)* Random number generator (RNG) seed
                for positioning the seeds. Should be a non-negative integer.
            hold (list or numpy.ndarray): *(optional)* List of booleans for
                holding the positions of seeds.
                Defaults to False for all seeds.
            max_attempts (int): *(optional)* Number of random trials before
                removing a seed from the list.
                Defaults to 10,000.
            rtol (str or float): *(optional)* The relative overlap tolerance
                between seeds. This parameter should be between 0 and 1.
                Using the 'fit' option, a function will determine the value
                for rtol based on the mean and standard deviation in seed
                volumes.
            verbose (bool): *(optional)* This option will print a running
                counter of how many seeds have been positioned.
                Defaults to False.

        """  # NOQA: E501
        if len(hold) == 0:
            hold = [False for seed in self]

        # set the spatial distributions
        u_dist = [scipy.stats.uniform(lb, ub - lb) for lb, ub in
                  domain.sample_limits]

        distribs = []
        n_phases = max([s.phase for s in self]) + 1
        for i in range(n_phases):
            distribs.append(pos_dists.get(i, u_dist))

        # Add hold seeds
        tree = aabbtree.AABBTree()
        for i in range(len(self)):
            if hold[i]:
                # add to tree
                aabb = aabbtree.AABB(self[i].geometry.limits)
                tree.add(aabb, i)

        positioned = np.array(hold)
        i_sort = np.flip(np.argsort([s.volume for s in self]))
        posd_sort = positioned[i_sort]
        i_position = i_sort[~posd_sort]

        # allowable overlap, relative to radius
        vols = np.array([s.volume for s in self])
        cv = scipy.stats.variation(vols)
        if domain.n_dim == 2 and rtol == 'fit':
            numer = 0.362954 * cv * cv - 0.419069 * cv + .184959
            denom = cv * cv - 1.05989 * cv + 0.365096
            rtol = numer / denom
        elif rtol == 'fit':
            numer = 0.471115 * cv * cv - 0.602324 * cv + 0.297562
            denom = cv * cv - 1.08469 * cv + 0.428216
            rtol = numer / denom

        # position the remaining seeds
        i_reject = []
        np.random.seed(rng_seed)

        for k, i in enumerate(i_position):
            if verbose:
                print(k + 1, 'of', len(i_position))

            seed = self[i]
            pos_dist = distribs[seed.phase]

            searching = True
            n_attempts = 0
            while searching and n_attempts < max_attempts:
                pt = sample_pos(pos_dist)

                if domain.within(pt):
                    seed.position = pt
                    n_attempts += 1
                else:
                    continue

                bkdwn = np.array(seed.breakdown)
                cens = bkdwn[:, :-1]
                rads = bkdwn[:, -1].reshape(-1, 1)

                aabb = aabbtree.AABB(seed.geometry.limits)
                olap_inds = tree.overlap_values(aabb)
                olap_seeds = self[olap_inds]
                clears = True
                for olap_seed in olap_seeds:
                    o_bkdwn = np.array(olap_seed.breakdown)
                    o_cens = o_bkdwn[:, :-1]
                    o_rads = o_bkdwn[:, -1].reshape(1, -1)

                    dists = distance.cdist(cens, o_cens)
                    tol = rtol * np.minimum(rads, o_rads)
                    total_dists = dists + tol - rads - o_rads
                    if np.any(total_dists < 0):
                        clears = False
                        break

                searching = not clears

            if searching:
                i_reject.append(i)
            else:
                positioned[i] = True
                self[i] = seed

                # add to tree
                aabb = aabbtree.AABB(seed.geometry.limits)
                tree.add(aabb, i)

        keep_mask = np.array(len(self) * [True])
        keep_mask[i_reject] = False

        reject_seeds = self[~keep_mask]
        self.seeds = self[keep_mask].seeds
        if len(reject_seeds) > 0:
            f = 'seed_position_reject.log'
            reject_seeds.write(f)

            w_str = 'Seeds were removed from the seed list during positioning.'
            w_str += ' Their data has beeen written to ' + f + ' and their'
            w_str += ' indices were ' + str(i_reject) + '.'
            warnings.warn(w_str, RuntimeWarning)


def sample_pos(distribution, n=1):
    """ Sample position distribution

    This function returns a sample of the postion distribution.
    This distribution can be either a list of independent distributions
    for each axis, or a single multi-variate distribution. A list of
    multi-variate distributions is given on the `SciPy stats website`_.

    Two examples of position distributions are given below.

    .. code-block:: python

        # three independent distributions
        distribution = [scipy.stats.uniform(-1, 2),
                        scipy.stats.norm(0, 1),
                        scipy.stats.binom(5, 0.4)]

        # one multi-variate distribution
        mu = [2, -3 , 5]
        sigma = [[1, 3, 0], [3, 1, 2], [0, 2, 2]]
        distribution = scipy.stats.multivariate_normal(mu, sigma)

    Args:
        distribution (list or scipy.stats distribution): The position
            distribution.

        n (int): *(optional)* Number of samples. Defaults to 1.

    Returns:
        list: A sample of the distribution.
    """  # NOQA : E501
    if type(distribution) is list:
        pos = np.full((n, len(distribution)), 0, dtype='float')
        for j, coord_dist in enumerate(distribution):
            try:
                pos[:, j] = coord_dist.rvs(n)
            except AttributeError:
                pos[:, j] = coord_dist
    else:
        pos = distribution.rvs(n)

    if n == 1:
        return pos[0]
    else:
        return pos
