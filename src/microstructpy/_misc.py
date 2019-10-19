"""Miscellaneous functions.

This private module contains miscellaneous definitions and the string
conversion function.

"""
import ast

import numpy as np

__author__ = 'Kenneth (Kip) Hart'

kw_solid = {'crystalline', 'granular', 'solid'}
kw_amorph = {'amorphous', 'glass', 'matrix'}
kw_void = {'void', 'crack', 'hole'}

ori_kws = {'orientation', 'matrix', 'angle', 'angle_deg', 'angle_rad',
           'rot_seq', 'rot_seq_rad', 'rot_seq_deg'}
gen_kws = {'material_type', 'fraction', 'shape', 'name', 'color', 'position'}

demo_needs = {'basalt_circle.xml': ['aphanitic_cdf.csv', 'olivine_cdf.csv'],
              'from_image.py': ['aluminum_micro.png']}

try:
    _unicode = unicode
except NameError:
    _unicode = str


# --------------------------------------------------------------------------- #
#                                                                             #
# Convert String to Value (Infer Type)                                        #
#                                                                             #
# --------------------------------------------------------------------------- #
def from_str(string):
    """Convert string to number.

    This function takes a string and converts it into a number or a list.

    Args:
        string (str): The string.

    Returns:
        The value in the string.

    """
    if type(string) not in (str, _unicode):
        err_str = 'from_str() arg 1 must be a string or unicode'
        raise TypeError(err_str)

    try:
        value = ast.literal_eval(string.strip())
    except (ValueError, SyntaxError):
        value = string

        # Catch lowercase booleans
        cap_str = string.capitalize()
        if cap_str != string:
            cap_value = from_str(cap_str)
            if type(cap_value) not in (str, _unicode):
                value = cap_value
    return value


# --------------------------------------------------------------------------- #
#                                                                             #
# Safe Versions of scipy.stats Distribution Methods                           #
#                                                                             #
# --------------------------------------------------------------------------- #
#
# These functions compute statistical quantities, such as mean or random
# samples, for deterministic quantities.

def rvs(d, *args, **kwargs):
    """Random sample for distribution.

    This function (safely) computes a random sample from a deterministic or
    random variable.

    Args:
        d: Scalar, list, or scipy.stats distribution.
        *args: Positional arguments for d.rvs().
        **kwargs: Keyword arguments for d.rvs().

    Returns:
        int, float, or list: Sample(s) from distribution.

    """
    try:
        iter(d)
    except TypeError:
        pass
    else:
        return [rvs(di, *args, **kwargs) for di in d]

    try:
        val = d.rvs(*args, **kwargs)
    except AttributeError:
        if len(args) > 0:
            size = args[0]
        elif 'size' in kwargs:
            size = kwargs['size']
        else:
            size = 1

        if size > 1:
            val = np.full(size, d)
        else:
            val = d
    return val


def moment(d, n):
    """n-th moment of distribution.

    This function (safely) computes the n-th moment of a deterministic or
    random variable.

    Args:
        d: Scalar, list, or scipy.stats distribution.
        n: Order of the moment.

    Returns:
        int, float, or list: n-th moment of the distribution.

    """
    try:
        iter(d)
    except TypeError:
        pass
    else:
        return [moment(di, n) for di in d]

    try:
        m = d.moment(n)
    except AttributeError:
        m = d ** n
    return m


def mean(d):
    """Mean of the distribution.

    This function (safely) computes the mean of a deterministic or random
    variable.

    Args:
        d: Scalar, list, or scipy.stats distribution.
    
    Returns:
        int, float, or list: Mean of the distribution.

    """
    try:
        iter(d)
    except TypeError:
        pass
    else:
        return [mean(di) for di in d]

    try:
        mu = d.mean()
    except AttributeError:
        mu = d
    return mu
