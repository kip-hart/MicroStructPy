"""Miscellaneous functions.

This private module contains miscellaneous definitions and the string
conversion function.

"""
import ast

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
