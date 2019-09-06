# -*- coding: utf-8 -*-
r"""
1. Basic Example
++++++++++++++++

This input files contains two phases with a few options.

XML Input File
^^^^^^^^^^^^^^

The basename for this file is ``intro_1_basic.xml``.
The file can be run using this command::

    microstructpy --demo=intro_1_basic.xml

File contents:

.. literalinclude:: ../../../../examples/intro_1_basic.xml
    :language: xml

Explanation
^^^^^^^^^^^

There are two materials, in a 2:1 ratio based on volume.
The first is a matrix, which is represented with small circles.
The second material consists of circular inclusions with diameter 2.
These two materials fill a square domain.
The bottom-left corner of the rectangle is the origin, which puts the
rectangle in the first quadrant.
The side length is 20, which is 10x the size of the inclusions.
PNG files of each step in the process will be output, as well as the
intermediate text files.
They are saved in a folder named ``intro_1_basic``, in the current directory
(i.e ``./intro_1_basic``).

Output Plots
^^^^^^^^^^^^

"""

# sphinx_gallery_thumbnail_number = 3

import locale
import os
import shutil

import matplotlib.pyplot as plt
import pylab

import microstructpy as msp

locale.setlocale(locale.LC_ALL, '')

filename = '../../examples/intro_1_basic.xml'

in_data = msp.cli.read_input(filename)
phases = in_data['material']
domain = in_data['domain']
kwargs = in_data['settings']
kwargs['verbose'] = False
msp.cli.run(phases, domain, **kwargs)

dpi = 300

for plot_name in ['seeds', 'polymesh', 'trimesh']:
    src = os.path.join('../../examples/intro_1_basic', plot_name + '.png')
    im = plt.imread(src)

    figsize = (im.shape[0] / dpi, im.shape[1] / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(im)
    plt.axis('tight')
