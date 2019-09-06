# -*- coding: utf-8 -*-
r"""
Plot all demos

"""

import glob
import locale
import os
import subprocess

import microstructpy as msp

locale.setlocale(locale.LC_NUMERIC, "C")

example_dir = '../../../examples'

# Run XML files
xml_pattern = os.path.join(example_dir, '*.xml')
for filename in glob.glob(xml_pattern):
    msp.cli.run_file(filename)

# Run Python Scripts
py_pattern = os.path.join(example_dir, '*.py')
for filename in glob.glob(py_pattern):
    subprocess.call(['python', filename])
