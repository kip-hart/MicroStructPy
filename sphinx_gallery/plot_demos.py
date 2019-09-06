# -*- coding: utf-8 -*-
r"""
piss off

"""

import glob
import os
import subprocess

import microstructpy as msp


example_dir = '../examples'

# Run XML files
xml_pattern = os.path.join(example_dir, '*.xml')
for filename in glob.glob(xml_pattern):
    msp.cli.run_file(filename)

# Run Python Scripts
py_pattern = os.path.join(example_dir, '*.py')
for filename in glob.glob(py_pattern):
    subprocess.call(['python', filename])
