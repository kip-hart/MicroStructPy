#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(fname):
    return open(join(dirname(__file__), fname)).read()


setup(
    name='MicroStructPy',
    version='1.0',
    license='MIT License',
    description='Statistical microstructure generation.',
    long_description=read('README.rst'),
    long_description_content_type=' text/x-rst',
    author='Kenneth (Kip) Hart',
    author_email='kiphart91@gmail.com',
    url='https://github.com/kip-hart/MicroStructPy',
    project_urls={
        'Documentation': 'https://microstructpy.readthedocs.io',
    },
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    keywords=[
        'microstructure',
        'micromechanics',
        'finite element',
        'FEM', 'FEA',
        'mesh',
        'polycrystal',
        'tessellation',
        'Laguerre tessellation',
        'multi-sphere'
    ],
    install_requires=[
        'aabbtree',
        'matplotlib>=2.2.0',
        'pybind11',  # must come before meshpy for successful install
        'meshpy',
        'numpy>=1.13.0',
        'pyquaternion',
        'pyvoro-mmalahe',  # install issue with pyvoro
        'scipy',
        'xmltodict'
    ],
    entry_points={
        'console_scripts': [
            'microstructpy = microstructpy.cli:main',
        ]
    },
)
