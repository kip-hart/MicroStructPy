.. MicroStructPy documentation master file.

MicroStructPy - Microstructure Mesh Generation in Python
========================================================

|s-travis|
|s-license|

|l-github| `Repository`_
|l-rtd| `Documentation`_
|l-pypi| `PyPI`_


MicroStructPy is a microstructure mesh generator written in Python.
Features of MicroStructPy include:

* 2D and 3D microstructures
* Grain size, shape, orientation, and position control
* Polycrystals, amorphous phases, and voids
* Mesh verification
* Visualizations
* Output to common file formats
* Customizable workflow

.. image:: ../../examples/docs_banner/banner.png
    :alt: Banner image showing the three steps for creating microstructure.


Quick Start
-----------

To install MicroStructPy, download it from PyPI using::

    pip install microstructpy

If there is an error with the install, try ``pip install pybind11`` first,
then install MicroStructPy.
This will create a command line executable and python package both
named ``microstructpy``.
To use the command line interface, create a file called ``input.xml`` and copy
this into it:

.. code-block:: XML

    <?xml version="1.0" encoding="UTF-8"?>
    <input>
        <material>
            <shape> circle </shape>
            <size> 0.15 </size>
        </material>

        <domain>
            <shape> square </shape>
        </domain>
    </input>

Next, run the file from the command line::

    microstructpy input.xml

This will produce three text files and three image files: ``seeds.txt``,
``polymesh.txt``, ``trimesh.txt``, ``seeds.png``, ``polymesh.png``, and
``trimesh.png``.
The text files contain all of the data related to the seed geometries and
meshes.
The image files contain:

.. image:: ../../examples/seeds.png
    :width: 30%
    :alt: Seed geometries for minimal example.

.. image:: ../../examples/polymesh.png
    :width: 30%
    :alt: Polygonal mesh for minimal example.

.. image:: ../../examples/trimesh.png
    :width: 30%
    :alt: Unstructured mesh for minimal example.

The same results can be produced using this script:

.. code-block:: python

    import matplotlib.pyplot as plt
    import microstructpy as msp


    phase = {'shape': 'circle', 'size': 0.15}
    domain = msp.geometry.Square()

    # Unpositioned list of seeds
    seeds = msp.seeding.SeedList.from_info(phase, domain.area)

    # Position seeds in domain
    seeds.position(domain)

    # Create polygonal mesh
    polygon_mesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

    # Create triangular mesh
    triangle_mesh = msp.meshing.TriMesh.from_polymesh(polygon_mesh)

    # Plot outputs
    for output in [seeds, polygon_mesh, triangle_mesh]:
        plt.figure()
        output.plot(edgecolor='k')
        plt.axis('image')
        plt.axis([-0.5, 0.5, -0.5, 0.5])
        plt.show()


License and Attribution
-----------------------

MicroStructPy is open source and freely availabe under the terms of the the
MIT license.
Copyright for MicroStructPy is held by Georgia Tech Research Corporation.
MicroStructPy is a major part of Kenneth (Kip) Hart's doctoral thesis,
advised by Prof. Julian Rimoli.

Contents
--------

.. toctree::
   :maxdepth: 2

   getting_started
   examples
   cli
   package_guide
   troubleshooting
   API <microstructpy>

.. LINKS

.. _Documentation : https://microstructpy.readthedocs.io
.. _GitHub: https://github.com/kip-hart/MicroStructPy
.. _PyPI : https://pypi.org/project/microstructpy/
.. _Repository: https://github.com/kip-hart/MicroStructPy

.. EXTERNAL IMAGES

.. |l-github| image:: https://api.iconify.design/octicon:mark-github.svg?color=black0&inline=true&height=16
    :alt: GitHub

.. |l-rtd| image:: https://api.iconify.design/simple-icons:readthedocs.svg?color=black&inline=true&height=16
    :alt: ReadTheDocs

.. |l-pypi| image:: https://api.iconify.design/mdi:cube-outline.svg?color=black&inline=true&height=16
    :alt: PyPI


.. SHIELDS

.. |s-travis| image:: https://img.shields.io/travis/kip-hart/MicroStructPy
    :target: https://travis-ci.org/kip-hart/MicroStructPy
    :alt: Travis CI

.. |s-license| image:: https://img.shields.io/github/license/kip-hart/MicroStructPy
    :target: https://github.com/kip-hart/MicroStructPy/blob/master/LICENSE.rst
    :alt: License