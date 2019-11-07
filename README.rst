MicroStructPy - Microstructure Mesh Generation in Python
========================================================

|s-travis|
|s-license|
|s-doi|

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


.. image:: https://docs.microstructpy.org/en/latest/_images/banner.png
    :alt: Banner image showing the three steps for creating microstructure.

*The three steps to creating a microstructure are:
1) seed the domain with particles,
2) create a Voronoi power diagram, and
3) convert the diagram into an unstructured mesh.*

Download & Installation
-----------------------

To install MicroStructPy, download it from PyPI using::

    pip install microstructpy

If there is an error with the install, try ``pip install pybind11`` first,
then install MicroStructPy.


MicroStructPy can also be installed from source::

    git clone https://github.com/kip-hart/MicroStructPy.git
    pip install -e MicroStructPy/

Installing MicroStructPy creates the command line program ``microstructpy`` and
the Python package ``microstructpy``.
The command line program executes a standard workflow on XML input files,
while the package exposes classes and functions for a customized workflow.


Run a Demo
----------

MicroStructPy includes several demo and example files to help new users get
started with the program.
A full list of examples is available online at
https://docs.microstructpy.org/examples.html.

Here is minimal example input file:

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

This example can be run from the command line by excuting::

    microstructpy --demo=minimal.xml

Alternatively, you can copy the text to a file such as
``my_input.xml`` and run ``microstructpy my_input.xml``.

The same output can be obtained from using the package in a script:

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

Documentation
-------------

MicroStructPy documentation is available online at
https://docs.microstructpy.org.

To build a local copy of the documentation, execute the following from the
top-level directory of the MicroStructPy repository::

    pip install tox
    tox -e docs

Once built, the documentation will be in ``docs/build/``.

Contributing
------------

Contributions to the project are welcome.
Please use the GitHub pull request and issue submission features.
See the `Contributing Guidelines`_ for more details.

License and Attributions
------------------------

MicroStructPy is open source and freely available.
Copyright for MicroStructPy is held by Georgia Tech Research Corporation.
MicroStructPy is a major part of Kenneth (Kip) Hart's doctoral thesis,
advised by Prof. Julian Rimoli.


.. LINKS

.. _Documentation : https://microstructpy.readthedocs.io
.. _GitHub: https://github.com/kip-hart/MicroStructPy
.. _PyPI : https://pypi.org/project/microstructpy/
.. _Repository: https://github.com/kip-hart/MicroStructPy
.. _`Contributing Guidelines`: https://github.com/kip-hart/MicroStructPy/blob/dev/.github/CONTRIBUTING.md

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

.. |s-doi| image:: https://zenodo.org/badge/206468500.svg
   :target: https://zenodo.org/badge/latestdoi/206468500
   :alt: DOI
