MicroStructPy - Microstructure Mesh Generation in Python
========================================================

|s-ci|
|s-license|

|s-doi1|
|s-doi2|

|l-github| `Repository <https://github.com/kip-hart/MicroStructPy>`_
|l-rtd| `Documentation <https://docs.microstructpy.org>`_
|l-pypi| `PyPI <https://pypi.org/project/microstructpy/>`_

.. end-badges

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

.. begin-publications

Publications
------------

If you use MicroStructPy in you work, please consider including these citations
in your bibliography:

K. A. Hart and J. J. Rimoli, Generation of statistically representative
microstructures with direct grain geomety control,
*Computer Methods in Applied Mechanics and Engineering*, 370 (2020), 113242.
(`BibTeX <https://github.com/kip-hart/MicroStructPy/raw/master/docs/publications/cmame2020.bib>`__)
(`DOI <https://doi.org/10.1016/j.cma.2020.113242>`__)

K. A. Hart and J. J. Rimoli, MicroStructPy: A statistical microstructure mesh
generator in Python, *SoftwareX*, 12 (2020), 100595.
(`BibTeX <https://github.com/kip-hart/MicroStructPy/raw/master/docs/publications/swx2020.bib>`__)
(`DOI <https://doi.org/10.1016/j.softx.2020.100595>`__)

The news article `AE Doctoral Student Kenneth A. Hart Presents MicroStructPy to the World <https://www.ae.gatech.edu/news/2020/07/ae-doctoral-student-kenneth-hart-presents-microstructpy-world>`__,
written by the School of Aerospace Engineering at Georgia Tech,
describes MicroStructPy for a general audience.

.. end-publications

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

.. _`Contributing Guidelines`: https://github.com/kip-hart/MicroStructPy/blob/dev/.github/CONTRIBUTING.md

.. external-images

.. |l-github| image:: https://api.iconify.design/octicon:mark-github.svg?color=black0&inline=true&height=16
    :alt: GitHub

.. |l-rtd| image:: https://api.iconify.design/simple-icons:readthedocs.svg?color=black&inline=true&height=16
    :alt: ReadTheDocs

.. |l-pypi| image:: https://api.iconify.design/mdi:cube-outline.svg?color=black&inline=true&height=16
    :alt: PyPI


.. SHIELDS

.. |s-ci| image:: https://github.com/kip-hart/MicroStructPy/workflows/CI/badge.svg
    :target: https://github.com/kip-hart/MicroStructPy/actions
    :alt: Continuous Integration

.. |s-license| image:: https://img.shields.io/github/license/kip-hart/MicroStructPy
    :target: https://github.com/kip-hart/MicroStructPy/blob/master/LICENSE.rst
    :alt: License

.. |s-doi1| image:: https://img.shields.io/badge/DOI-10.1016%2Fj.cma.2020.113242-blue
   :target: https://doi.org/10.1016/j.cma.2020.113242
   :alt: CMAME DOI

.. |s-doi2| image:: https://img.shields.io/badge/DOI-10.1016%2Fj.softx.2020.100595-blue
   :target: https://doi.org/10.1016/j.softx.2020.100595
   :alt: SoftwareX DOI
