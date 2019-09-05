.. begin-readme

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

.. begin-banner

.. image:: https://microstructpy.readthedocs.io/en/latest/_images/banner.png
    :alt: Banner image showing the three steps for creating microstructure.

.. end-banner

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


MicroStructPy can also be installed from source, hosted on the project GitHub_
page::

    git clone https://github.com/kip-hart/MicroStructPy.git
    pip install -e MicroStructPy/

Installing MicroStructPy creates the command line program ``microstructpy`` and
the Python package ``microstructpy``.
The command line program executes a standard workflow on XML input files,
while the package exposes classes and functions for a customized workflow.

.. end-download-install


Run a Demo
----------

MicroStructPy includes several demo and example files to help new users get
started with the program.
Here is a example XML input file:

.. begin-demo-block

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

.. end-demo-block

You can run this input file from the command line::

    microstructpy --demo=minimal.xml

Alternatively, you can copy the text to a file such as
``my_input.xml`` and run::

    microstructpy my_input.xml

.. demo-midpoint

To build all of the available demos and examples, run::

    microstructpy --demo=all

Note that this may take up to 10 minutes, depending on your system.

.. begin-doc

Documentation
-------------

MicroStructPy documentation is available online at
https://microstructpy.readthedocs.io.

To build a local copy of the documentation, execute the following from the
top-level directory of the MicroStructPy repository::

    pip install tox
    tox -e docs

Once built, the documentation will be in ``docs/build/``.

.. end-doc

Contributing
------------

Contributions to the project are welcome.
Please visit the `repository`_ to clone the source files,
create a pull request, and submit issues.


License and Attributions
------------------------

MicroStructPy is open source and freely availabe under the terms of the the
MIT license.
Copyright for MicroStructPy is held by Georgia Tech Research Corporation.
MicroStructPy is a major part of Kenneth (Kip) Hart's doctoral thesis,
advised by Prof. Julian Rimoli.

.. end-license


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

.. |s-travis| image:: https://travis-ci.org/kip-hart/MicroStructPy.svg?branch=master
    :target: https://travis-ci.org/kip-hart/MicroStructPy
    :alt: Travis CI

.. |s-license| image:: https://img.shields.io/github/license/kip-hart/MicroStructPy
    :target: https://github.com/kip-hart/MicroStructPy/blob/master/LICENSE.rst
    :alt: License

.. end-readme