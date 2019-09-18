.. _getting_started:

Getting Started
===============

Download & Installation
-----------------------

To install MicroStructPy, download it from PyPI using::

    pip install microstructpy

This installs both the ``microstructpy`` Python package and the
``microstructpy`` command line interface (CLI).
If there is an error with the install, try to install ``pybind11`` first.
You may need to add the ``--user`` flag, depending on your permissions.

To verify installation of the package, run the following commands::

    python -c 'import microstructpy'
    microstructpy --help

This verifies that 1) the python package has installed correctly and 2) the
command line interface (CLI) has been found by the shell.
If there is an issue with the CLI, the install location may not be in the
PATH variable.
The most likely install location is ``~/.local/bin`` for Mac or Linux machines.
For Windows, it may be in a path similar to
``~\AppData\Roaming\Python\Python36\Scripts\``.

.. note::
    If the install fails and the last several error messages reference
    ``pybind11``, run ``pip install pybind11`` first then install MicroStructPy.

Running Demonstrations
----------------------

MicroStructPy comes with several demonstrations to familiarize users with its
capabilities and options.
A demonstration can be run from the command line by::

    microstructpy --demo=minimal.xml

When a demo is run, the XML input file is copied to the current working
directory.
See :ref:`examples_page` for a full list of available examples and 
demostrations.


Development
-----------

Contributions to MicroStructPy are most welcome.
To download and install the source code for the project::

    git clone https://github.com/kip-hart/MicroStructPy.git
    pip install -e MicroStructPy

MicroStructPy uses tox_ to run tests and build the documentation.
To perform these tests::

    pip install tox
    cd MicroStructPy
    tox

Please use the issue and pull request features of the `GitHub repository`_ 
to report bugs and modify the code.




.. _`GitHub repository`: https://github.com/kip-hart/MicroStructPy
.. _tox: https://tox.readthedocs.io
