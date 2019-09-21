.. _troubleshooting:

Troubleshooting
==============================================================================

This page addresses some problems that may be encountered with MicroStructPy.
If this page does not address your problem, please submit an issue through the
package GitHub_ page.

.. _GitHub: https://github.com/kip-hart/MicroStructPy

Installation
------------------------------------------------------------------------------

These are problems encountered when installing MicroStructPy.

MeshPy fails to install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Problem Description**

When installing the package, either through PyPI as
``pip install microstructpy`` or from the source as ``pip install -e .`` in
the top-level directory, an error message appears during the meshpy install.
The error message indicates that Visual Studio cannot find the pybind11
headers.

**Problem Solution**

Install pybind11 first by running ``pip install pybind11``, then try to
install MicroStructPy.

Command Line Interface
------------------------------------------------------------------------------

These are problems encountered when running ``microstructpy input_file.xml``.

Command not found on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem Description**

The MicroStructPy package installs without a problem, however on running
``microstructpy example_file.xml`` the following message appears::

  microstructpy: command not found

**Problem Solution**

The command line interface (CLI) is install to a directory that is not in
the PATH variable. Check for the CLI in ``~/.local/bin`` and if it is there,
add the following to your ``~/.bash_profile`` file::

  export PATH=$PATH:~/.local/bin

then source the .bash_profile file by running ``source ~/.bash_profile``.

'tkinter' not found on Linux
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem Description**

The MicroStructPy package installs without a problem, however on running
``microstructpy example_file.xml`` the following error is raised::

  ModuleNotFoundError: No module named 'tkinter'

**Problem Solution**

To install ``tkinter`` for Python 3 on Linux, run the following command::

    sudo apt-get install python3-tk

For Python 2, run the following instead::

    sudo apt-get install python-tk

Program quits/segfaults while calculating Voronoi diagram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem Description**

During the calculating Voronoi diagram step, the program either quits or
segfaults.

**Problem Solution**

This issue was experienced while running 32-bit Python with a large number of
seeds. Python ran out of memory addresses and segfaulted. Switching from 32-bit
to 64-bit Python solved the problem.
