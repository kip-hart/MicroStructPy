============
Introduction
============

Using the Command Line Interface
--------------------------------

.. cli-start

The command line interface (CLI) for this package is ``microstructpy``.
This command accepts the names of user-generated files and demonstration files.
Multiple filenames can be specified.

To run demos, you can specify a particular demo file or to run all of them::

    microstructpy --demo=minimal.xml
    microstructpy --demo=all

Demo files are copied to the current working directory and then executed.
Running all of the demonstration files may take several minutes.

User-generated input files can be run in a number of ways::

    microstructpy /path/to/my/input_file.xml
    microstructpy input_1.xml input_2.xml input_3.xml
    microstructpy input_*.xml

Both relative and absolute filepaths are acceptable.

.. cli-end

Command Line Procedure
----------------------

The following tasks are performed by the CLI:

1. Make the output directory, if necessary
2. **Create a list of unpositioned seeds**
3. **Position the seeds in the domain**
4. Save the seeds in a text file
5. Save a plot of the seeds to an image file
6. **Create a polygon mesh from the seeds**
7. Save the mesh to the output directory
8. Save a plot of the mesh to the output directory
9. **Create an unstructured (triangular or tetrahedral) mesh**
10. Save the unstructured mesh
11. Save a plot of the unstructured mesh
12. (optional) Verify the output mesh against the input file.

Intermediate results are saved in steps 4, 7, and 10 to give the option of
restarting the procedure.
The format of the output files can be specified in the input file
(e.g. PNG and/or PDF plots).

Example Input File
------------------

Input files for MicroStructPy must be in XML format.
The three fields of the input file that MicroStructPy looks for are:
``<material>``, ``<domain>``, and ``<settings>`` (optional).
For example:

.. literalinclude:: ../../../src/microstructpy/examples/minimal_paired.xml
    :language: xml


This will create a microstructure with approximately circular grains that
fill a domain that is 11x larger and color them according to the colormap
Paired.

.. note::

    XML fields that are not recognized by MicroStructPy will be ignored by the
    software. For example, material properties or notes can be included in the
    input file without affecting program execution.

.. note::

    The order of fields in the XML input file is not strictly important,
    since the file is converted into a Python dictionary.
    When fields are repeated, such as including multiple materials, the order
    is preserved.

The following pages describe in detail the various uses and options for the
material, domain, and settings fields of a MicroStructPy input file.