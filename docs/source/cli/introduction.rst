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

The following pages describe in detail the various uses and options for the
material, domain, and settings fields of a MicroStructPy input file.

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


Including References to Other Input Files
-----------------------------------------

The input file can optionally *include* references to other input files.
For example if the file ``materials.xml`` contains:

.. code-block:: xml

    <input>
        <material>
            <shape> circle </shape>
            <size> 0.1 </size>
        </material>
    </input>

and another file, ``domain_1.xml``, contains:

.. code-block:: xml

    <input>
        <include> materials.xml </include>
        <domain>
            <shape> square </shape>
            <side_length> 10 </side_length>
        </domain>
    </input>

then MicroStructPy will read the contents of ``materials.xml`` when
``microstructpy domain_1.xml`` is called. This functionality can allows multiple
input files to reference the same material properties. For example, a mesh
convergence study could keep the materials and domain definitions in a single
file, then the input files for each mesh size would contain the run settings
and a reference to the definitions file.

This way, if a parameter such as the grain size distribution needs to be
updated, it only needs to be changed in a single file.

Advanced Usage
++++++++++++++

The ``<include>`` tag can be included at any heirarchical level of the
input file. It can also be nested, with ``<include>`` tags in the file being
included. For example, if the file ``fine_grained.xml`` contains:

.. code-block:: xml

    <material>
        <shape> circle </shape>
        <size> 0.1 </size>
    </material>

and the file ``materials.xml`` contains:



.. code-block:: xml

    <input>
        <material>
            <name> Fine 1 </name>
            <include> fine_grained.xml </include>
        </material>

        <material>
            <name> Fine 2 </name>
            <include> fine_grained.xml </include>
        </material>

        <material>
            <name> Coarse </name>
            <shape> circle </shape>
            <size> 0.3 </size>
        </material>
    </input>

and the file ``input.xml`` contains:

.. code-block:: xml

    <input>
        <include> materials.xml </include>
        <domain>
            <shape> square </shape>
            <side_length> 20 </side_length>
        </domain>
    </input>

then running ``microstructpy input.xml`` would be equivalent to running this
file:

.. code-block:: xml

    <input>
        <material>
            <name> Fine 1 </name>
            <shape> circle </shape>
            <size> 0.1 </size>
        </material>

        <material>
            <name> Fine 2 </name>
            <shape> circle </shape>
            <size> 0.1 </size>
        </material>

        <material>
            <name> Coarse </name>
            <shape> circle </shape>
            <size> 0.3 </size>
        </material>

        <domain>
            <shape> square </shape>
            <side_length> 20 </side_length>
        </domain>
    </input>


The ``<include>`` tag can reduce file sizes and the amount of copy/paste for
microstructures with multiple materials of the same size distribution,
or multiple runs with the same material.
