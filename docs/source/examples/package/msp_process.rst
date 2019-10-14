.. _ex_msp_process:

===============================
MicroStructPy Welcome Flowchart
===============================

Python Script
=============

The basename for this file is ``msp_process.py``.
The file can be run using this command::

    microstructpy --demo=msp_process.py

The full text of the file is:

.. literalinclude:: ../../../../examples/msp_process.py
    :language: python

XML File
========

This example first writes an XML file and calls the CLI to produce
the output plots.
This file is saved to ``msp_process/process.xml``.

Breakdown Plot
==============

MicroStructPy does no produce plots of seed breakdowns by default.
To include this plot in the flow chart, the :meth:`.SeedList.plot_breakdown`
method is called, then the plot is formatted and saved.

Flowchart
=========

The flowchart is created by loading the XML file and plots into annotation
boxes, then adding arrows to connect them.
The chart is shown in :numref:`f_ex_msp_process`.

.. _f_ex_msp_process:
.. figure:: ../../../../examples/msp_process/process.png
    :alt: MicroStructPy welcome page flowchart.

    MicroStructPy welcome page flowchart.