.. _package_guide:

====================
Python Package Guide
====================

The Python package for MicroStructPy includes the following::

    microstructpy
    ├─ cli
    ├─ geometry
    │  ├─ Box
    │  ├─ Cube
    │  ├─ Circle
    │  ├─ Ellipse
    │  ├─ Ellipsoid
    │  ├─ Rectangle
    │  ├─ Square
    │  └─ Sphere
    ├─ seeding
    │  ├─ Seed
    │  └─ SeedList
    └─ meshing
    │  ├─ PolyMesh
    │  └─ TriMesh
    └─ verification

The cli module contains the functions related to the command line interface
(CLI), including converting XML input files into dictionaries.
The geometry module contains classes for seed and domain geometries.
In the seeding package, there is the single Seed class and the SeedList class,
which functions like a Python list but includes some additional methods such
as positioning and plotting the seeds.
Next, the PolyMesh and TriMesh classes are contained in the meshing module.
A PolyMesh can be created from a SeedList and a TriMesh can be created from
a PolyMesh.
Finally, the verification module contains functions to compare the output
PolyMesh and TriMesh with desired microstructural properties.

**This guide explains how to use the MicroStructPy Python package.**
It starts with a script that executes an abbreviated version of the
standard workflow.
The checks, restarts, etc are excluded to show how the principal classes are
used in a workflow.
The following section lists the file I/O and plotting functions, while the last
section explains the formatting of a phase dictionary.

The Standard Workflow
---------------------

Below is an input file similar to the :ref:`intro_examples`.
The script that follows will produce the same results as running this script
from the command line interface.

.. code-block:: xml
	:caption: XML Input File

	<?xml version="1.0" encoding="UTF-8"?>
	<input>
		<material>
			<name> Matrix </name>
			<material_type> matrix </material_type>
			<fraction> 2 </fraction>
			<shape> circle </shape>
			<size>
				<dist_type> uniform </dist_type>
				<loc> 0 </loc>
				<scale> 1.5 </scale>
			</size>
		</material>

		<material>
			<name> Inclusions </name>
			<fraction> 1 </fraction>
			<shape> circle </shape>
			<diameter> 2 </diameter>
		</material>

		<domain>
			<shape> square </shape>
			<side_length> 20 </side_length>
			<corner> (0, 0) </corner>
		</domain>

		<settings>
			<rng_seeds>
				<size> 1 </size>
			</rng_seeds>

			<mesh_min_angle> 25 </mesh_min_angle>
		</settings>
	</input>

.. code-block:: python
	:caption: Equivalent Python Script
	:emphasize-lines: 29, 32, 35, 39

	import matplotlib.pyplot as plt
	import microstructpy as msp
	import scipy.stats

	# Create Materials
	material_1 = {
		'name': 'Matrix',
		'material_type': 'matrix',
		'fraction': 2,
		'shape': 'circle',
		'size': scipy.stats.uniform(loc=0, scale=1.5)
	}

	material_2 = {
		'name': 'Inclusions',
		'fraction': 1,
		'shape': 'circle',
		'diameter': 2
	}

	materials = [material_1, material_2]

	# Create Domain
	domain = msp.geometry.Square(side_length=15, corner=(0, 0))

	# Create List of Un-Positioned Seeds
	seed_area = domain.area
	rng_seeds = {'size': 1}
	seeds = msp.seeding.SeedList.from_info(materials,
										   seed_area,
										   rng_seeds)

	# Position Seeds in Domain
	seeds.position(domain)

	# Create Polygonal Mesh
	pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

	# Create Triangular Mesh
	min_angle = 25
	tmesh = msp.meshing.TriMesh.from_polymesh(pmesh,
											  materials,
											  min_angle)

	# Save txt files
	seeds.write('seeds.txt')
	pmesh.write('polymesh.txt')
	tmesh.write('trimesh.txt')

	# Plot outputs
	seed_colors = ['C' + str(s.phase) for s in seeds]
	seeds.plot(facecolors=seed_colors, edgecolor='k')
	plt.axis('image')
	plt.savefig('seeds.png')
	plt.clf()

	poly_colors = [seed_colors[n] for n in pmesh.seed_numbers]
	pmesh.plot(facecolors=poly_colors, edgecolor='k')
	plt.axis('image')
	plt.savefig('polymesh.png')
	plt.clf()

	tri_colors = [seed_colors[n] for n in tmesh.element_attributes]
	tmesh.plot(facecolors=tri_colors, edgecolor='k')
	plt.axis('image')
	plt.savefig('trimesh.png')
	plt.clf()

Highlighted are the four principal methods used in generating a microstructure.

.. todo- edit the content below

Seeds are given a geometry and a material number,
SeedLists are lists of Seeds,
the PolyMesh can be created from a SeedList,
and finally the TriMesh can be created from a PolyMesh.
This is the flow of information built into the MicroStructPy command line
interface (CLI).
Custom algorithms for seeding or meshing can be implemented using the classes
above and a few key methods.

**The following describes the 3-step process of generating a microstructure
mesh in MicroStructPy**, including the relevant classes and methods.
See :ref:`api-index` for the complete list of MicroStructPy classes.
For examples using the API, see :ref:`Examples <package_examples>`.

0. List of Seed Geometries
--------------------------

The list of seed geometries is a :class:`.SeedList`.
The SeedList can be created from a list of :class:`.Seed` instances, which
each contain a geometry and a phase.

A SeedList can also be generated from a list of material phase dictionaries
and a total seed volume using the :meth:`.SeedList.from_info` class method.
The default seed volume is the volume of the domain.
For more information on how to format the phase information, see the
:ref:`phase_dict_guide` below.

One convenience function is :meth:`.Seed.factory`, which takes in a
geometry name and keyword arguments and returns a Seed with that geometry.


1. Pack Geometries into Domain
------------------------------

The standard domain is a geometry from the :mod:`microstructpy.geometry`.
To pack the geometries into the domain, the centers of the seeds are specified
such that there is a tolerable about of overlap with other seeds, if any.

The standard method for positioning seeds in a domain is
:meth:`.SeedList.position`.
This function updates the :any:`Seed.position` property of each Seed in the
SeedList.
The centers of all the seeds are within the domain geometry.


2. Tessellate the Domain
------------------------

A tessellation of the domain divides its interior into polygonal/polyhedral
cells with no overlap or gaps between them.
This tessellation is stored in a :class:`.PolyMesh` class.
The default method for creating a PolyMesh from a positioned list of seeds and
a domain is :meth:`.PolyMesh.from_seeds`.
This method creates a Voronoi-Laguerre diagram using the `Voro++`_ package.
Note that the only supported 3D domains are cubes and boxes.


3. Unstructured Meshing
-----------------------

Unstructured (triangular or tetrahedral) meshes can be used in finite
element software to analyze the behavior of the microstructure.
Their data are contained in the :class:`.TriMesh` class.
This mesh can be created from a polygonal tessellation using the
:meth:`.TriMesh.from_polymesh` method.
The mesh can be output to several different file formats.

The unstructured meshes are generated using `Triangle`_ in 2D, `TetGen`_ in 3D,
and `MeshPy`_ is the wrapper.


File I/O
--------

There are file read and write functions associated with each of the classes
listed above.

The read methods are:

* :meth:`.SeedList.from_file`
* :meth:`.PolyMesh.from_file`
* :meth:`.TriMesh.from_file`

The write methods are:

* :meth:`.SeedList.write`
* :meth:`.PolyMesh.write`
* :meth:`.TriMesh.write`

The read functions currently only support reading cache text files.
The SeedList only writes to cache text files, while PolyMesh and TriMesh can
output to several file formats.

Plotting
--------

The SeedList, PolyMesh, and TriMesh classes have the following plotting
methods:

* :meth:`.SeedList.plot`
* :meth:`.SeedList.plot_breakdown`
* :meth:`.PolyMesh.plot`
* :meth:`.PolyMesh.plot_facets`
* :meth:`.TriMesh.plot`


These functions operate like the matplotlib ``plt.plot`` function in that
they just plot to the current figure.
You still need to add ``plt.axis('equal')``, ``plt.show()``, etc to format and
view the plots.


.. _phase_dict_guide:

Phase Dictionaries
------------------

Functions with phase information input require a list of dictionaries, one for
each material phase.
The dictionaries should be organized in a manner similar to the example below.

.. code-block:: python

       phase = {
              'name': 'Example Phase',
              'color': 'blue',
              'material_type': 'crystalline',
              'fraction': 0.5,
              'max_volume': 0.1,
              'shape': 'ellipse',
              'size': 1.2,
              'aspect_ratio': 2
       }

The dictionary contains both data about the phase as a whole, such as its
volume fraction and material type, and about the individual grains.
The keywords ``size`` and ``aspect_ratio`` are keyword arguments for defining
an :class:`.Ellipse`, so those are passed through to the Ellipse class when
creating the seeds.
For a non-uniform size (or aspect ratio) distribution, replace the constant
value with a `SciPy statistical distribution`_.
For example:

.. code-block:: python

       import scipy.stats
       size_dist = scipy.stats.uniform(loc=1, scale=0.4)
       phase['size'] = size_dist

The ``max_volume`` option allows for maximum element volume controls to be
phase-specific.


.. _`MeshPy`: https://mathema.tician.de/software/meshpy/
.. _`SciPy statistical distribution`: https://docs.scipy.org/doc/scipy/reference/stats.html
.. _`TetGen`: http://wias-berlin.de/software/tetgen/
.. _`Triangle`: https://www.cs.cmu.edu/~quake/triangle.html
.. _`Voro++`: http://math.lbl.gov/voro++/
