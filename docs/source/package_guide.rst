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
    ├─ meshing
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
The following sections describe the meshing methods, the file I/O and plotting
functions, and the format of a material phase dictionary.

The Standard Workflow
---------------------

Below is an input file similar to the :ref:`intro_examples`.
The script that follows will produce the same results as running this script
from the command line interface.

**XML Input File**

.. code-block:: xml

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

**Equivalent Python Script**

.. code-block:: python
	:emphasize-lines: 29-31, 34, 37, 41-43

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

Highlighted are the four principal methods used in generating a microstructure:
:meth:`.SeedList.from_info`,
:meth:`.SeedList.position`,
:meth:`.PolyMesh.from_seeds`,
:meth:`.TriMesh.from_polymesh`.

Meshing Methods
---------------

Laguerre-Voronoi Tessellation
+++++++++++++++++++++++++++++

Polygonal/polyhedral meshes are generated in MicroStructPy using a
Laguerre-Voronoi tessellation, also known as a `Power Diagram`_.
It is conceptually similar to a Voronoi diagram, the difference being that seed
points are weighted rather than unweighted.
In the :meth:`.PolyMesh.from_seeds` method, the center of a seed is consider
a Voronoi seed point and the radius is its weight.

Non-circular seeds are replaced by their breakdown, resulting in
multiple Voronoi cells representing a single grain.
To retrieve all of the cells that represent a single grain, mask the
``seed_numbers`` property of a :class:`.PolyMesh`.

The Laguerre-Voronoi diagram is created by `Voro++`_, which is accessed
using `pyvoro`_.

Unstructured Meshing
++++++++++++++++++++

The triangular/tetrahedral meshes are generated in MicroStructPy using the
`MeshPy`_ package.
It links with `Triangle`_ to create 2D triangular meshes and with `TetGen`_
to create 3D tetrahedral meshes.

A polygonal mesh, :class:`.PolyMesh`, can be converted into an unstructured
mesh using the :meth:`.TriMesh.from_polymesh` method.
Cells of the same seed number are merged before meshing to prevent unnecessary
internal geometry.
Similarly, if the ``material_type`` of a phase is set to ``amorphous``, then
cells of the same phase number are also merged.
Cells with the ``material_type`` set to ``void`` are treated as holes in
MeshPy, resulting in voids in the output mesh.

File I/O & Plot Methods
-----------------------

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
value with a distribution from the SciPy :mod:`scipy.stats` module.
For example:

.. code-block:: python

       import scipy.stats
       size_dist = scipy.stats.uniform(loc=1, scale=0.4)
       phase['size'] = size_dist

The ``max_volume`` option allows for maximum element volume controls to be
phase-specific.


.. _`MeshPy`: https://mathema.tician.de/software/meshpy/
.. _`Power Diagram`: https://en.wikipedia.org/wiki/Power_diagram
.. _`pyvoro`: https://github.com/mmalahe/pyvoro
.. _`TetGen`: http://wias-berlin.de/software/tetgen/
.. _`Triangle`: https://www.cs.cmu.edu/~quake/triangle.html
.. _`Voro++`: http://math.lbl.gov/voro++/
