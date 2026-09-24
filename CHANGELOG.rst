Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.

Unreleased
----------
Added
'''''
- Periodic microstructures, in 2D and 3D: the ``<periodic>`` field of the
  domain (or the ``periodic`` argument of ``cli.run``, ``SeedList.position``
  and ``PolyMesh.from_seeds``) selects the periodic axes. Seeds crossing a
  periodic face are placed without overlapping the opposite side, the
  Laguerre tessellation is periodic across those faces (cells crossing a
  face are cut and their pieces tile the domain), and the triangular,
  tetrahedral and raster meshes have matching nodes on opposite faces
  with the quality and size settings (``min_angle``, ``max_volume``, the
  ``max_volume`` of each phase, ``max_edge_length``) acting as on
  non-periodic meshes: in 2D the cells next to the periodic faces are
  copied outside the faces while meshing, so that Triangle refines both
  faces the same way; in 3D the facets are triangulated with the points
  TetGen adds on them in a first pass, identically on opposite faces, and
  the mesh is built again with the facets fixed. The pairs of periodic
  points/nodes and facets
  are stored in the meshes and their text files; the Abaqus output has a
  node set per periodic face in matching order; the verification unwraps
  grains that are split by the faces. The examples ``periodic_2D.xml``,
  ``periodic_3D.xml`` and ``periodic_tiling.py`` demonstrate periodic
  microstructures, ``pbx_2D.xml`` and ``pbx_3D.xml`` a periodic
  particulate composite (crystalline inclusions in a binder), and
  ``pbx_interface_2D.xml`` and ``pbx_interface_3D.xml`` meshes refined at
  the grain boundaries. Cells of the same amorphous phase
  that touch across a periodic face are merged into one region, like cells
  that share a facet, and the merged region is labelled with the smallest
  seed number among its cells by every mesher and writer. The element
  attributes and the facets of periodic meshes are computed from the
  geometry of the polymesh, since TetGen can leave sub-faces of a facet
  unmarked when it may not modify the boundary and its region attributes
  then leak between cells. gmsh is not supported for periodic meshes.
- ``periodic_margin`` (a setting, and an argument of ``SeedList.position``
  and ``cli.run``): the minimum distance between the surface of a seed and
  a periodic face. A seed that ends within the margin of a face, or crosses
  it by less, is placed elsewhere, since it would leave a thin piece of its
  grain on the opposite face and elements much smaller than the target size
  of the mesh there. ``auto`` uses half the target edge length of the mesh,
  or an eighth of the size of the smallest seed if that is smaller (the
  smallest seed needs about four elements across it, and cannot satisfy a
  margin larger than itself).
- ``PolyMesh.from_seeds(edge_opt=True, periodic_margin=...)``: the edge
  optimization of periodic meshes treats the thickness of each piece of a
  cell at a periodic face as a feature like an edge, and moves the seeds of
  the pieces thinner than the margin (and of their neighbors) normal to the
  face until the piece reaches the margin or the cell no longer crosses the
  face. The CLI passes its ``periodic_margin`` setting on. A trial of the
  optimization is now kept when the shortest feature that it changes gets
  longer (for the shortest edge of the mesh, the criterion is unchanged),
  a target that does not improve in ``n_iter`` trials is left alone and the
  next one is taken, and the seeds moved across a periodic face are wrapped
  back into the domain. The CLI writes and plots the seeds again after the
  optimization, so that the seed files match the polygonal mesh. The
  periodic examples use ``periodic_margin`` ``auto`` and ``edge_opt``.
  With ``min_angle`` (the minimum angle of the mesh to be built, passed by
  the CLI from ``mesh_min_angle``), the corners of the cells at the
  periodic faces narrower than that angle are features too, since the
  mesher cannot reach the minimum angle there and fills them with shells of
  very small elements: the seeds on both sides of the facet are moved along
  it to open the corner.
- When the nodes on the periodic faces of a 2D mesh do not match after the
  first pass, the next pass starts from all the points of the mesh (those on
  a periodic face and on its image merged and put on both faces), so that
  Triangle only refines it around the merged points, instead of meshing the
  cells again with the points on the faces only, which split the narrow
  corners of the cells again at every pass, down to very small elements.
  The copies of the cells at the corners of the domain, used while meshing,
  were open on one side and partly discarded by Triangle.

Fixed
'''''
- Seed generation is reproducible: the RNG seed chain no longer depends on
  the (hash-randomized) iteration order of the phase keywords, and
  ``SeedList.from_info`` and ``cli.run`` no longer modify the ``rng_seeds``
  and ``filetypes`` arguments (or their mutable defaults).
- ``<dist_type> cdf </dist_type>`` inputs are no longer distorted when the
  x-values in the CSV file are not evenly spaced (``density=False`` is now
  passed to ``scipy.stats.rv_histogram``); ``pdf`` is accepted as an alias
  of ``histogram``.
- 3D ``mesh_max_volume`` and per-phase ``max_volume`` are now honored by
  TetGen; in 2D a per-phase ``max_volume`` larger than the global value is
  no longer capped, and an infinite ``mesh_max_volume`` is no longer passed
  to Triangle as the (mis-parsed) switch ``ainf``.
- ``Ellipsoid.approximate`` mapped the axes incorrectly for the ordering
  c >= a >= b, so those grains were tessellated with the wrong orientation;
  the b >= c >= a ordering is now sorted explicitly as well.
- Seeds read back from ``seeds.txt`` can be repositioned; ellipsoid seeds
  with a rotation sequence are written in a form that can be read back.
- Cells that intersect a circular or elliptical domain without having a
  vertex inside it are no longer dropped, cells cut twice by the boundary
  are clipped correctly, and the stored areas of clipped cells are correct
  (``PolyMesh.volumes``, ``verification.volume_fractions``).
- ``_segment_cross`` no longer hangs for large coordinate values;
  ``sample_pos_within`` raises instead of looping forever when the position
  distribution does not cover the domain.
- ``cli.plot_tri`` no longer hangs in 3D when a void grain touches the
  boundary of the domain.
- Relative ``<filename>`` and ``<directory>`` paths inside repeated tags
  (e.g. several ``<material>`` blocks) are resolved relative to the input
  file; a top-level ``<include>`` no longer discards materials; values such
  as ``true_cdf.csv`` no longer cause infinite recursion; ``inf`` is parsed
  as a float.
- Verification: ``angle_rad`` inputs are no longer replaced by a uniform
  distribution, ``<orientation> random </orientation>`` and vector-valued
  parameters (``side_lengths``, ``axes``) no longer crash, unknown phase
  fields are ignored, the caller's phases are not modified.
- ``RasterMesh``: elements are counter-clockwise / right-handed (valid for
  Abaqus CPS4/C3D8), facets and their attributes are correct, ``vtk`` and
  ``abaqus`` output work (including with voids), 3D plotting works.
- ``TriMesh.write``: valid ``.ele``/``.edge``/``.face`` files, Abaqus
  exterior surface unions reference only defined surfaces, full-precision
  points in text files, no dangling headers for meshes without attributes.
- ``PolyMesh.from_seeds(edge_opt=True)`` leaves the seed list in the
  accepted state (positions and breakdowns consistent) and is quiet unless
  ``verbose``; ``PolyMesh.write(format='poly')`` writes the file;
  ``PolyMesh.__eq__`` is silent and no longer cubic.
- ``Ellipse(axes=...)``, ``Ellipse(matrix=...)``, ``Ellipsoid(c=..,
  ratio_bc=..)``, ``Square.area_expectation(side_lengths=...)``, the
  ``*_expectation`` methods with numpy scalars, ``Sphere.plot`` and 3D
  ``PolyMesh.plot``/``SeedList.plot_breakdown`` on a fresh figure,
  ``Rectangle.within`` for rotated rectangles, ``reflect`` for ellipses
  and ellipsoids, single-material ``color_by`` settings, numpy arrays as
  per-item plot keywords.

Changed
'''''''
- The continuous integration installs the current pytest and no longer
  installs tox from the requirements: the pinned tox 3.14 forced an old
  pluggy that the current pytest-cov cannot load, so no test could run.
  The jobs of the test matrix no longer cancel each other on a failure.
- The facets of the triangular and tetrahedral meshes created from a
  polygonal mesh are sorted (nodes in ascending order within a facet,
  facets in lexicographic order), whatever the mesher. Triangle and TetGen
  list the edges/faces of a mesh in an order, and with an orientation, that
  vary from one run to the next, so the mesh files of otherwise identical
  runs differed in the order of their facets.
- ``max_edge_length`` (``mesh_max_edge_length``) acts in 3D on the triangles
  of the grain boundaries: when it is set, the facets of the polyhedral mesh
  are triangulated to that edge length (with Triangle, minimum angle 20
  degrees) before TetGen meshes the cells, for periodic and non-periodic
  meshes alike, so that the elements can be smaller at the interfaces than
  inside the grains (see the ``pbx_interface_3D.xml`` example). The
  geometric tests of a mesh against its polymesh use the non-planarity of
  the facets (from the snapping of the points to the periodic faces) as
  their tolerance, and each element is assigned to the cell in which its
  centroid is deepest.
- The overlap tolerance fit ``rtol='fit'`` uses the coefficients published
  in Hart and Rimoli, CMAME 370 (2020) 113242, Eqs. (14) and (15). For very
  wide size distributions this allows less overlap than before (2D
  asymptote 0.18 instead of 0.36), so some seeds of high-cv inputs may be
  rejected during placement.
- ``Ellipsoid.limits`` is exact for rotated ellipsoids (it was sampled).
- A ``Seed`` created with a ``position`` (or a geometry with a center) has
  its breakdown at that position; the geometry center is no longer reset
  to the origin.
- ``Ellipse``, ``Ellipsoid`` and ``NBox`` geometries compare equal when
  their parameters are equal.
- Unused sampling helpers were removed from ``seeding.seedlist``.


`1.5.9`_ - 2023-10-05
--------------------------
Added
'''''
- Support for Python 3.12
  
Changed
'''''''
- Updated version numbers of dependencies per security report.


`1.5.8`_ - 2023-08-25
--------------------------
Added
'''''
- Support for Python 3.11

Fixed
'''''''
- Errors associated with using keyword arguments in ``plt.gca()`` deprecated in matplotlib 3.4.


`1.5.7`_ - 2023-08-06
--------------------------
Fixed
'''''''
- Errors associated with using keyword arguments in ``plt.gca()`` deprecated in matplotlib 3.4.
- 3D plots now use matplotlib standard for setting equal aspect ratios.


`1.5.6`_ - 2022-09-01
--------------------------
Fixed
'''''''
- Bug in RasterMesh VTK writer.

`1.5.5`_ - 2022-08-31
--------------------------
Changed
'''''''
- VTK writer for RasterMesh uses ``RECTILINEAR_GRID`` data type.

`1.5.4`_ - 2022-04-16
--------------------------
Added
'''''
- Support for Python 3.9 and 3.10.

Changed
'''''''
- Dependent versions of NumPy, SciPy, and matplotlib.

Removed
'''''''
- Support for Python 3.7 and below.

Security
''''''''
- Upgraded to NumPy v1.22.0 to address overflow vulnerability.

`1.5.3`_ - 2022-03-30
--------------------------
Fixed
'''''''
- String types in generating Abaqus inp files.

`1.5.2`_ - 2021-09-09
--------------------------
Fixed
'''''''
- Plotting 3D raster meshes.

`1.5.1`_ - 2021-09-09
--------------------------
Fixed
'''''''
- Plaid issue in 2D raster mesh array output.
- Initialization and plotting of 3D raster meshes.

`1.5.0`_ - 2021-09-08
--------------------------
Added
'''''
- RasterMesh class, to create pixel/voxel meshes. (addresses `#44`_)

`1.4.10`_ - 2021-06-08
--------------------------
Fixed
'''''''
- Bug in gmsh for multi-circle seeds.

`1.4.9`_ - 2021-05-14
--------------------------
Fixed
'''''''
- Bug in gmsh for amorphous and void phases.

`1.4.8`_ - 2021-05-13
--------------------------
Fixed
'''''''
- Default behavior of ``cli.plot_*`` functions when ``plot_files`` is not
  specified.

`1.4.7`_ - 2021-02-07
--------------------------
Changed
'''''''
- Updated numpy and matplotlib versions.

Fixed
'''''''
- String parsing errors.

`1.4.6`_ - 2021-02-07
--------------------------
Fixed
'''''''
- String parsing errors.
- Logo example failing on ReadTheDocs.
- 3D gmsh with variable mesh sizes.

`1.4.5`_ - 2020-12-24
--------------------------
Added
'''''''
- Meshing with gmsh can now use different mesh sizes in the interior and on the
  boundary of grains. The ``<mesh_max_edge_length>`` tag specifies edge lengths
  on the boundary and ``<mesh_size>`` on the interior.
  If ``<mesh_max_edge_length>`` is not used, ``<mesh_size>`` is used
  throughout.

`1.4.4`_ - 2020-12-22
--------------------------
Fixed
'''''''
- Reading absolute paths from ``<include>`` tags.

`1.4.3`_ - 2020-11-11
--------------------------
Fixed
'''''''
- PLY file format in 2D.

`1.4.2`_ - 2020-11-3
--------------------------
Fixed
'''''''
- XML parsing text with parentheses.

`1.4.1`_ - 2020-10-13
--------------------------
Changed
'''''''
- Upgraded to pygmsh v7.0.2.

`1.4.0`_ - 2020-10-06
--------------------------
Added
'''''''
- References within XML input files using the ``<include>`` tag.
- Support for gmsh. (addresses `#16`_)
- Citation to SoftwareX publication.

Fixed
'''''''
- Color-by seed number in CLI TriMesh plot function.
- Expansion of "~" in input filepaths.

`1.3.5`_ - 2020-09-20
--------------------------
Fixed
'''''''
- Tetrahedral mesh maximum volume setting no longer ignored.

`1.3.4`_ - 2020-08-31
--------------------------
Removed
'''''''
- Debug print statements from SeedList population fractions method.

`1.3.3`_ - 2020-08-31
--------------------------
Added
'''''
- Helper functions for SeedList class.

Fixed
'''''''
- Dictionary conversion issue with lists of SciPy distributions.
- XML tags in documentation on position distributions.


`1.3.2`_ - 2020-07-11
--------------------------
Added
'''''
- VTK output for 2D triangular meshes.

Changed
'''''''
- Updated reference to CMAME publication.

`1.3.1`_ - 2020-07-09
--------------------------
Added
'''''
- VTK output for seed lists and polyhedral meshes.
- Option to compute expected area of ellipse from area distribution.
- Option to compute expected volume of ellipsoid from volume distribution.

Fixed
'''''
- Error in verification module for 2D uniform random orientations.

`1.3.0`_ - 2020-06-25
--------------------------
Added
'''''
- Option to reduce the presence of short edges in polygonal meshes.

Changed
'''''''
- Optimized seed positioning algorithm by using breadth-first search
  in the AABB tree.
- Facets in polygonal meshes are now always defined with a positive
  outward normal vector.

Fixed
'''''
- Plotting of 3D meshes. 
- Documentation for empirical PDFs.
- Minor errors in examples.

`1.2.2`_ - 2020-05-14
--------------------------
Fixed
'''''
- Matplotlib error with undefined axes.

`1.2.1`_ - 2020-05-14
--------------------------
Changed
'''''''
- Plot methods automatically update figure axes.

Fixed
'''''
- CLI plotting function for triangular/tetrahedral meshes.

`1.2.0`_ - 2020-05-13
--------------------------
Added
'''''
- Options to shorten input keyword argument lists for plot methods
  (addresses `#14`_)

Changed
'''''''
- Ellipse of best fit method calls the `lsq-ellipse`_ package.

Removed
'''''''
- Removed support for Python 2.7.

`1.1.2`_ - 2019-11-07
---------------------
Fixed
'''''
- Paths to demo files in CLI, moved into source directory.

`1.1.1`_ - 2019-11-05
---------------------
Added
'''''
- DOI links to readme and documentation.

Changed
'''''''
- Added logos, icons, social meta data for HTML documentation.

Fixed
'''''
- Paths to demo files in CLI.

`1.1.0`_ - 2019-09-27
---------------------

Added
'''''
- An ``__add__`` method to the SeedList class.

Changed
'''''''
- Project documentation.

`1.0.1`_ - 2019-09-07
---------------------

Changed
'''''''
- Project documentation.
- Made project name lowercase in PyPI.


`1.0.0`_ - 2019-09-07
---------------------

Added
'''''
- Project added to GitHub.



.. LINKS

.. _`Unreleased`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.9...HEAD
.. _`1.5.9`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.8...v1.5.9
.. _`1.5.8`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.7...v1.5.8
.. _`1.5.7`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.6...v1.5.7
.. _`1.5.6`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.5...v1.5.6
.. _`1.5.5`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.4...v1.5.5
.. _`1.5.4`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.3...v1.5.4
.. _`1.5.3`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.2...v1.5.3
.. _`1.5.2`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.1...v1.5.2
.. _`1.5.1`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.0...v1.5.1
.. _`1.5.0`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.10...v1.5.0
.. _`1.4.10`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.9...v1.4.10
.. _`1.4.9`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.8...v1.4.9
.. _`1.4.8`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.7...v1.4.8
.. _`1.4.7`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.6...v1.4.7
.. _`1.4.6`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.5...v1.4.6
.. _`1.4.5`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.4...v1.4.5
.. _`1.4.4`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.3...v1.4.4
.. _`1.4.3`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.2...v1.4.3
.. _`1.4.2`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.1...v1.4.2
.. _`1.4.1`: https://github.com/kip-hart/MicroStructPy/compare/v1.4.0...v1.4.1
.. _`1.4.0`: https://github.com/kip-hart/MicroStructPy/compare/v1.3.5...v1.4.0
.. _`1.3.5`: https://github.com/kip-hart/MicroStructPy/compare/v1.3.4...v1.3.5
.. _`1.3.4`: https://github.com/kip-hart/MicroStructPy/compare/v1.3.3...v1.3.4
.. _`1.3.3`: https://github.com/kip-hart/MicroStructPy/compare/v1.3.2...v1.3.3
.. _`1.3.2`: https://github.com/kip-hart/MicroStructPy/compare/v1.3.1...v1.3.2
.. _`1.3.1`: https://github.com/kip-hart/MicroStructPy/compare/v1.3.0...v1.3.1
.. _`1.3.0`: https://github.com/kip-hart/MicroStructPy/compare/v1.2.2...v1.3.0
.. _`1.2.2`: https://github.com/kip-hart/MicroStructPy/compare/v1.2.1...v1.2.2
.. _`1.2.1`: https://github.com/kip-hart/MicroStructPy/compare/v1.2.0...v1.2.1
.. _`1.2.0`: https://github.com/kip-hart/MicroStructPy/compare/v1.1.2...v1.2.0
.. _`1.1.2`: https://github.com/kip-hart/MicroStructPy/compare/v1.1.1...v1.1.2
.. _`1.1.1`: https://github.com/kip-hart/MicroStructPy/compare/v1.1.0...v1.1.1
.. _`1.1.0`: https://github.com/kip-hart/MicroStructPy/compare/v1.0.1...v1.1.0
.. _`1.0.1`: https://github.com/kip-hart/MicroStructPy/compare/v1.0.0...v1.0.1
.. _`1.0.0`: https://github.com/kip-hart/MicroStructPy/releases/tag/v1.0.0

.. _`Keep a Changelog`: https://keepachangelog.com/en/1.0.0/
.. _`lsq-ellipse`: https://pypi.org/project/lsq-ellipse
.. _`Semantic Versioning`: https://semver.org/spec/v2.0.0.html

.. _`#14`: https://github.com/kip-hart/MicroStructPy/issues/14
.. _`#16`: https://github.com/kip-hart/MicroStructPy/issues/16
.. _`#44`: https://github.com/kip-hart/MicroStructPy/issues/44
