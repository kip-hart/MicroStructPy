Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.

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

.. _`Unreleased`: https://github.com/kip-hart/MicroStructPy/compare/v1.5.0...HEAD
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
