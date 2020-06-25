Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.

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

.. _`Unreleased`: https://github.com/kip-hart/MicroStructPy/compare/v1.3.0...HEAD
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
