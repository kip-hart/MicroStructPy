Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_,
and this project adheres to `Semantic Versioning`_.

`Unreleased`_
--------------
Changed
'''''''
- Fixed verification module handling of "random" orientations.
- Moved ellipse of best fit code to separate file, mimicking source.

Removed
'''''''
- Single elliptical grain example.
- Unused ``_misc.tangent_sphere`` function.
- Unused bounding circles properties.
- Unused reflect methods.
- Redundant ``sample_limits`` properties.

`1.1.1`_ - 2019-11-05
---------------------
Added
'''''
- DOI links to readme and documentation.
- Area-based definition to ellipse geometry.
- Flowchart example for standard workflow.
- Unit tests.
- ``position`` property to Ellipse class.
- ``__eq__`` method to Ellipse class.

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

.. _`Unreleased`: https://github.com/kip-hart/MicroStructPy/compare/v1.1.1...HEAD
.. _`1.1.1`: https://github.com/kip-hart/MicroStructPy/compare/v1.1.0...v1.1.1
.. _`1.1.0`: https://github.com/kip-hart/MicroStructPy/compare/v1.0.1...v1.1.0
.. _`1.0.1`: https://github.com/kip-hart/MicroStructPy/compare/v1.0.0...v1.0.1
.. _`1.0.0`: https://github.com/kip-hart/MicroStructPy/releases/tag/v1.0.0

.. _`Keep a Changelog`: https://keepachangelog.com/en/1.0.0/
.. _`Semantic Versioning`: https://semver.org/spec/v2.0.0.html