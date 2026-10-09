#!/usr/bin/env python3
"""Digest of everything that changes the output of the documentation examples.

The documentation build runs every example through
``docs/source/sphinx_gallery/plot_demos.py``, which takes about twenty
minutes.  Sphinx-Gallery skips an example whose source file still matches
the ``.md5`` stored next to its generated output, so a build that starts
from the output of an earlier one does no work at all.

``plot_demos.py`` finds the examples with :func:`glob.glob`, and names
none of them, so its own hash does not move when an example is edited or
added, nor when the library that meshes them changes.  Caching on that
hash alone would serve stale figures and never say so.

This module closes that gap.  It digests the example inputs and the
library source, and keeps the result in ``plot_demos.py`` as the value of
``EXAMPLES_DIGEST``.  Editing an example changes the digest, which changes
the hash of ``plot_demos.py``, which makes Sphinx-Gallery run the examples
again.

Usage::

    python docs/example_digest.py            # print the digest
    python docs/example_digest.py --write    # update plot_demos.py
    python docs/example_digest.py --check    # exit 1 if it is out of date

"""

import argparse
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLE_DIR = os.path.join(ROOT, 'src', 'microstructpy', 'examples')
PACKAGE_DIR = os.path.join(ROOT, 'src', 'microstructpy')
DEMOS = os.path.join(HERE, 'source', 'sphinx_gallery', 'plot_demos.py')

#: Inputs of the examples. Everything the examples read.
INPUT_SUFFIXES = ('.xml', '.py', '.csv')

#: The line that holds the digest in plot_demos.py.
DIGEST_RE = re.compile(r"^EXAMPLES_DIGEST = '[0-9a-f]*'$", re.M)


def _digest_files():
    """The files the digest covers, as absolute paths, sorted.

    The example inputs, and the library that turns them into figures. The
    output directories of the examples are not included: they are what the
    digest is about to describe.
    """
    paths = []

    for name in sorted(os.listdir(EXAMPLE_DIR)):
        path = os.path.join(EXAMPLE_DIR, name)
        if os.path.isfile(path) and name.endswith(INPUT_SUFFIXES):
            paths.append(path)

    for dirpath, dirnames, filenames in os.walk(PACKAGE_DIR):
        dirnames.sort()
        if os.path.abspath(dirpath).startswith(os.path.abspath(EXAMPLE_DIR)):
            continue
        for name in sorted(filenames):
            if name.endswith('.py'):
                paths.append(os.path.join(dirpath, name))

    return sorted(set(paths))


def digest():
    """The digest of the example inputs and the library.

    Returns:
        str: The first 16 characters of the SHA-256 of the files, each
        entered by its path relative to the root of the repository and by
        its contents.
    """
    sha = hashlib.sha256()
    for path in _digest_files():
        rel = os.path.relpath(path, ROOT).replace(os.sep, '/')
        sha.update(rel.encode('utf-8'))
        sha.update(b'\0')
        with open(path, 'rb') as f:
            sha.update(f.read())
        sha.update(b'\0')
    return sha.hexdigest()[:16]


def stored_digest():
    """The digest currently written in plot_demos.py, or an empty string."""
    with open(DEMOS, 'r') as f:
        match = DIGEST_RE.search(f.read())
    return match.group(0).split("'")[1] if match else ''


def write_digest(value):
    """Write a digest into plot_demos.py. Returns True if it changed."""
    with open(DEMOS, 'r') as f:
        text = f.read()
    new = DIGEST_RE.sub("EXAMPLES_DIGEST = '%s'" % value, text)
    if new == text:
        return False
    with open(DEMOS, 'w') as f:
        f.write(new)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--write', action='store_true',
                       help='update the digest in plot_demos.py')
    group.add_argument('--check', action='store_true',
                       help='exit 1 if the stored digest is out of date')
    args = parser.parse_args()

    current = digest()

    if args.write:
        if write_digest(current):
            print('updated EXAMPLES_DIGEST to %s' % current)
        else:
            print('EXAMPLES_DIGEST already %s' % current)
        return 0

    if args.check:
        stored = stored_digest()
        if stored == current:
            print('EXAMPLES_DIGEST is up to date (%s)' % current)
            return 0
        print('EXAMPLES_DIGEST is %s, the examples digest to %s.'
              % (stored or 'missing', current), file=sys.stderr)
        print('Run: python docs/example_digest.py --write', file=sys.stderr)
        return 1

    print(current)
    return 0


if __name__ == '__main__':
    sys.exit(main())
