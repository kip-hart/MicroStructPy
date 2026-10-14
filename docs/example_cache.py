#!/usr/bin/env python3
"""Pack and restore the figures the documentation examples produce.

The documentation build runs every example in
``src/microstructpy/examples``, which takes about twenty minutes and does
not fit in the fifteen minute limit of a Read the Docs build on the free
plan.  The work is the same on every build, so this module moves it to a
machine without a time limit: the documentation workflow packs what the
examples produced, and the Read the Docs build restores it before Sphinx
runs.

Two sets of files are needed, and both are in the archive:

* ``docs/source/auto_examples``, the output of Sphinx-Gallery.  It holds
  the ``.md5`` of ``plot_demos.py``, which is what makes Sphinx-Gallery
  skip the examples instead of running them.
* The PNG files under ``src/microstructpy/examples``, which the pages in
  ``docs/source/examples`` embed with ``figure::``.  The meshes beside
  them are not in the archive, since no page refers to them.

An archive carries the digest of the inputs it was built from, from
:mod:`docs.example_digest`.  ``restore`` refuses an archive whose digest
is not the digest of the working tree, so an edited example cannot be
served with the figures of an older one.  Refusing leaves the build to
run the examples itself, which is slow and may time out, and that is the
intended outcome: a build that cannot be warm should be loud, not stale.

Usage::

    python docs/example_cache.py pack examples.tar.gz
    python docs/example_cache.py restore examples.tar.gz
    python docs/example_cache.py name          # archive name for this tree

"""

import argparse
import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

sys.path.insert(0, HERE)
from example_digest import digest  # noqa: E402

GALLERY = os.path.join('docs', 'source', 'auto_examples')
EXAMPLES = os.path.join('src', 'microstructpy', 'examples')
MANIFEST = 'microstructpy-example-cache.json'


def _gallery_members():
    """Every file of the Sphinx-Gallery output, relative to the root."""
    members = []
    root = os.path.join(ROOT, GALLERY)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            members.append(os.path.relpath(path, ROOT))
    return members


def _figure_members():
    """Every PNG an example wrote, relative to the root.

    Only the files in the output directories of the examples, so that an
    input such as ``aluminum_micro.png`` is not carried in the archive.
    """
    members = []
    root = os.path.join(ROOT, EXAMPLES)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue  # the inputs live here, the figures are in subdirs
        for name in sorted(filenames):
            if name.endswith('.png'):
                path = os.path.join(dirpath, name)
                members.append(os.path.relpath(path, ROOT))
    return members


def pack(filename):
    """Write an archive of the figures and the gallery output."""
    gallery = _gallery_members()
    figures = _figure_members()

    if not gallery:
        print('No %s to pack. Build the documentation first.' % GALLERY,
              file=sys.stderr)
        return 1
    if not figures:
        print('No figures under %s to pack.' % EXAMPLES, file=sys.stderr)
        return 1

    value = digest()
    manifest = json.dumps({'digest': value,
                           'gallery': len(gallery),
                           'figures': len(figures)}, indent=2).encode()

    with tarfile.open(filename, 'w:gz') as tar:
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(manifest)
        tar.addfile(info, _BytesIO(manifest))
        for rel in gallery + figures:
            tar.add(os.path.join(ROOT, rel), arcname=rel)

    size = os.path.getsize(filename) / 1048576.0
    print('packed %d gallery files and %d figures for digest %s '
          'into %s (%.1f MB)'
          % (len(gallery), len(figures), value, filename, size))
    return 0


def restore(filename):
    """Extract an archive, if its digest is the digest of this tree."""
    if not os.path.exists(filename):
        print('No archive at %s.' % filename, file=sys.stderr)
        return 1

    with tarfile.open(filename, 'r:gz') as tar:
        try:
            manifest = json.loads(tar.extractfile(MANIFEST).read().decode())
        except KeyError:
            print('%s has no %s, refusing to restore it.'
                  % (filename, MANIFEST), file=sys.stderr)
            return 1

        current = digest()
        if manifest.get('digest') != current:
            print('The archive was built from %s and the examples digest '
                  'to %s. Refusing to restore it, the examples will be run.'
                  % (manifest.get('digest'), current), file=sys.stderr)
            return 1

        members = [m for m in tar.getmembers() if m.name != MANIFEST]
        for m in members:
            if m.name.startswith('/') or '..' in m.name.split('/'):
                print('Refusing a path outside the tree: %s' % m.name,
                      file=sys.stderr)
                return 1
        try:
            tar.extractall(ROOT, members=members, filter='data')
        except TypeError:
            # filter= arrived in Python 3.12. The paths are checked above.
            tar.extractall(ROOT, members=members)

    print('restored %d files for digest %s' % (len(members), current))
    return 0


def name():
    """Print the archive name for this working tree."""
    print('microstructpy-examples-%s.tar.gz' % digest())
    return 0


class _BytesIO(object):
    """A minimal file object, so tarfile can add bytes already in memory."""

    def __init__(self, data):
        self._data = data
        self._pos = 0

    def read(self, size=-1):
        if size < 0:
            chunk = self._data[self._pos:]
        else:
            chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack', help='write an archive of the figures')
    p.add_argument('filename')
    r = sub.add_parser('restore', help='extract an archive of the figures')
    r.add_argument('filename')
    sub.add_parser('name', help='the archive name for this working tree')

    args = parser.parse_args()
    if args.command == 'pack':
        return pack(args.filename)
    if args.command == 'restore':
        return restore(args.filename)
    return name()


if __name__ == '__main__':
    sys.exit(main())
