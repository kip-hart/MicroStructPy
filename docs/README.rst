Building the documentation
==========================

::

    pip install -r docs/requirements.txt
    pip install -r requirements.txt
    pip install -e .
    sphinx-build -Wnb html docs/source docs/build-html

The first build takes about twenty minutes, because
``docs/source/sphinx_gallery/plot_demos.py`` runs every example in
``src/microstructpy/examples`` to produce the figures the pages embed.
Later builds reuse that work and take seconds.


The example cache
-----------------

A build on Read the Docs is stopped at fifteen minutes on the free plan,
which is less than the examples take. The documentation workflow runs
them on a machine without that limit and publishes what they produced,
and the Read the Docs build restores it instead of running them.

Two pieces make that safe.

``docs/example_digest.py`` digests the example inputs and the library
source, and keeps the result in ``plot_demos.py`` as ``EXAMPLES_DIGEST``.
Sphinx-Gallery decides whether to run a script by hashing it, and
``plot_demos.py`` finds the examples with ``glob`` rather than naming
them, so without this its hash would not move when an example was edited
or added, and a cached build would serve stale figures. Editing an
example now changes the digest, which changes the hash, which runs the
examples again.

``docs/example_cache.py`` packs and restores the figures. An archive
carries the digest it was built from, and ``restore`` refuses one that
does not match the working tree. A build that cannot be warm runs the
examples and may time out, which is intended: it is better to be slow and
correct than fast and stale.

After editing an example or anything in ``src/microstructpy``::

    python docs/example_digest.py --write

The documentation check runs ``python docs/example_digest.py --check`` and
fails if the stored digest is out of date.


Publishing a cache by hand
--------------------------

The workflow does this on every pull request, from the ``html`` job. To do
it manually, after a full build::

    name=$(python docs/example_cache.py name)
    python docs/example_cache.py pack "$name"
    gh release upload docs-cache "$name" --clobber

Read the Docs fetches the archive for the digest of the commit it is
building, so a commit whose examples and library are unchanged reuses the
archive of an earlier one.

A pull request that changes an example has no cache for its new digest
until its ``html`` job finishes. A Read the Docs build that starts before
that will run the examples and time out. Rebuild it once the job has
published the archive.
