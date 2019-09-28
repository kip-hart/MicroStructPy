<!-- adapted from pandas .github/CONTRIBUTING.md -->
<!-- https://github.com/pandas-dev/pandas/blob/master/.github/CONTRIBUTING.md -->

# Contributing to MicroStructPy

Whether you are a novice or experienced software developer,
all contributions and suggestions are welcome!

## Getting Started

If you are looking to contribute to the *MicroStructPy* codebase,
the best place to start is the
[GitHub "issues" tab](https://github.com/kip-hart/MicroStructPy/issues).
This is also a great place for filing bug reports and making suggestions for
ways in which we can improve the code and documentation.

## Filing Issues

If you notice a bug in the code or documentation, or have suggestions for
how we can improve either, feel free to create an issue on the
[GitHub "issues" tab](https://github.com/kip-hart/MicroStructPy/issues) using
[GitHub's "issue" form](https://github.com/kip-hart/MicroStructPy/issues/new).
The form contains some questions that will help us best address your issue.

## Contributing to the Codebase

The code is hosted on [GitHub](https://www.github.com/kip-hart/MicroStructPy),
so you will need to use [Git](http://git-scm.com/) to clone the project and
make changes to the codebase.
Once you have obtained a copy of the code, you should create a development
environment that is separate from your existing Python environment so that
you can make and test changes without compromising your own work environment.
Consider using the Python
[venv](https://docs.python.org/3/library/venv.html#module-venv) module to
create a development environment.

Before submitting your changes for review, make sure to check that your
changes do not break any tests. 
Install [tox](https://tox.readthedocs.io) and run it from the top-level project
directory.
Tox will run tests on the code, the documentation, the build, and the code
style.
Please ensure that all tests are passing before pushing to the repository.

Once your changes are ready to be submitted, make sure to push your changes to
GitHub before creating a pull request.
We will review your changes, and you may be asked to make additional changes
before it is finally ready to merge.
However, once it's ready, we will merge it, and you will have successfully
contributed to the codebase!
