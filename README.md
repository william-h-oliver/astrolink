# Welcome to AstroLink

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/william-h-oliver/astrolink/ci.yml?branch=main)](https://github.com/william-h-oliver/astrolink/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/astrolink/badge/)](https://astrolink.readthedocs.io/)
[![codecov](https://codecov.io/gh/william-h-oliver/astrolink/branch/main/graph/badge.svg)](https://codecov.io/gh/william-h-oliver/astrolink)

## Installation

The Python package `astrolink` can be installed from PyPI:

```
python -m pip install astrolink
```

## Development installation

If you want to contribute to the development of `astrolink`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/william-h-oliver/astrolink.git
cd astrolink
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
