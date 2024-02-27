# Welcome to AstroLink

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/william-h-oliver/astrolink/ci.yml?branch=main)](https://github.com/william-h-oliver/astrolink/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/astrolink/badge/)](https://astrolink.readthedocs.io/)
[![codecov](https://codecov.io/gh/william-h-oliver/astrolink/branch/main/graph/badge.svg)](https://codecov.io/gh/william-h-oliver/astrolink)

AstroLink is a general purpose clustering algorithm built for extracting meaningful hierarchical structure from astrophysical data sets. In practice AstroLink rarely requires any parameter tuning before application, nevertheless, it has a small number intuitive-to-adjust parameters should this be necessary. As such, it is readily capable of finding an arbitary number of arbitrarily shaped clusters (and their structural relationship within the broader hierarchy) from arbitrarily defined data sets.

Clusters found by AstroLink are defined as being statistically distinct overdensities when compared to their surrounds and to the noisy density fluctuations within the data set.

## Installation

The Python package `astrolink` can be installed from PyPI:

```
python -m pip install astrolink
```

## Basic Usage

AstroLink can be easily applied to any point-based input data expressed as a `np.ndarray` with shape `(n_samples, d_features)`.

```
from astrolink import AstroLink
from sklearn.datasets import make_blobs

P, _ = make_blobs(10**4)    # P consists of 10**4 points in 2-dimensions

clusterer = AstroLink(P)
clusterer.run()
```

For low-dimensional input data, such as in this example, it is then possible to visualise the estimated density field by plotting the input data and colouring it by the `logRho` attribute.

```
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lscm

cm = lscm.from_list('density', [(0, 'royalblue'), (1, 'red')])

d_field = plt.scatter(P[:, 0], P[:, 1], c = clusterer.logRho, cmap = cm)
plt.colorbar(d_field, label = r'$\log\hat\rho$')
plt.show()
```

Regardless of the dimensionality of the input data, the clustering structure within it can always be visualised via the 2-dimensional AstroLink ordered-density plot.

```
plt.plot(range(clusterer.n_samples), clusterer.logRho[clusterer.ordering])
plt.show()
```

Although, since the input data in this example can be easily visualised as well, we may as well view this alongside the clusters themselves (as predicted by AstroLink).

```
# Create two figures
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# Make the ordered-density plot
ordered_density = clusterer.logRho[clusterer.ordering]
ax1.plot(range(clusterer.n_samples), ordered_density, c = 'k', zorder = 2)

# Colour the ordered-density and input data plots by each cluster
for i, (clst, clst_id) in enumerate(zip(clusterer.clusters, clusterer.ids)):
    clst_indices = clusterer.ordering[clst[0]:clst[1]]
    clst_ordered_density = clusterer.logRho[clst_indices]
    ax1.fill_between(range(clst[0], clst[1]), clst_ordered_density, color = f"C{i}",
                     zorder = 1)

    clst_P = P[clst_indices]
    ax2.scatter(clst_P[:, 0], clst_P[:, 1], facecolors = f"C{i}", edgecolors = 'k',
                lw = 0.1, label = clst_id)

ax1.set_xlim(0, clusterer.n_samples - 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel('Ordered Index')
ax1.set_ylabel(r'$\log\hat\rho$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_aspect('equal')
ax2.legend()
plt.show()
```

It is worth noting here that AstroLink always returns a cluster that is equal to the entire input data (with ID `'1'` by default) which allows it to be (re-)applied to a disjoint data set in a modular fashion.

To do further analysis on the clustering output, the user may wish to which points (with respect to the order in which they appear within the input data) belong to the clusters that AstroLink has found. These sets can be constructed from the `ordering` and the `clusters` attributes.

```
cluster_members = [clusterer.ordering[clst[0]:clst[1]] for clst in clusterer.clusters]
```

## Development installation

If you want to contribute to the development of `astrolink`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/william-h-oliver/astrolink.git
cd astrolink
python -m pip install --editable .[tests]
```

Having done so, the test suite can then be run using `pytest`:

```
python -m pytest
```

## Citing

If you have used AstroLink in a scientific publication, please use the following citation:

```
@misc{Oliver2023,
      title={The Hierarchical Structure of Galactic Haloes: Differentiating Clusters from Stochastic Clumping with AstroLink}, 
      author={William H. Oliver and Pascal J. Elahi and Geraint F. Lewis and Tobias Buck},
      year={2023},
      eprint={2312.14632},
      archivePrefix={arXiv},
      primaryClass={astro-ph.GA}
}
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
