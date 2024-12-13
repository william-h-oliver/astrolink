"""
Plotting routines to be used with the AstroLink class.

Author: William H. Oliver <william.hardie.oliver@gmail.com>
License: MIT
"""

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import beta, halfnorm, lognorm

def plotOrderedDensity(c, ax = None):
    """
    Make the AstroLink ordered-density plot.

    Parameters
    ----------
    c : AstroLink
        An instance of the AstroLink class.
    ax : matplotlib.axes._axes.Axes, optional
        The axes on which to plot the ordered density. If not provided, the 
        current axes will be used.

    Returns
    -------
    line : matplotlib.lines.Line2D
        The line object representing the dendrogram.
    polyCollections : list of matplotlib.collections.PolyCollection
        The PolyCollection objects representing the clusters as filled regions 
        on the ordered-density plot.
    """

    # Check if the axes have been provided
    if ax is None: ax = plt.gca()

    # Plot the ordered density
    logRhoOrdered = c.logRho[c.ordering]
    line, = ax.plot(logRhoOrdered, c = 'k', lw = 1, zorder = 2)

    # Show clusters
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    polyCollections = []
    for i, cluster in enumerate(c.clusters[1:]):
        polyCollection = ax.fill_between(range(cluster[0], cluster[1]), logRhoOrdered[cluster[0]:cluster[1]],
                                         facecolor = colours[i%len(colours)], edgecolor = None, lw = 0, zorder = 1)
        polyCollections.append(polyCollection)

    # Axis limits
    ax.set_xlim(0, c.n_samples - 1)
    ax.set_ylim(0, 1)

    # Add labels
    ax.set_xlabel('Ordered Index')
    ax.set_ylabel(r'$\log\hat\rho$')

    return line, polyCollections

def plotProminenceModel(c, ax = None):
    """
    Make the AstroLink prominences plot.

    Parameters
    ----------
    c : AstroLink
        An instance of the AstroLink class.
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the prominences. If not provided, the current 
        axes will be used.

    Returns
    -------
    h : array
        The values of the histogram bins.
    bins : array
        The edges of the bins.
    patches : list of matplotlib.patches.Patch
        The Patch objects representing the histogram bars.
    line : matplotlib.lines.Line2D
        The line object representing the fitted prominence model.
    lineCollection : matplotlib.collections.LineCollection
        The LineCollection object representing the cutoff over the prominences
        histogram.
    """

    # Check if the axes have been provided
    if ax is None: ax = plt.gca()

    # Plot prominences histogram
    bw = 2*np.subtract(*np.percentile(c.prominences[:, 1], [75, 25]))*c.prominences[:, 1].size**(-1/3) # Freedman-Diaconis rule
    h, bins, patches = ax.hist(c.prominences[:, 1], bins = np.arange(np.ceil(c.prominences[:, 1].max()/bw).astype(np.int64) + 1)*bw,
                               density = True, histtype = 'stepfilled', facecolor = mcolors.to_rgba('k', alpha = 0.2), edgecolor = 'k', lw = 1)

    # Plot fitted prominence model
    xs = np.linspace(0, 1, 10**4)
    if c._noiseModel == 'Beta': ys = beta.pdf(xs, c.pFit[1], c.pFit[2])
    elif c._noiseModel == 'Half-normal': ys = halfnorm.pdf(xs, 0, c.pFit[1])
    elif c._noiseModel == 'Log-normal': ys = lognorm.pdf(xs, c.pFit[2], scale = np.exp(c.pFit[1]))
    line, = ax.plot(xs, ys, 'red', lw = 2, alpha = 0.7)

    # Plot cutoff over prominences histogram
    lineCollection = ax.vlines(c.pFit[0], 0, h.max(), color = 'royalblue', ls = 'dashed', lw = 1)

    # Axis limits
    ax.set_xlim(0, min(ax.get_xlim()[1], 1))
    ax.set_ylim(0, ax.get_ylim()[1])

    # Add labels
    ax.set_xlabel(r'Prominences, $p$')
    ax.set_ylabel('Probability Density')

    return h, bins, patches, line, lineCollection