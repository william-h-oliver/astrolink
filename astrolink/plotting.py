"""
Plotting routines to be used with the AstroLink class.

Author: William H. Oliver <william.hardie.oliver@gmail.com>
License: MIT
"""

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm, beta, halfnorm, lognorm

def plotOrderedDensity(clusterer, ax = None):
    """
    Make the AstroLink ordered-density plot.

    Parameters
    ----------
    clusterer : AstroLink
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
    logRhoOrdered = clusterer.logRho[clusterer.ordering]
    line, = ax.plot(logRhoOrdered, c = 'k', lw = 1, zorder = 2)

    # Show clusters
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    polyCollections = []
    for i, cluster in enumerate(clusterer.clusters[1:]):
        polyCollection = ax.fill_between(range(cluster[0], cluster[1]), logRhoOrdered[cluster[0]:cluster[1]],
                                         facecolor = colours[i%len(colours)], edgecolor = None, lw = 0, zorder = 1)
        polyCollections.append(polyCollection)

    # Axis limits
    ax.set_xlim(0, clusterer.n_samples - 1)
    ax.set_ylim(0, 1)

    # Add labels
    ax.set_xlabel('Ordered Index')
    ax.set_ylabel(r'$\log\hat\rho$')

    return line, polyCollections

def plotAggregationTree(clusterer, ax = None):
    """
    Make the AstroLink aggregation tree plot.

    Parameters
    ----------
    clusterer : AstroLink
        An instance of the AstroLink class.
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the aggregation tree. If not provided, the 
        current axes will be used.
    """

    # Check if the axes have been provided
    if ax is None: ax = plt.gca()

    # Plot the ordered density
    logRhoOrdered = clusterer.logRho[clusterer.ordering]
    ax.plot(logRhoOrdered, c = 'k', lw = 1, zorder = 2)

    offSet = 0.02
    yPos_leq = clusterer.groups[:, 2] - 1
    yPos_geq = clusterer.groups[:, 1] - 1
    heights = np.column_stack((logRhoOrdered[yPos_leq], logRhoOrdered[yPos_geq])).mean(axis = 1) - offSet 

    # Span
    ax.hlines(heights, clusterer.groups[:, 1], yPos_leq, colors = 'red', linewidth = 0.8, zorder = 3)
    ax.hlines(heights, clusterer.groups[:, 0], yPos_geq, colors = 'royalblue', linewidth = 0.8, zorder = 2)
    
    # Ends
    for i in range(2):
        xPos_leq = clusterer.groups[:, i + 1] if i == 0 else clusterer.groups[:, i + 1] - 1
        ax.vlines(xPos_leq, heights, heights + 0.75*offSet, colors = 'red', linewidth = 0.8, zorder = 3)
        xPos_geq = clusterer.groups[:, i] if i == 0 else clusterer.groups[:, i] - 1
        ax.vlines(xPos_geq, heights, heights + 0.75*offSet, colors = 'royalblue', linewidth = 0.8, zorder = 2)
    
    # Joins
    for i, group in enumerate(clusterer.groups):
        parent_leq = next((i - j - 1 for j, g in enumerate(clusterer.groups[:i][::-1]) if group[1] > g[1] and group[2] <= g[2]), None)

        possible_parents = np.where(np.logical_and(clusterer.groups[:, 0] <= group[0], clusterer.groups[:, 1] > group[1]))[0]
        if possible_parents.size:
            parent_geq = possible_parents[np.diff(clusterer.groups[possible_parents, :2], axis = -1).argmin()]
            yLower_geq = heights[parent_geq]
        else: yLower_geq = 0
        parent_geq = next((i - j - 1 for j, g in enumerate(clusterer.groups[:i][::-1]) if group[0] >= g[0] and group[1] <= g[1]), None)

        yLower_leq = heights[parent_leq] if parent_leq is not None else 0
        yLower = max(yLower_leq, yLower_geq)
        ax.vlines((group[1] + group[2] - 1)/2, yLower, heights[i], colors = 'red', linestyles = (0, (1, 0.5)), linewidth = 0.5, zorder = 2)
        ax.vlines((group[0] + group[1] - 1)/2, yLower, heights[i], colors = 'royalblue', linestyles = (0, (1, 0.5)), linewidth = 0.5, zorder = 1)

    # Axis limits
    ax.set_xlim(0, clusterer.n_samples - 1)
    ax.set_ylim(0, 1)

    # Add labels
    ax.set_xlabel('Ordered Index')
    ax.set_ylabel(r'$\log\hat\rho$')

    # Add legend
    offSetx, offSety = 0.02, 0.03
    ax.text(0.8825 - offSetx, 0.99 - offSety, r'$G_\geq$', ha = 'left', va = 'top', color = 'royalblue', zorder = 3, transform = ax.transAxes)
    ax.text(0.9505 - offSetx, 0.99 - offSety, ', ', ha = 'right', va = 'top', zorder = 3, transform = ax.transAxes)
    ax.text(0.99 - offSetx, 0.99 - offSety, r'$G_\leq$', ha = 'right', va = 'top', color = 'red', zorder = 3, transform = ax.transAxes)

def plotProminenceModel(clusterer, ax = None):
    """
    Make a plot of AstroLink prominences.

    Parameters
    ----------
    clusterer : AstroLink
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
    bw = 2*np.subtract(*np.percentile(clusterer.prominences[:, 1], [75, 25]))*clusterer.prominences[:, 1].size**(-1/3) # Freedman-Diaconis rule
    h, bins, patches = ax.hist(clusterer.prominences[:, 1], bins = np.arange(np.ceil(clusterer.prominences[:, 1].max()/bw).astype(np.int64) + 1)*bw,
                               density = True, histtype = 'stepfilled', facecolor = mcolors.to_rgba('k', alpha = 0.2), edgecolor = 'k', lw = 1)

    # Plot fitted prominence model
    xs = np.linspace(0, 1, 10**4)
    if clusterer._noiseModel == 'Beta': ys = beta.pdf(xs, clusterer.pFit[1], clusterer.pFit[2])
    elif clusterer._noiseModel == 'Half-normal': ys = halfnorm.pdf(xs, 0, clusterer.pFit[1])
    elif clusterer._noiseModel == 'Log-normal': ys = lognorm.pdf(xs, clusterer.pFit[2], scale = np.exp(clusterer.pFit[1]))
    line, = ax.plot(xs, ys, 'red', lw = 2, alpha = 0.7)

    # Plot cutoff over prominences histogram
    lineCollection = ax.vlines(clusterer.pFit[0], 0, h.max(), color = 'royalblue', ls = 'dashed', lw = 1)

    # Axis limits
    ax.set_xlim(0, min(ax.get_xlim()[1], 1))
    ax.set_ylim(0, ax.get_ylim()[1])

    # Add labels
    ax.set_xlabel(r'Prominences, $p_{g_\leq}$')
    ax.set_ylabel('Probability Density')

    return h, bins, patches, line, lineCollection

def plotSignificanceModel(clusterer, ax = None):
    """
    Make a plot of AstroLink significances.

    Parameters
    ----------
    clusterer : AstroLink
        An instance of the AstroLink class.
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the significances. If not provided, the 
        current axes will be used.

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

    # Plot group significances histogram
    finiteSigs = clusterer.groups_sigs[np.isfinite(clusterer.groups_sigs[:, 1]), 1]
    h, bins, patches = ax.hist(finiteSigs, bins = 'fd', density = True, histtype = 'stepfilled',
                               facecolor = mcolors.to_rgba('k', alpha = 0.2), edgecolor = 'k', lw = 1)

    # Plot fitted prominence model
    xs = np.linspace(np.floor(finiteSigs.min()), np.ceil(finiteSigs.max()), 10**4)
    ys = norm.pdf(xs)
    line, = ax.plot(xs, ys, 'red', lw = 2, alpha = 0.7)

    # Plot cutoff over prominences histogram
    lineCollection = ax.vlines(clusterer.S, 0, h.max(), color = 'royalblue', ls = 'dashed', lw = 1)

    # Add labels
    ax.set_xlabel(r'Significances, $S_{g_\leq}$')
    ax.set_ylabel('Probability Density')

    return h, bins, patches, line, lineCollection

def plotLogRhoOnX(clusterer, X, ax = None, colorbar = True):
    """
    Make a plot of the data X coloured by the AstroLink log-density on the data.

    Parameters
    ----------
    clusterer : AstroLink
        An instance of the AstroLink class.
    X : array
        The data to be plotted.
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the data. If not provided, the current axes 
        will be used.
    """

    # Check if the axes have been provided
    if ax is None: ax = plt.gca()

    # Colour map that shows low/high values in blue/red
    cmap = mcolors.LinearSegmentedColormap.from_list('density', [(0, 'royalblue'), (1, 'red')])

    # Plot the data
    densityField = ax.scatter(*X.T, c = clusterer.logRho, cmap = cmap)

    # Add colour bar
    if colorbar: plt.colorbar(densityField, label = r'$\log\hat\rho$', ax = ax)