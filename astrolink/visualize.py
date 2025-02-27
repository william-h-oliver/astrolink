"""
Plotting routines to be used with the AstroLink class.

Author: William H. Oliver <william.hardie.oliver@gmail.com>
License: MIT
"""

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm, beta

def orderedDensity(clusterer, skipZeroth = True, ax = None, plotKwargs = {}, fillKwargs = {}):
    """
    Make the AstroLink ordered-density plot.

    Parameters
    ----------
    clusterer : AstroLink
        An instance of the AstroLink class.
    ax : matplotlib.axes._axes.Axes, optional
        The axes on which to plot the ordered density. If not provided, the 
        current axes will be used.
    plotKwargs : dict, optional
        Keyword arguments to be passed to the plot function that makes the 
        ordered-density function.
    fillKwargs : dict, optional
        Keyword arguments to be passed to the fill_between function that 
        highlights the clusters.

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

    # Set default plot and fill kwargs
    if ('c' not in plotKwargs) and ('color' not in plotKwargs): plotKwargs['c'] = 'k'
    if ('lw' not in plotKwargs) and ('linewidth' not in plotKwargs): plotKwargs['lw'] = 1
    if 'zorder' not in plotKwargs: plotKwargs['zorder'] = 2
    if 'fc' in fillKwargs: del fillKwargs['fc']
    if 'facecolor' in fillKwargs: del fillKwargs['facecolor']
    if 'color' in fillKwargs: del fillKwargs['color']
    if ('edgecolor' not in fillKwargs) and ('ec' not in fillKwargs): fillKwargs['edgecolor'] = None
    if ('lw' not in fillKwargs) and ('linewidth' not in fillKwargs): fillKwargs['lw'] = 0
    if 'label' in fillKwargs: del fillKwargs['label']
    if ('zorder' not in fillKwargs): fillKwargs['zorder'] = 1

    # Plot the ordered density
    logRhoOrdered = clusterer.logRho[clusterer.ordering]
    line, = ax.plot(logRhoOrdered, **plotKwargs)

    # Set the starting index for cluster labels to plot
    if skipZeroth: startIndex = 1
    else: startIndex = 0

    # Show clusters
    colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    polyCollections = []
    for i, (clst, clstID) in enumerate(zip(clusterer.clusters[startIndex:], clusterer.ids[startIndex:])):
        polyCollection = ax.fill_between(range(clst[0], clst[1]), logRhoOrdered[clst[0]:clst[1]],
                                         facecolor = colours[i%len(colours)], label = clstID, **fillKwargs)
        polyCollections.append(polyCollection)

    # Axis limits
    ax.set_xlim(0, clusterer.n_samples - 1)
    ax.set_ylim(0, 1)

    # Add labels
    ax.set_xlabel('Ordered Index')
    ax.set_ylabel(r'$\log\hat\rho$')

    return line, polyCollections

def aggregationTree(clusterer, ax = None, plotKwargs = {}, treeKwargs = {}):
    """
    Make the AstroLink aggregation tree plot.

    Parameters
    ----------
    clusterer : AstroLink
        An instance of the AstroLink class.
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the aggregation tree. If not provided, the 
        current axes will be used.
    plotKwargs : dict, optional
        Keyword arguments to be passed to the plot function that makes the 
        ordered-density function.
    treeKwargs : dict, optional
        Keyword arguments to be passed to the hlines and vlines functions that 
        make the aggregation tree.
    """

    # Check if the axes have been provided
    if ax is None: ax = plt.gca()

    # Set default plot, line, and text kwargs
    if ('c' not in plotKwargs) and ('color' not in plotKwargs): plotKwargs['c'] = 'k'
    if ('lw' not in plotKwargs) and ('linewidth' not in plotKwargs): plotKwargs['lw'] = 1
    if 'zorder' not in plotKwargs: plotKwargs['zorder'] = 1
    if 'color' in treeKwargs: del treeKwargs['color']
    if 'colors' in treeKwargs: del treeKwargs['colors']
    if ('lw' not in treeKwargs) and ('linewidth' not in treeKwargs): treeKwargs['lw'] = 0.8
    if 'ls' in treeKwargs: del treeKwargs['ls']
    if 'linestyle' in treeKwargs: del treeKwargs['linestyle']
    if 'linestyles' in treeKwargs: del treeKwargs['linestyles']

    # Plot the ordered density
    logRhoOrdered = clusterer.logRho[clusterer.ordering]
    ax.plot(logRhoOrdered, **plotKwargs)

    # Plot the aggregation tree
    offSet = 0.02
    yPos_geq = clusterer.groups[:, 1] - 1
    yPos_leq = clusterer.groups[:, 2] - 1
    heights = np.column_stack((logRhoOrdered[yPos_geq], logRhoOrdered[yPos_leq])).mean(axis = 1) - offSet

    # Spans
    ax.hlines(heights, clusterer.groups[:, 0], yPos_geq, colors = 'royalblue', zorder = 2, label = r'$G_\geq$', **treeKwargs)
    ax.hlines(heights, clusterer.groups[:, 1], yPos_leq, colors = 'red', zorder = 3, label = r'$G_\leq$', **treeKwargs)
    
    # Ends
    for i in range(2):
        xPos_geq = clusterer.groups[:, i] if i == 0 else clusterer.groups[:, i] - 1
        ax.vlines(xPos_geq, heights, heights + 0.75*offSet, colors = 'royalblue', zorder = 2, **treeKwargs)
        xPos_leq = clusterer.groups[:, i + 1] if i == 0 else clusterer.groups[:, i + 1] - 1
        ax.vlines(xPos_leq, heights, heights + 0.75*offSet, colors = 'red', zorder = 3, **treeKwargs)
    
    # Joins
    for i, group in enumerate(clusterer.groups):
        # Find the height of the parent group of g_leq
        parent_leq = next((i - j - 1 for j, g in enumerate(clusterer.groups[:i][::-1]) if group[1] > g[1] and group[2] <= g[2]), None)
        yLower_leq = heights[parent_leq] if parent_leq is not None else 0

        # Find the height of the parent group of g_geq
        possible_parents = np.where(np.logical_and(clusterer.groups[:, 0] <= group[0], clusterer.groups[:, 1] > group[1]))[0]
        if possible_parents.size:
            parent_geq = possible_parents[np.diff(clusterer.groups[possible_parents, :2], axis = -1).argmin()]
            yLower_geq = heights[parent_geq]
        else: yLower_geq = 0

        # Find the height the actual parent group
        yLower = max(yLower_leq, yLower_geq)

        # Plot the joining lines
        ax.vlines((group[1] + group[2] - 1)/2, yLower, heights[i], colors = 'red', linestyles = (0, (1, 0.5)), zorder = 2, **treeKwargs)
        ax.vlines((group[0] + group[1] - 1)/2, yLower, heights[i], colors = 'royalblue', linestyles = (0, (1, 0.5)), zorder = 1, **treeKwargs)

    # Axis limits
    ax.set_xlim(0, clusterer.n_samples - 1)
    ax.set_ylim(0, 1)

    # Add labels
    ax.set_xlabel('Ordered Index')
    ax.set_ylabel(r'$\log\hat\rho$')

def prominenceModel(clusterer, ax = None, histKwargs = {}, modelKwargs = {}, cutoffKwargs = {}):
    """
    Make a plot of AstroLink prominences.

    Parameters
    ----------
    clusterer : AstroLink
        An instance of the AstroLink class.
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the prominences. If not provided, the current 
        axes will be used.
    histKwargs : dict, optional
        Keyword arguments to be passed to the hist function.
    modelKwargs : dict, optional
        Keyword arguments to be passed to the plot function for the fitted 
        prominence model.
    cutoffKwargs : dict, optional
        Keyword arguments to be passed to the vlines function for the cutoff 
        value.

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

    # Set default histogram, model, and cutoff kwargs
    histKwargs['density'] = True
    histKwargs['histtype'] = 'stepfilled'
    if ('color' not in histKwargs) and ('fc' not in histKwargs) and ('facecolor' not in histKwargs): histKwargs['facecolor'] = np.array([mcolors.to_rgba('k', alpha = 0.2)])
    if ('color' not in histKwargs) and ('ec' not in histKwargs) and ('edgecolor' not in histKwargs): histKwargs['edgecolor'] = 'k'
    if ('lw' not in histKwargs) and ('linewidth' not in histKwargs): histKwargs['lw'] = 1
    if ('c' not in modelKwargs) and ('color' not in modelKwargs): modelKwargs['c'] = 'red'
    if ('lw' not in modelKwargs) and ('linewidth' not in modelKwargs): modelKwargs['lw'] = 2
    if ('alpha' not in modelKwargs): modelKwargs['alpha'] = 0.7
    if ('color' not in cutoffKwargs) and ('colors' not in cutoffKwargs): cutoffKwargs['color'] = 'royalblue'
    if ('ls' not in cutoffKwargs) and ('linestyle' not in cutoffKwargs) and ('linestyles' not in cutoffKwargs): cutoffKwargs['ls'] = 'dashed'
    if ('lw' not in cutoffKwargs) and ('linewidth' not in cutoffKwargs): cutoffKwargs['lw'] = 1

    # Plot prominences histogram
    bw = 2*np.subtract(*np.percentile(clusterer.prominences[:, 1], [75, 25]))*clusterer.prominences[:, 1].size**(-1/3) # Freedman-Diaconis rule
    h, bins, patches = ax.hist(clusterer.prominences[:, 1], bins = np.arange(np.ceil(clusterer.prominences[:, 1].max()/bw).astype(np.int64) + 1)*bw, **histKwargs)

    # Plot fitted prominence model
    xs = np.linspace(0, clusterer.prominences[:, 1].max(), 10**4)
    ys = beta.pdf(xs, clusterer.pFit[0], clusterer.pFit[1])
    line, = ax.plot(xs, ys, **modelKwargs)

    # Plot cutoff over prominences histogram
    cutoff = beta.isf(1 - norm.cdf(3)**(1/clusterer.prominences.shape[0]), clusterer.pFit[0], clusterer.pFit[1])
    lineCollection = ax.vlines(cutoff, 0, h.max(), **cutoffKwargs)

    # Axis limits
    ax.set_xlim(0, min(ax.get_xlim()[1], 1))
    ax.set_ylim(0, ax.get_ylim()[1])

    # Add labels
    ax.set_xlabel(r'Prominences, $p_{g_\leq}$')
    ax.set_ylabel('Probability Density')

    return h, bins, patches, line, lineCollection

def significanceModel(clusterer, ax = None, histKwargs = {}, modelKwargs = {}, cutoffKwargs = {}):
    """
    Make a plot of AstroLink significances.

    Parameters
    ----------
    clusterer : AstroLink
        An instance of the AstroLink class.
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the significances. If not provided, the 
        current axes will be used.
    histKwargs : dict, optional
        Keyword arguments to be passed to the hist function.
    modelKwargs : dict, optional
        Keyword arguments to be passed to the plot function for the fitted 
        significance model.
    cutoffKwargs : dict, optional
        Keyword arguments to be passed to the vlines function for the cutoff 
        value.

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

    # Set default histogram, model, and cutoff kwargs
    histKwargs['density'] = True
    histKwargs['histtype'] = 'stepfilled'
    if 'bins' not in histKwargs: histKwargs['bins'] = 'fd'
    if ('color' not in histKwargs) and ('fc' not in histKwargs) and ('facecolor' not in histKwargs): histKwargs['facecolor'] = np.array([mcolors.to_rgba('k', alpha = 0.2)])
    if ('color' not in histKwargs) and ('ec' not in histKwargs) and ('edgecolor' not in histKwargs): histKwargs['edgecolor'] = 'k'
    if ('lw' not in histKwargs) and ('linewidth' not in histKwargs): histKwargs['lw'] = 1
    if ('c' not in modelKwargs) and ('color' not in modelKwargs): modelKwargs['c'] = 'red'
    if ('lw' not in modelKwargs) and ('linewidth' not in modelKwargs): modelKwargs['lw'] = 2
    if ('alpha' not in modelKwargs): modelKwargs['alpha'] = 0.7
    if ('color' not in cutoffKwargs) and ('colors' not in cutoffKwargs): cutoffKwargs['color'] = 'royalblue'
    if ('ls' not in cutoffKwargs) and ('linestyle' not in cutoffKwargs) and ('linestyles' not in cutoffKwargs): cutoffKwargs['ls'] = 'dashed'
    if ('lw' not in cutoffKwargs) and ('linewidth' not in cutoffKwargs): cutoffKwargs['lw'] = 1

    # Plot group significances histogram
    finiteSigs = clusterer.groups_sigs[np.isfinite(clusterer.groups_sigs[:, 1]), 1]
    h, bins, patches = ax.hist(finiteSigs, **histKwargs)

    # Plot fitted prominence model
    xs = np.linspace(np.floor(finiteSigs.min()), np.ceil(finiteSigs.max()), 10**4)
    ys = norm.pdf(xs)
    line, = ax.plot(xs, ys, **modelKwargs)

    # Plot cutoff over prominences histogram
    if clusterer.S == 'auto':
        S = norm.isf(1 - norm.cdf(3)**(1/clusterer.prominences.shape[0]))
        lineCollection = ax.vlines(S, 0, h.max(), **cutoffKwargs)
    else: lineCollection = ax.vlines(clusterer.S, 0, h.max(), **cutoffKwargs)

    # Add labels
    ax.set_xlabel(r'Significances, $S_{g_\leq}$')
    ax.set_ylabel('Probability Density')

    return h, bins, patches, line, lineCollection

def logRhoOnX(clusterer, X, ax = None, colorbar = True, scatterKwargs = {}, colorbarKwargs = {}):
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
    colorbar : bool, optional
        Whether to add a colour bar to the plot.
    scatterKwargs : dict, optional
        Keyword arguments to be passed to the scatter function.
    colorbarKwargs : dict, optional
        Keyword arguments to be passed to the colorbar function.
    """

    # Check if the axes have been provided
    if ax is None: ax = plt.gca()

    # Set default scatter and colorbar kwargs
    if ('c' not in scatterKwargs) and ('color' not in scatterKwargs): scatterKwargs['c'] = clusterer.logRho
    if 'cmap' not in scatterKwargs: scatterKwargs['cmap'] = mcolors.LinearSegmentedColormap.from_list('density', [(0, 'royalblue'), (1, 'red')])
    if 'label' in colorbarKwargs: del colorbarKwargs['label']
    if 'ax' in colorbarKwargs: del colorbarKwargs['ax']

    # Plot the data
    densityField = ax.scatter(*X.T, **scatterKwargs)

    # Add colour bar
    if colorbar: plt.colorbar(densityField, label = r'$\log\hat\rho$', ax = ax, **colorbarKwargs)

def labelsOnX(clusterer, X, skipZeroth = True, ax = None, scatterKwargs = {}):
    """
    Make a plot of the data X coloured by the AstroLink cluster labels.

    Parameters
    ----------
    clusterer : AstroLink
        An instance of the AstroLink class.
    X : array
        The data to be plotted.
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the data. If not provided, the current axes 
        will be used.
    scatterKwargs : dict, optional
        Keyword arguments to be passed to the scatter function.
    """

    # Check if the axes have been provided
    if ax is None: ax = plt.gca()

    # Set default scatter kwargs
    if 'c' in scatterKwargs: del scatterKwargs['c']
    if 'color' in scatterKwargs: del scatterKwargs['color']
    if 'fc' in scatterKwargs: del scatterKwargs['fc']
    if 'facecolor' in scatterKwargs: del scatterKwargs['facecolor']
    if 'label' in scatterKwargs: del scatterKwargs['label']

    # Set the starting index for cluster labels to plot
    if skipZeroth: startIndex = 1
    else: startIndex = 0

    # Plot the cluster labels onto X
    for i, (clst, clstID) in enumerate(zip(clusterer.clusters[startIndex:], clusterer.ids[startIndex:])):
        clusterMembers = clusterer.ordering[clst[0]:clst[1]]
        ax.scatter(*X[clusterMembers].T, facecolor = f"C{i}", label = clstID, **scatterKwargs)