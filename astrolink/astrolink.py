"""
AstroLink: A general-purpose algorithm for finding astrophysically-relevant
clusters from point-cloud data.

Author: William H. Oliver <william.hardie.oliver@gmail.com>
License: MIT
"""

# Standard libraries
import os
import time

# Third-party libraries
from numba import njit, prange
import numpy as np
from pykdtree.kdtree import KDTree
from scipy.optimize import minimize
from scipy.special import beta as beta_fun
from scipy.special import betainc as betainc_fun
from scipy.stats import norm, beta
from sklearn.utils import gen_batches, get_chunk_n_rows


class AstroLink:
    """A class to represent the AstroLink clustering algorithm.

    AstroLink, the algorithmic successor of CluSTAR-ND and Halo-OPTICS, finds
    clusters from point-cloud data that are statistical outliers from the
    stochastic clumps within the input data. In addition to this, clusters are
    defined as contiguous groupings of points with local density greater than
    some bounding iso-density contour.

    Given a point-cloud data set, AstroLink;
    * (optionally) transforms the data so that each feature has unit variance,
    * estimates the local-density of each point,
    * aggregates points into an ordered list while keeping track of groups,
    * calculates the prominence (a measure of clusteredness) for each group,
    * fits a descriptive model to the prominence distribution, and
    * labels outliers (from this distribution) as clusters.

    Parameters
    ----------
    P : `numpy.ndarray` of shape (n_samples, d_features)
        A feature array of the point-cloud data set from which AstroLink finds
        clusters.
    k_den : `int`, default = 20
        The number of nearest neighbours used to estimate the local density of
        each point.
    adaptive : `int`, default = 1
        A flag that determines the behaviour of the input data transformation.
        If `adaptive` is set to 0, then AstroLink does not transform the
        data. If `adaptive` is set to 1, then AstroLink rescales each feature so
        that is has unit variance.
    S : `float` or `str`, default = 'auto'
        The lower significance threshold of any clusters found, i.e clusters are
        `S`-sigma outliers from the stochastic clumps within the input data.
        If `S` is set to a float, then this value is used directly. If `S`
        is set to 'auto', then a data-driven value of `S` is used.
    k_link : `int` or `str`, default = 'auto'
        The number of nearest neighbours used to aggregate points. If `k_link`
        is set to an int, this value is used directly. If `k_link` is set to
        'auto', then a data-driven value of `k_link` is used.
    h_style : `int`, default = 1
        A flag that determines the behaviour of the cluster hierarchy. If
        `h_style` is set to 0, then the resultant hierarchy is styled
        similarly to SubFind and EnLink. If `h_style` is set to 1, then
        additional clusters are incorporated into the cluster hierarchy.
    workers : `int`, default = -1
        The number of processors used in parallelised computations. If workers`
        is set to -1, then AstroLink will use all processors available.
        Otherwise, `workers` must be a value between 1 and N_cpu.
    verbose : `int`, default = 1
        The verbosity of the AstroLink class. If `verbose` is set to 0, then
        AstroLink will not report any of its activity. Increasing `verbose` will
        make AstroLink report more of its activity.

    Attributes
    ----------
    logRho : `numpy.ndarray` of shape (n_samples,)
        The normalised log-scaled local densities of each point, such that
        `logRho[i]` corresponds to the data point `P[i]`.
    ordering : `numpy.ndarray` of shape (n_samples,)
        The ordered list of points, such that `P[ordering[i]]` is the point that
        belongs in the `i`-th position in the ordered list. The ordered-density
        plot, which can used to visualise clustering structure, can be created
        by plotting `y = logRho[ordering]` vs `x = range(n_samples)`.
    clusters : `numpy.ndarray` of shape (n_clusters, 2)
        The start and end positions of each cluster as it appears in the ordered
        list, such that `ordering[clusters[i, 0]:clusters[i, 1]]` gives an array
        of the indices of the points within cluster `i`.
    ids : `numpy.ndarray` of shape (n_clusters,)
        The id strings of the clusters, such that `ids[i]` corresponds to
        cluster `i`. These id strings have a hierarchical structure to their
        notation; the cluster with id '1', is the parent of the cluster with id
        '1-1', which is the sibling of the cluster with id '1-2', which is the
        parent of the cluster with id '1-2-1' (and so on...).
    significances : `numpy.ndarray` of shape (n_clusters,)
        The statistical significance of each cluster, such that
        `significance[i]` corresponds to cluster `i`. These values equivalent to
        'how many sigma' outliers the clusters are from the noise within the
        input data.
    groups_leq : `numpy.ndarray` of shape (n_groups_leq, 2)
        Similar to structure and meaning as `clusters`, however `groups_leq`
        includes all smaller groups merged within the aggregation tree
        (including noise).
    prominences_leq : `numpy.ndarray` of shape (n_groups_leq,)
        The prominence values of each group in `groups_leq`, such that
        `prominences_leq[i]` corresponds to group `i` in `groups_leq`.
    groups_leq_sigs : `numpy.ndarray` of shape (n_groups_leq,)
        The statistical significance of each group, such that
        `groups_leq_sigs[i]` corresponds to group `i` in `groups_leq`.
    groups_geq : `numpy.ndarray` of shape (n_groups_geq, 2)
        Similar to structure and meaning as `clusters`, however `groups_geq`
        includes all larger groups merged within the aggregation tree
        (including noise).
    prominences_geq : `numpy.ndarray` of shape (n_groups_geq,)
        The prominence values of each group in `groups_geq`, such that
        `prominences_geq[i]` corresponds to group `i` in `groups_geq`.
    groups_geq_sigs : `numpy.ndarray` of shape (n_groups_geq,)
        The statistical significance of each group in `groups_geq`, such that
        `groups_geq_sigs[i]` corresponds to group `i` in `groups_geq`.
    pFit : `numpy.ndarray` of shape (3,)
        The model parameters `[c, a, b]` for the model that fits the
        distribution of `prominences_leq`. `c` is the cutoff between the Beta
        and Uniform distributions. `a` and `b` are the shape parameters for the
        Beta distribution.

    Methods
    -------
    run():
        Runs the AstroLink algorithm and produces a hierarchy of clusters from
        the input data.
    transform_data():
        Transforms the input data according to the parameter `adaptive`.
    estimate_density_and_kNN():
        Estimates the log-scaled local density of each point from its `k_den`
        nearest neighbours and then keeps only the `k_link` nearest neighbours
        of each point.
    aggregate():
        Aggregates points together to form the ordered list whilst keeping track
        of groups.
    compute_significances():
        Fits a model to the distribution of subgroup prominences and finds each
        group's significance value.
    extract_clusters(rootID = '1'):
        Classifies groups that have significance of at least `S` as clusters and
        forms the hierarchy according to the parameter `h_style`.

    Examples
    --------
    When applying AstroLink to uniform noise, one 'cluster' is found that
    represents the entire data set.

    >>> from astrolink import * # Numpy is imported (as np) from astrolink
    >>> P = np.random.uniform(0, 1 (10**5, 3)) # Random input data (no clusters)
    >>> astrolink = AstroLink(P)
    >>> astrolink.run()
    >>> astrolink.clusters
    array([[0, 100000]])
    >>> astrolink.ids
    array(['1'], dtype='<U1')

    When applying AstroLink to structured data, two Gaussians blobs for example,
    then an estimate of the hierarchy of clusters is found.

    >>> from astrolink import *
    >>> gauss2D_1 = np.random.normal(0, 1, (10**4, 2)) # Two gaussian blobs
    >>> gauss2D_2 = np.random.normal([10, 0], 1, (10**4, 2))
    >>> P = np.concatenate((gauss2D_1, gauss2D_2), axis = 0)
    >>> astrolink = AstroLink(P)
    >>> astrolink.run()
    >>> astrolink.clusters
    array([[0, 20000], [0, 10000], [10000, 20000]])
    >>> astrolink.ids
    array(['1', '1-1', '1-2'], dtype='<U3')
    """

    def __init__(self, P, k_den = 20, adaptive = 1, S = 'auto', k_link = 'auto', h_style = 1, workers = -1, verbose = 1):
        # Input Data
        check_P = isinstance(P, np.ndarray) and len(P.shape) == 2
        assert check_P, "Input data 'P' needs to be a 2D numpy array!"
        self.P = P
        self.n_samples, self.d_features = self.P.shape

        # Parameters
        check_k_den = int(k_den) == k_den and k_den <= self.n_samples
        assert check_k_den, "Parameter 'k_den' must be an integer less than P.shape[0]!"
        self.k_den = int(k_den)

        check_adaptive = adaptive in [0, 1]
        assert check_adaptive, "Parameter 'adaptive' must be set as either '0' or '1'!"
        self.adaptive = adaptive

        check_S = S == 'auto' or S <= 38.44939448087599
        assert check_S, "Parameter 'S' must be set to 'auto' or be less than or equal to 38.44939448087599 due to the limiting precision of the (inverse) survival functions from scipy.stats!"
        self.S = S

        check_k_link = k_link == 'auto' or 1 < k_link <= self.k_den
        assert check_k_link, "Parameter 'k_link' needs to be an integer satisfying 1 < k_link <= k_den or 'auto'!"
        if k_link == 'auto': self.k_link = max(int(np.ceil(11.97*self.d_features**(-2.23) - 22.97*self.k_den**(-0.57) + 10.03)), 7) # Fitted params [ 11.97420072  -2.22754059 -22.96660813  -0.56948651   9.03446908]
        else: self.k_link = k_link

        check_h_style = h_style in [0, 1]
        assert check_h_style, "Parameter 'h_style' must be set as either '0' or '1'!"
        self.h_style = h_style

        check_workers = 1 <= workers <= os.cpu_count() or workers == -1
        assert check_workers, f"Parameter 'workers' must be set as either '-1' or needs to be an integer that is >= 1 and <= N_cpu (= {os.cpu_count()})"
        os.environ["OMP_NUM_THREADS"] = f"{workers}" if workers != -1 else f"{os.cpu_count()}"
        self.workers = workers
        self.verbose = verbose

    def _printFunction(self, message, returnLine = True):
        if self.verbose:
            if returnLine: print(f"AstroLink: {message}\r", end = '')
            else: print(f"AstroLink: {message}")

    def run(self):
        """Runs the AstroLink algorithm.

        This method runs `transform_data()`, `estimate_density_and_kNN()`,
        `aggregate()`, `compute_significances()`, `extract_clusters()`.
        """

        self._printFunction(f"Started             | {time.strftime('%Y-%m-%d %H:%M:%S')}", returnLine = False)
        begin = time.perf_counter()

        # Transform the data
        self.transform_data()

        # Compute densities and nearest neighbours
        self.estimate_density_and_kNN()

        # Order points, find groups, compute prominences
        self.aggregate()

        # Fit model to subgroup prominences and find group significance
        self.compute_significances()

        # Find clusters and hierarchy
        self.extract_clusters()

        self._totalTime = time.perf_counter() - begin
        if self.verbose > 1:
            self._printFunction(f"Transformation time | {100*self._transformTime/self._totalTime:.2f}%    ", returnLine = False)
            self._printFunction(f"kNN query time      | {100*self._logRhoTime/self._totalTime:.2f}%       ", returnLine = False)
            self._printFunction(f"Aggregation time    | {100*self._aggregateTime/self._totalTime:.2f}%    ", returnLine = False)
            self._printFunction(f"Regression time     | {100*self._regrTime/self._totalTime:.2f}%         ", returnLine = False)
            self._printFunction(f"Rejection time      | {100*self._rejTime/self._totalTime:.2f}%          ", returnLine = False)
        self._printFunction(f"Completed           | {time.strftime('%Y-%m-%d %H:%M:%S')}       ", returnLine = False)

    def transform_data(self):
        """Transforms the input data.

        If `adaptive` is set to 0, then the data is not transformed. However, if
        `adaptive` is set to 1 (default), then each feature of the data is
        rescaled so that it has unit variance.

        This method generates the `P_transform` attribute.
        """

        start = time.perf_counter()
        if self.adaptive:
            if self.verbose > 1: self._printFunction('Transforming data...      ')
            if self.d_features > 1: self.P_transform = self._rescale(self.P)
            else: self.P_transform = self.P/self.P.std()
        else: self.P_transform = self.P
        self._transformTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _rescale(P):
        P_transform = np.empty_like(P)
        for i in prange(P.shape[-1]):
            P_transform[:, i] = P[:, i]/P[:, i].std()
        return P_transform

    def estimate_density_and_kNN(self):
        """Estimates the normalised log-scaled local density of each point.

        Uses a KD-Tree to find the `k_den` nearest neighbours of every point in
        the input data and then uses these neighbours to estimate the local
        density of each point using an Epanechnikov kernel + Balloon estimator.
        The local densities are then log-scaled and normalised between 0 and 1.

        This method requires the `P_transform` attribute to have already been
        created, via the `transform_data()` method or otherwise.

        This method generates the `logRho` and `kNN` attributes.
        """

        if self.verbose > 1: self._printFunction('Computing densities...    ')
        start = time.perf_counter()
        self.logRho = np.empty_like(self.P_transform, shape = (self.n_samples,))
        self.kNN = np.empty((self.n_samples, self.k_link), dtype = np.uint32)
        nbrs = KDTree(self.P_transform)
        for sl in gen_batches(self.n_samples, get_chunk_n_rows(row_bytes = 16*self.k_den, max_n_rows = self.n_samples)):
            sqr_distances, indices = nbrs.query(self.P_transform[sl], k = self.k_den, sqr_dists = True)
            self.logRho[sl] = self._compute_logRho(sqr_distances, self.k_den, self.d_features)
            self.kNN[sl] = indices[:, :self.k_link]
        del self.P_transform
        self.logRho = self._normalise(self.logRho)
        self._logRhoTime = time.perf_counter() - start

    @staticmethod
    @njit()
    def _compute_logRho(sqr_distances, k_den, d_features):
        coreSqrDist = sqr_distances[:, -1]
        return np.log((k_den - sqr_distances.sum(axis = 1)/coreSqrDist)/coreSqrDist**(d_features/2))

    @staticmethod
    @njit(fastmath = True)
    def _normalise(x):
        minVal = x[0]
        maxVal = x[0]
        for xi in x[1:]:
            if xi < minVal: minVal = xi
            elif xi > maxVal: maxVal = xi
        return (x - minVal)/(maxVal - minVal)

    def aggregate(self):
        """Aggregates the data points.

        Computes edge weights between each point and each of its `k_link`
        nearest neighbours, sorts these into descending order, aggregates points
        in this order while keeping track of structural information about the
        data.

        This method requires the `logRho` and 'kNN' attributes to have already
        been created, via the `estimate_density_and_kNN` method or otherwise.

        This method generates the `ordering`, `groups`, `prominences`,
        `groups_comp`, and `prominences_comp` attributes.

        This method deletes the `kNN` attribute.
        """

        if self.verbose > 1: self._printFunction('Aggregating points...        ')
        start = time.perf_counter()
        if self.logRho.dtype == np.float64: self.ordering, self.groups_leq, self.prominences_leq, self.groups_geq, self.prominences_geq = self._aggregate_njit_float64(self.logRho, self.kNN)
        else: self.ordering, self.groups_leq, self.prominences_leq, self.groups_geq, self.prominences_geq = self._aggregate_njit_float32(self.logRho, self.kNN)
        del self.kNN
        self._aggregateTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _aggregate_njit_float64(logRho, kNN):
        # Order points
        n_samples, k_link = kNN.shape
        shape_0 = n_samples*(k_link - 1)
        edges = np.empty(shape_0, dtype = np.float64)
        ordered_pairs = np.empty((shape_0, 2), dtype = np.uint32)
        for id_i in prange(n_samples):
            lr_i = logRho[id_i]
            pos = id_i*(k_link - 1)
            for id_j in kNN[id_i]:
                if id_i != id_j:
                    lr_j = logRho[id_j]
                    pair_pos = ordered_pairs[pos]
                    if lr_i >= lr_j:
                        edges[pos] = lr_j
                        pair_pos[0] = id_i
                        pair_pos[1] = id_j
                    else:
                        edges[pos] = lr_i
                        pair_pos[0] = id_j
                        pair_pos[1] = id_i
                    pos += 1
        ordered_pairs = ordered_pairs[edges.argsort()[::-1]]

        # Kruskal's minimum spanning tree + hierarchy tracking
        ids = np.full((logRho.size,), logRho.size, dtype = np.uint32)
        count = 0
        aggregations = [[np.uint32(0) for i in range(0)] for i in range(0)]
        emptyIntList = [np.uint32(0) for i in range(0)]
        # For smaller groups
        starts_leq = [np.uint32(0) for i in range(0)]
        sizes_leq = [np.uint32(0) for i in range(0)]
        prominences_leq = [np.float64(0.0) for i in range(0)]
        children = [[np.uint32(0) for i in range(0)] for i in range(0)]
        # For larger groups
        starts_geq = [np.uint32(0) for i in range(0)]
        sizes_geq = [np.uint32(0) for i in range(0)]
        prominences_geq = [np.float64(0.0) for i in range(0)]

        for pair in ordered_pairs:
            id_0, id_1 = ids[pair]
            if id_0 != logRho.size: # pair[0] is already aggregated
                if id_0 == id_1: pass # Same group
                elif id_1 == logRho.size: # pair[1] is not yet aggregated
                    p_1 = pair[1]
                    ids[p_1] = id_0
                    aggregations[id_0].append(p_1)
                    sizes_leq[id_0] += 1
                else: # Different groups -> merge groups
                    if sizes_leq[id_0] < sizes_leq[id_1]: id_0, id_1 = id_1, id_0
                    for id_i in aggregations[id_1]: ids[id_i] = id_0
                    aggregations[id_0].extend(aggregations[id_1])
                    aggregations[id_1] = emptyIntList
                    currLogRho = logRho[pair[1]]
                    # Track complementary group
                    starts_geq[id_1] = starts_leq[id_0]
                    sizes_geq[id_1] = sizes_leq[id_0]
                    prominences_geq[id_1] = prominences_leq[id_0] - currLogRho
                    # Merge
                    starts_leq[id_1] += sizes_leq[id_0]
                    sizes_leq[id_0] += sizes_leq[id_1]
                    prominences_leq[id_0] = max(prominences_leq[id_0], prominences_leq[id_1])
                    prominences_leq[id_1] -= currLogRho
                    children[id_0].append(id_1)
            elif id_1 == logRho.size: # Neither are aggregated
                ids[pair] = count
                count += 1
                aggregations.append([pair[0], pair[1]])
                # Create group
                starts_leq.append(0)
                sizes_leq.append(2)
                prominences_leq.append(logRho[pair[0]])
                children.append([np.uint32(0) for i in range(0)])
                # Track complementary group
                starts_geq.append(0)
                sizes_geq.append(0)
                prominences_geq.append(0.0)
            else: # pair[1] is already aggregated (but not pair[0])
                p_0 = pair[0]
                ids[p_0] = id_1
                aggregations[id_1].append(p_0)
                sizes_leq[id_1] += 1
                prominences_leq[id_1] = max(prominences_leq[id_1], logRho[pair[0]])

        # Merge separate aggregations in order of decreasing size
        aggArr = np.unique(ids)
        if aggArr.size == 1: id_final = aggArr[0]
        else: # If points were not all aggregated together, make it so.
            sortedAggregations = sorted(zip([sizes_leq[id_i] for id_i in aggArr], aggArr))
            _, id_final = sortedAggregations[-1]
            for size_leq, id_leq in sortedAggregations[-2::-1]:
                aggregations[id_final].extend(aggregations[id_leq])
                aggregations[id_leq] = emptyIntList
                # Track complementary group
                starts_geq[id_leq] = starts_leq[id_final]
                sizes_geq[id_leq] = sizes_leq[id_final]
                prominences_geq[id_leq] = prominences_leq[id_final]
                # Merge
                starts_leq[id_leq] += sizes_leq[id_final]
                sizes_leq[id_final] += size_leq
                children[id_final].append(id_leq)
        emptyIntArr = np.empty(0, dtype = np.uint32)
        ids = emptyIntArr
        aggArr = emptyIntArr

        # Ordered list
        ordering = np.array(aggregations[id_final], dtype = np.uint32)
        aggregations[id_final] = emptyIntList

        # Finalise groups and correct for noise
        activeGroups = [id_i for id_i in children[id_final]]
        while activeGroups:
            id_leq = activeGroups.pop()
            childIDs = children[id_leq]
            if childIDs:
                startAdjust = starts_leq[id_leq]
                activeGroups.extend(childIDs)
                noise = 0.0
                for id_geq, childID in enumerate(childIDs):
                    starts_leq[childID] += startAdjust
                    starts_geq[childID] += startAdjust
                    if id_geq > 0: prominences_geq[childID] -= np.sqrt(noise/id_geq)
                    noise += prominences_leq[childID]**2
                prominences_leq[id_leq] -= np.sqrt(noise/(id_geq + 1))
                children[id_leq] = emptyIntList

        # Lists to Arrays
        starts_leq = np.array(starts_leq, dtype = np.uint32)
        sizes_leq = np.array(sizes_leq, dtype = np.uint32)
        prominences_leq = np.array(prominences_leq, dtype = np.float64)
        starts_geq = np.array(starts_geq, dtype = np.uint32)
        sizes_geq = np.array(sizes_geq, dtype = np.uint32)
        prominences_geq = np.array(prominences_geq, dtype = np.float64)

        # Clean arrays
        starts_leq = np.delete(starts_leq, id_final)
        groups_leq = np.column_stack((starts_leq, starts_leq + np.delete(sizes_leq, id_final)))
        starts_leq, sizes_leq = emptyIntArr, emptyIntArr
        prominences_leq = np.delete(prominences_leq, id_final)
        starts_geq = np.delete(starts_geq, id_final)
        groups_geq = np.column_stack((starts_geq, starts_geq + np.delete(sizes_geq, id_final)))
        starts_geq, sizes_geq = emptyIntArr, emptyIntArr
        prominences_geq = np.delete(prominences_geq, id_final)

        # Reorder arrays
        reorder = groups_leq[:, 0].argsort()
        return ordering, groups_leq[reorder], prominences_leq[reorder], groups_geq[reorder], prominences_geq[reorder]

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _aggregate_njit_float32(logRho, kNN):
        # Order points
        n_samples, k_link = kNN.shape
        shape_0 = n_samples*(k_link - 1)
        edges = np.empty(shape_0, dtype = np.float32)
        ordered_pairs = np.empty((shape_0, 2), dtype = np.uint32)
        for id_i in prange(n_samples):
            lr_i = logRho[id_i]
            pos = id_i*(k_link - 1)
            for id_j in kNN[id_i]:
                if id_i != id_j:
                    lr_j = logRho[id_j]
                    pair_pos = ordered_pairs[pos]
                    if lr_i >= lr_j:
                        edges[pos] = lr_j
                        pair_pos[0] = id_i
                        pair_pos[1] = id_j
                    else:
                        edges[pos] = lr_i
                        pair_pos[0] = id_j
                        pair_pos[1] = id_i
                    pos += 1
        ordered_pairs = ordered_pairs[edges.argsort()[::-1]]

        # Kruskal's minimum spanning tree + hierarchy tracking
        ids = np.full((logRho.size,), logRho.size, dtype = np.uint32)
        count = 0
        aggregations = [[np.uint32(0) for i in range(0)] for i in range(0)]
        emptyIntList = [np.uint32(0) for i in range(0)]
        # For smaller groups
        starts_leq = [np.uint32(0) for i in range(0)]
        sizes_leq = [np.uint32(0) for i in range(0)]
        prominences_leq = [np.float32(0.0) for i in range(0)]
        children = [[np.uint32(0) for i in range(0)] for i in range(0)]
        # For larger groups
        starts_geq = [np.uint32(0) for i in range(0)]
        sizes_geq = [np.uint32(0) for i in range(0)]
        prominences_geq = [np.float32(0.0) for i in range(0)]

        for pair in ordered_pairs:
            id_0, id_1 = ids[pair]
            if id_0 != logRho.size: # pair[0] is already aggregated
                if id_0 == id_1: pass # Same group
                elif id_1 == logRho.size: # pair[1] is not yet aggregated
                    p_1 = pair[1]
                    ids[p_1] = id_0
                    aggregations[id_0].append(p_1)
                    sizes_leq[id_0] += 1
                else: # Different groups -> merge groups
                    if sizes_leq[id_0] < sizes_leq[id_1]: id_0, id_1 = id_1, id_0
                    for id_i in aggregations[id_1]: ids[id_i] = id_0
                    aggregations[id_0].extend(aggregations[id_1])
                    aggregations[id_1] = emptyIntList
                    currLogRho = logRho[pair[1]]
                    # Track complementary group
                    starts_geq[id_1] = starts_leq[id_0]
                    sizes_geq[id_1] = sizes_leq[id_0]
                    prominences_geq[id_1] = prominences_leq[id_0] - currLogRho
                    # Merge
                    starts_leq[id_1] += sizes_leq[id_0]
                    sizes_leq[id_0] += sizes_leq[id_1]
                    prominences_leq[id_0] = max(prominences_leq[id_0], prominences_leq[id_1])
                    prominences_leq[id_1] -= currLogRho
                    children[id_0].append(id_1)
            elif id_1 == logRho.size: # Neither are aggregated
                ids[pair] = count
                count += 1
                aggregations.append([pair[0], pair[1]])
                # Create group
                starts_leq.append(0)
                sizes_leq.append(2)
                prominences_leq.append(logRho[pair[0]])
                children.append([np.uint32(0) for i in range(0)])
                # Track complementary group
                starts_geq.append(0)
                sizes_geq.append(0)
                prominences_geq.append(0.0)
            else: # pair[1] is already aggregated (but not pair[0])
                p_0 = pair[0]
                ids[p_0] = id_1
                aggregations[id_1].append(p_0)
                sizes_leq[id_1] += 1
                prominences_leq[id_1] = max(prominences_leq[id_1], logRho[pair[0]])

        # Merge separate aggregations in order of decreasing size
        aggArr = np.unique(ids)
        if aggArr.size == 1: id_final = aggArr[0]
        else: # If points were not all aggregated together, make it so.
            sortedAggregations = sorted(zip([sizes_leq[id_i] for id_i in aggArr], aggArr))
            _, id_final = sortedAggregations[-1]
            for size_leq, id_leq in sortedAggregations[-2::-1]:
                aggregations[id_final].extend(aggregations[id_leq])
                aggregations[id_leq] = emptyIntList
                # Track complementary group
                starts_geq[id_leq] = starts_leq[id_final]
                sizes_geq[id_leq] = sizes_leq[id_final]
                prominences_geq[id_leq] = prominences_leq[id_final]
                # Merge
                starts_leq[id_leq] += sizes_leq[id_final]
                sizes_leq[id_final] += size_leq
                children[id_final].append(id_leq)
        emptyIntArr = np.empty(0, dtype = np.uint32)
        ids = emptyIntArr
        aggArr = emptyIntArr

        # Ordered list
        ordering = np.array(aggregations[id_final], dtype = np.uint32)
        aggregations[id_final] = emptyIntList

        # Finalise groups and correct for noise
        activeGroups = [id_i for id_i in children[id_final]]
        while activeGroups:
            id_leq = activeGroups.pop()
            childIDs = children[id_leq]
            if childIDs:
                startAdjust = starts_leq[id_leq]
                activeGroups.extend(childIDs)
                noise = 0.0
                for id_geq, childID in enumerate(childIDs):
                    starts_leq[childID] += startAdjust
                    starts_geq[childID] += startAdjust
                    if id_geq > 0: prominences_geq[childID] -= np.sqrt(noise/id_geq)
                    noise += prominences_leq[childID]**2
                prominences_leq[id_leq] -= np.sqrt(noise/(id_geq + 1))
                children[id_leq] = emptyIntList

        # Lists to Arrays
        starts_leq = np.array(starts_leq, dtype = np.uint32)
        sizes_leq = np.array(sizes_leq, dtype = np.uint32)
        prominences_leq = np.array(prominences_leq, dtype = np.float32)
        starts_geq = np.array(starts_geq, dtype = np.uint32)
        sizes_geq = np.array(sizes_geq, dtype = np.uint32)
        prominences_geq = np.array(prominences_geq, dtype = np.float32)

        # Clean arrays
        starts_leq = np.delete(starts_leq, id_final)
        groups_leq = np.column_stack((starts_leq, starts_leq + np.delete(sizes_leq, id_final)))
        starts_leq, sizes_leq = emptyIntArr, emptyIntArr
        prominences_leq = np.delete(prominences_leq, id_final)
        starts_geq = np.delete(starts_geq, id_final)
        groups_geq = np.column_stack((starts_geq, starts_geq + np.delete(sizes_geq, id_final)))
        starts_geq, sizes_geq = emptyIntArr, emptyIntArr
        prominences_geq = np.delete(prominences_geq, id_final)

        # Reorder arrays
        reorder = groups_leq[:, 0].argsort()
        return ordering, groups_leq[reorder], prominences_leq[reorder], groups_geq[reorder], prominences_geq[reorder]

    def compute_significances(self):
        """Computes statistical significances for all groups.

        Constructs the subgroup prominence model (a combination of a Beta
        and a Uniform distribution) and fits it by minimising the negative
        log-likelihood of the resulting probability distribution. The Beta
        distribution (with model-fitted parameters) is then used alongisde the
        standard normal distribution to transform prominence values into
        statistical significance values.

        The method requires the `prominences` and `prominences_comp` attributes
        to have already been created, via the `aggregate` method or otherwise.

        This method generates the `group_sigs`, `group_comp_sigs`, and `pFit`
        attributes.
        """

        if self.verbose > 1: self._printFunction('Finding significances...  ')
        start = time.perf_counter()

        # Fit model
        self._proms_ordered, self._lnx_cumsum, self._ln_1_minus_x_cumsum, self.pFit = self._minimize_init(self.prominences_leq)
        tol = 1e-5
        bnds = ((tol, 1 - tol), (1 + tol, None), (1, None))
        sol = minimize(self._negLL, self.pFit, jac = '3-point', bounds = bnds, tol = tol)
        del self._proms_ordered, self._lnx_cumsum, self._ln_1_minus_x_cumsum
        if sol.success: self.pFit = sol.x
        else: self._printFunction('[Warning] Prominence model may be incorrectly fitted!', returnLine = False)

        # Calculate statistical significance values
        self.groups_leq_sigs = norm.isf(beta.sf(self.prominences_leq, self.pFit[1], self.pFit[2]))
        self.groups_geq_sigs = norm.isf(beta.sf(self.prominences_geq, self.pFit[1], self.pFit[2]))
        self._regrTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True)
    def _minimize_init(prominences):
        proms_ordered = np.sort(prominences[np.logical_and(prominences > 0, prominences < 1)])
        med, var = proms_ordered[proms_ordered.size//2], proms_ordered.var()
        term1 = 1 - med
        term2 = med*term1/var - 1
        return proms_ordered, np.log(proms_ordered).cumsum(), np.log(1 - proms_ordered).cumsum(), np.array([proms_ordered[int(0.99*proms_ordered.size)], term2*med, term2*term1])

    def _negLL(self, p):
        return self._negLL_njit(p, self._proms_ordered, self._lnx_cumsum, self._ln_1_minus_x_cumsum, beta_fun(p[1], p[2])*betainc_fun(p[1], p[2], p[0]))

    @staticmethod
    @njit()
    def _negLL_njit(p, proms_ordered, lnx_cumsum, ln_1_minus_x_cumsum, norm_constant):
        c, a, b = p
        beta_cut, right = 0, lnx_cumsum.size - 1
        while right - beta_cut > 1:
            middle = (right - beta_cut)//2 + beta_cut
            if proms_ordered[middle] < c: beta_cut = middle
            else: right = middle
        uniform_term = (c**(a - 1))*((1 - c)**(b - 1))
        return lnx_cumsum.size*np.log(norm_constant + uniform_term*(1 - c)) - (a - 1)*lnx_cumsum[beta_cut] - (b - 1)*ln_1_minus_x_cumsum[beta_cut] - (lnx_cumsum.size - beta_cut - 1)*np.log(uniform_term)

    def extract_clusters(self, rootID = '1'):
        """Extract clusters from among the structure found.

        First classifies subgroups that are statistical outliers. Then if
        `h_style` is set to 1, finds their corresponding complementary groups
        that are also statistical outliers. For each of these that are the
        smallest out of those that share the same starting position within the
        ordered list, classify them too as clusters. Following this, if rootID
        is not set as `None` then also classify the input data the root cluster.
        Finally, generate the array of id strings for the clusters.

        This method requires the `groups` and `groups_comp` attributes to have
        already been created, via the `aggregate()` method or otherwise. It also
        requires the `group_sigs` and `group_comp_sigs` attributes to have
        already been created, via the `compute_significance` method or
        otherwise.

        This method generates the 'clusters', `ids`, and `significances`
        attributes.

        Parameters
        ----------
        rootID : `str` or `None`, default = '1'
            If `rootID` is set as a `str`, it must be in the format 'x-y-...',
            e.g. '1', '1-2', etc. In this case, a cluster representing the
            entire input data will appear in the `clusters` attribute.
            Contrarily, this will not be the case if `rootID` is set to `None`.
        """

        if self.verbose > 1: self._printFunction('Finding clusters...       ')
        start = time.perf_counter()

        # Classify clusters as groups that are significant outliers
        if self.S == 'auto': self.S = norm.isf(beta.sf(self.pFit[0], self.pFit[1], self.pFit[2]))
        sl = self.groups_leq_sigs >= self.S
        self.clusters = self.groups_leq[sl]
        self.significances = self.groups_leq_sigs[sl]

        # Optional hierarchy correction
        if self.h_style == 1:
            # Retrieve complementary groups whose corresponding subgroup is
            # significantly clustered and who is itself significantly clustered
            sl = np.logical_and(sl, self.groups_geq_sigs >= self.S)
            significances_geq = self.groups_geq_sigs[sl]
            clusters_geq = self.groups_geq[sl]

            # Keep only those complementary groups that are the smallest in their cascade
            sl = np.zeros(clusters_geq.shape[0], dtype = np.bool_)
            cascade_starts_unique = np.unique(clusters_geq[:, 0])
            for cascade_start in cascade_starts_unique:
                sl[np.where(clusters_geq[:, 0] == cascade_start)[0][0]] = 1

            # Merge the hierarchy and clean arrays
            self.significances = np.concatenate((self.significances, significances_geq[sl]))
            self.clusters = np.vstack((self.clusters, clusters_geq[sl]))
            reorder = np.array(sorted(np.arange(self.clusters.shape[0]), key = lambda i: [self.clusters[i, 0], self.n_samples - self.clusters[i, 1]]), dtype = np.uint32)
            self.significances = self.significances[reorder]
            self.clusters = self.clusters[reorder]

        # Optionally add on root-level cluster
        if rootID is not None:
            self.significances = np.concatenate((np.array([np.inf]), self.significances))
            self.clusters = np.vstack((np.array([[0, self.n_samples]], dtype = np.uint32), self.clusters))
        else: rootID = '1'

        # Label clusters according to their hierarchy
        self.ids = []
        children = []
        for i, clst in enumerate(self.clusters):
            children.append(0)
            parent = next((-j - 1 for j, v in enumerate(self.clusters[:i][::-1]) if clst[0] < v[1] and clst[1] <= v[1]), None)
            if parent is not None: # Child cluster of parent
                children[parent] += 1
                self.ids.append(self.ids[parent] + '-' + str(children[parent]))
            elif len(self.ids) > 0: # Multiple root clusters (not applicable to AstroLink)
                self.ids.append(str(max([int(id.split('-')[0]) for id in self.ids]) + 1))
            else: # First root cluster
                self.ids.append(rootID)
        self.ids = np.array(self.ids)
        self._rejTime = time.perf_counter() - start
