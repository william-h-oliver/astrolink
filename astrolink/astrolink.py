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
from numba import njit, prange, set_num_threads
import numpy as np
from pykdtree.kdtree import KDTree
from scipy.optimize import minimize
from scipy.special import beta as beta_fun
from scipy.stats import norm, beta
from sklearn import get_config
from sklearn.utils import gen_batches


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
    d_intrinsic : `int`, default = None
        The intrinsic dimensionality of the input data. If `d_intrinsic` is set
        to None, then AstroLink assumes that the data has the same 
        dimensionality as the number of features in `P`.
    weights : `numpy.ndarray` of shape (n_samples,), default = None
        The weights of each point in the input data. If `weights` is set to
        None, then all points are given equal weight. Otherwise, the weights
        must be a 1D numpy array with the same length as the number of samples
        in `P`.
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
    workers : `int`, default = 8
        The number of processors used in parallelised computations. If workers`
        is set to -1, then AstroLink will use all processors available.
        Otherwise, `workers` must be a value between 1 and N_cpu.
    verbose : `int`, default = 0
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
    groups : `numpy.ndarray` of shape (n_groups, 3)
        Similar to `clusters`, however `groups` contains the start and end 
        positions of all pairs of larger and smaller groups that were merged 
        together during the aggregation process. Here `groups[:, 0]` and 
        `groups[:, 1]` correspond to the start and end positions of the larger 
        groups, while `groups[:, 1]` and `groups[:, 2]` correspond to the start 
        and end positions of the smaller groups. `groups` can be used to 
        visualise the aggregation merge tree.
    prominences : `numpy.ndarray` of shape (n_groups, 2)
        The prominence values of each group in `groups`, such that
        `prominences[i]` contains the prominence values of both groups stored in 
        `groups[i]`.
    groups_sigs : `numpy.ndarray` of shape (n_groups, 2)
        The statistical significance of each group, such that `groups_sigs[i]` 
        contains the statistical significance values of both groups stored in 
        `groups[i]`.
    pFit : `numpy.ndarray` of shape (2,)
        The shape parameters, `[a, b]`, of the Beta distribution model that fits 
        the `prominences`.
    """

    def __init__(self, P, d_intrinsic = None, weights = None, k_den = 20, adaptive = 1, S = 'auto', k_link = 'auto', h_style = 1, workers = 8, verbose = 0):
        # Input Data
        check_P = isinstance(P, np.ndarray) and len(P.shape) == 2
        assert check_P, "Input data 'P' needs to be a 2D numpy array!"
        self.P = P
        self.n_samples, self.d_features = self.P.shape

        # Intrinsic Dimensionality
        check_d_intrinsic = d_intrinsic is None or (isinstance(d_intrinsic, int) and 1 <= d_intrinsic <= self.d_features)
        assert check_d_intrinsic, "Parameter 'd_intrinsic' must be None or an integer between 1 and the number of features in 'P'!"
        self.d_intrinsic = d_intrinsic if d_intrinsic is not None else self.d_features

        # Weights
        check_weights = weights is None or (isinstance(weights, np.ndarray) and len(weights.shape) == 1 and weights.size == self.n_samples)
        assert check_weights, "Parameter 'weights' must be None or a 1D numpy array with the same length as the number of samples in 'P'!"
        self.weights = weights

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
        if k_link == 'auto': self.k_link = max(int(np.ceil(11.97*self.d_intrinsic**(-2.23) - 22.97*self.k_den**(-0.57) + 10.03)), 7) # Fitted params [ 11.97420072  -2.22754059 -22.96660813  -0.56948651   9.03446908]
        else: self.k_link = k_link

        check_h_style = h_style in [0, 1]
        assert check_h_style, "Parameter 'h_style' must be set as either '0' or '1'!"
        self.h_style = h_style

        check_workers = 1 <= workers or workers == -1
        assert check_workers, f"Parameter 'workers' must be set as either '-1' or needs to be an integer that is >= 1 (values > N_cpu will be set to N_cpu)"
        os.environ["OMP_NUM_THREADS"] = f"{min(workers, os.cpu_count())}" if workers != -1 else f"{os.cpu_count()}"
        set_num_threads(min(workers, os.cpu_count()))
        self.workers = workers
        self.verbose = verbose

    def _printFunction(self, message, returnLine = True, urgent = False):
        if self.verbose or urgent:
            if returnLine: print(f"AstroLink: {message}\r", end = '')
            else: print(f"AstroLink: {message}")

    def run(self):
        """Runs the AstroLink algorithm and produces a hierarchy of clusters
        from the input data.

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
        """Transforms the input data according to the parameter `adaptive`.

        If `adaptive` is set to 0, then the data is not transformed. However, if
        `adaptive` is set to 1 (default), then each feature of the data is
        rescaled so that it has unit variance.

        This method generates the `P_transform` attribute.
        """

        start = time.perf_counter()
        if self.adaptive:
            if self.verbose > 1: self._printFunction('Transforming data...      ')
            if self.d_features > 1: self.P_transform = self._rescale_njit(self.P)
            else: self.P_transform = self.P/self.P.std()
        else: self.P_transform = self.P
        self._transformTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _rescale_njit(P):
        P_transform = np.empty_like(P)
        for i in prange(P.shape[-1]):
            std = P[:, i].std()
            if std > 0: P_transform[:, i] = P[:, i]/std
        return P_transform

    def estimate_density_and_kNN(self):
        """Estimates the normalised log-scaled local density of each point from
        its `k_den` nearest neighbours and then keeps only the `k_link` nearest
        neighbours of each point.

        Uses a KD-Tree to find the `k_den` nearest neighbours of every point in
        the input data and then uses these neighbours to estimate the local
        density of each point using an Epanechnikov kernel + Balloon estimator.
        The local densities are then log-scaled and normalised between 0 and 1.

        This method requires the `P_transform` attribute to have already been
        created, via the `transform_data()` method or otherwise.

        This method generates the `logRho` and `kNN` attributes.

        This method deletes the `P_transform` attribute.
        """

        if self.verbose > 1: self._printFunction('Computing densities...    ')
        start = time.perf_counter()

        # Empty arrays for logRho and kNN and build kd-tree
        self.logRho = np.empty_like(self.P_transform, shape = (self.n_samples,))
        intType = np.uint32 if self.n_samples < 2**32 - 1 else np.uint64
        self.kNN = np.empty((self.n_samples, self.k_link), dtype = intType)
        nbrs = KDTree(self.P_transform)

        # Chunking for memory efficiency
        working_memory = get_config()["working_memory"]
        chunk_n_rows = max(min(int(working_memory * (2**20) // 16*self.k_den), self.n_samples), 1)

        # Estimate densities and find kNN in a memory efficient way
        for sl in gen_batches(self.n_samples, chunk_n_rows):
            # k-nearest neighbours query
            sqr_distances, indices = nbrs.query(self.P_transform[sl], k = self.k_den, sqr_dists = True)

            # Compute logRho for this slice
            if self.weights is None: self.logRho[sl] = self._compute_logRho_njit(sqr_distances, self.k_den, self.d_intrinsic)
            else: self.logRho[sl] = self._compute_weighted_logRho_njit(sqr_distances, self.weights[indices], self.d_intrinsic)

            # Keep only the k_link nearest neighbours
            self.kNN[sl] = indices[:, :self.k_link]
        del self.P_transform

        # Normalise logRho
        self.logRho = self._normalise_njit(self.logRho)
        self._logRhoTime = time.perf_counter() - start

    @staticmethod
    @njit()
    def _compute_logRho_njit(sqr_distances, k_den, d_intrinsic):
        coreSqrDist = sqr_distances[:, -1]
        return np.log((k_den - sqr_distances.sum(axis = 1)/coreSqrDist)/coreSqrDist**(d_intrinsic/2))

    @staticmethod
    @njit()
    def _compute_weighted_logRho_njit(sqr_distances, weights, d_intrinsic):
        coreSqrDist = sqr_distances[:, -1]
        return np.log((weights.sum(axis = 1) - (weights*sqr_distances).sum(axis = 1)/coreSqrDist)/coreSqrDist**(d_intrinsic/2))

    @staticmethod
    @njit(fastmath = True)
    def _normalise_njit(x):
        minVal = x[0]
        maxVal = x[0]
        for xi in x[1:]:
            if xi < minVal: minVal = xi
            elif xi > maxVal: maxVal = xi
        return (x - minVal)/(maxVal - minVal)

    def aggregate(self):
        """Aggregates the data points together to form the ordered list whilst
        keeping track of groups.

        Computes edge weights between each point and each of its `k_link`
        nearest neighbours, sorts these into descending order, aggregates points
        in this order while keeping track of structural information about the
        data.

        This method requires the `logRho` and 'kNN' attributes to have already
        been created, via the `estimate_density_and_kNN` method or otherwise.

        This method generates the `ordering`, `groups`, `prominences` 
        attributes.

        This method deletes the `kNN` attribute.
        """

        if self.verbose > 1: self._printFunction('Aggregating points...        ')
        start = time.perf_counter()

        # Construct a graph from the edges between each point and its k_link nearest neighbours
        pairs, edges = self._make_graph_njit(self.logRho, self.kNN)
        del self.kNN

        # Order the pairs by descending edge weight (faster outside of njit)
        ordered_pairs = pairs[edges.argsort()[::-1]]
        del pairs, edges

        # Choose method based on typing and aggregate points into groups
        typeCase = (self.logRho.dtype == np.float64) + 2*(ordered_pairs.dtype == np.float64)
        if typeCase == 0: aggregate_func = self._aggregate_njit_float32_uint32
        elif typeCase == 1: aggregate_func = self._aggregate_njit_float64_uint32
        elif typeCase == 2: aggregate_func = self._aggregate_njit_float32_uint64
        elif typeCase == 3: aggregate_func = self._aggregate_njit_float64_uint64
        else: raise ValueError("Unexpected data type in AstroLink.aggregate()!")
        self.ordering, self.groups, self.prominences = aggregate_func(self.logRho, ordered_pairs)
        del ordered_pairs
        self._aggregateTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _make_graph_njit(logRho, kNN):
        # Create empty arrays
        n_samples, k_link = kNN.shape
        graphShape = n_samples*(k_link - 1)
        edges = np.empty(graphShape, dtype = logRho.dtype)
        pairs = np.empty((graphShape, 2), dtype = kNN.dtype)

        # For each pair of vertices find the adjoining edge weight
        for id_i in prange(n_samples):
            lr_i = logRho[id_i]
            pos = id_i*(k_link - 1)
            for id_j in kNN[id_i]:
                if id_i != id_j:
                    lr_j = logRho[id_j]
                    pair_pos = pairs[pos]
                    if lr_i >= lr_j:
                        edges[pos] = lr_j
                        pair_pos[0] = id_i
                        pair_pos[1] = id_j
                    else:
                        edges[pos] = lr_i
                        pair_pos[0] = id_j
                        pair_pos[1] = id_i
                    pos += 1

        return pairs, edges

    @staticmethod
    @njit(fastmath = True)
    def _aggregate_njit_float32_uint32(logRho, ordered_pairs):
        # Empty lists and arrays for...
        # ... tracking connected components,
        ids = np.full((logRho.size,), logRho.size, dtype = np.uint32)
        count = 0
        aggregations = [[np.uint32(0) for i in range(0)] for i in range(0)]
        # ... freeing up memory,
        emptyIntList = [np.uint32(0) for i in range(0)]
        # ... and tracking groups.
        groups = [[np.uint32(0) for i in range(0)] for i in range(0)]
        prominences = [[np.float32(0.0) for i in range(0)] for i in range(0)]
        children = [[np.uint32(0) for i in range(0)] for i in range(0)]

        # Kruskal's minimum spanning tree + hierarchy tracking
        for pair in ordered_pairs:
            id_0, id_1 = ids[pair]
            if id_0 != logRho.size: # pair[0] is already aggregated
                if id_0 == id_1: pass # Same group
                elif id_1 == logRho.size: # pair[1] is not yet aggregated
                    p_1 = pair[1]
                    ids[p_1] = id_0
                    aggregations[id_0].append(p_1)
                    groups[id_0][2] += 1
                else: # Different groups -> merge the smaller group into the larger group
                    # Make id_0 correspond to the larger group
                    if groups[id_0][2] < groups[id_1][2]: id_0, id_1 = id_1, id_0
                    
                    # Update the ids of the smaller group
                    for id_i in aggregations[id_1]: ids[id_i] = id_0
                    aggregations[id_0].extend(aggregations[id_1])
                    aggregations[id_1] = emptyIntList
                    currLogRho = logRho[pair[1]]

                    # Merge groups
                    groups[id_1][:2] = groups[id_0][1:]
                    groups[id_0][2] += groups[id_1][2]
                    prominences[id_1][0] = prominences[id_0][1] - currLogRho
                    prominences[id_0][1] = max(prominences[id_0][1], prominences[id_1][1])
                    prominences[id_1][1] -= currLogRho
                    children[id_0].append(id_1)
            elif id_1 == logRho.size: # Neither are aggregated
                ids[pair] = count
                count += 1
                aggregations.append([pair[0], pair[1]])

                # Create new group
                groups.append([np.uint32(0), np.uint32(0), np.uint32(2)])
                prominences.append([np.float32(0.0), logRho[pair[0]]])
                children.append([np.uint32(0) for i in range(0)])
            else: # pair[1] is already aggregated (but not pair[0])
                p_0 = pair[0]
                ids[p_0] = id_1
                aggregations[id_1].append(p_0)
                groups[id_1][2] += 1
                prominences[id_1][1] = max(prominences[id_1][1], logRho[p_0])

        # Check if all points were aggregated together
        aggArr = np.unique(ids)
        emptyIntArr = np.empty(0, dtype = np.uint32)
        ids = emptyIntArr
        if aggArr.size == 1: id_final = aggArr[0]
        else: # If points were not all aggregated together, merge them in order of decreasing size.
            sortedAggregations = sorted(zip([groups[id_i][2] for id_i in aggArr], aggArr))
            _, id_final = sortedAggregations[-1]
            for size_leq, id_leq in sortedAggregations[-2::-1]:
                aggregations[id_final].extend(aggregations[id_leq])
                aggregations[id_leq] = emptyIntList

                # Track complementary group
                groups[id_leq][:2] = groups[id_final][1:]
                prominences[id_leq][0] = prominences[id_final][1]
                
                # Merge
                groups[id_final][2] += size_leq
                children[id_final].append(id_leq)
        aggArr = emptyIntArr

        # Ordered list
        ordering = np.array(aggregations[id_final], dtype = np.uint32)
        aggregations[id_final] = emptyIntList

        # Lists to Arrays
        groups = np.array(groups, dtype = np.uint32)
        prominences = np.array(prominences, dtype = np.float32)

        # Finalise groups and correct for noise
        activeGroups = [id_final]
        while activeGroups:
            id_leq = activeGroups.pop()
            childIDs = children[id_leq]
            if childIDs:
                startAdjust = groups[id_leq][1]
                activeGroups.extend(childIDs)
                noise = 0.0
                for id_geq, childID in enumerate(childIDs):
                    groups[childID][:2] += startAdjust
                    if id_geq > 0: prominences[childID, 0] -= np.sqrt(noise/id_geq)
                    noise += prominences[childID, 1]**2
                prominences[id_leq, 0] -= np.sqrt(noise/(id_geq + 1))
                children[id_leq] = emptyIntList

        # Clean and reorder arrays
        keepBool = groups[:, 2] > 2
        groups = groups[keepBool]
        prominences = prominences[keepBool]
        groups[:, 2] += groups[:, 1]
        reorder = groups[:, 1].argsort()[1:]
        return ordering, groups[reorder], prominences[reorder]
    
    @staticmethod
    @njit(fastmath = True)
    def _aggregate_njit_float64_uint32(logRho, ordered_pairs):
        # Empty lists and arrays for...
        # ... tracking connected components,
        ids = np.full((logRho.size,), logRho.size, dtype = np.uint32)
        count = 0
        aggregations = [[np.uint32(0) for i in range(0)] for i in range(0)]
        # ... tracking groups,
        groups = [[np.uint32(0) for i in range(0)] for i in range(0)]
        prominences = [[np.float64(0.0) for i in range(0)] for i in range(0)]
        children = [[np.uint32(0) for i in range(0)] for i in range(0)]
        # ... and freeing up memory.
        emptyIntList = [np.uint32(0) for i in range(0)]

        # Kruskal's minimum spanning tree + hierarchy tracking
        for pair in ordered_pairs:
            id_0, id_1 = ids[pair]
            if id_0 != logRho.size: # pair[0] is already aggregated
                if id_0 == id_1: pass # Same group
                elif id_1 == logRho.size: # pair[1] is not yet aggregated
                    p_1 = pair[1]
                    ids[p_1] = id_0
                    aggregations[id_0].append(p_1)
                    groups[id_0][2] += 1
                else: # Different groups -> merge the smaller group into the larger group
                    # Make id_0 correspond to the larger group
                    if groups[id_0][2] < groups[id_1][2]: id_0, id_1 = id_1, id_0
                    
                    # Update the ids of the smaller group
                    for id_i in aggregations[id_1]: ids[id_i] = id_0
                    aggregations[id_0].extend(aggregations[id_1])
                    aggregations[id_1] = emptyIntList
                    currLogRho = logRho[pair[1]]

                    # Merge groups
                    groups[id_1][:2] = groups[id_0][1:]
                    groups[id_0][2] += groups[id_1][2]
                    prominences[id_1][0] = prominences[id_0][1] - currLogRho
                    prominences[id_0][1] = max(prominences[id_0][1], prominences[id_1][1])
                    prominences[id_1][1] -= currLogRho
                    children[id_0].append(id_1)
            elif id_1 == logRho.size: # Neither are aggregated
                ids[pair] = count
                count += 1
                aggregations.append([pair[0], pair[1]])

                # Create new group
                groups.append([np.uint32(0), np.uint32(0), np.uint32(2)])
                prominences.append([np.float64(0.0), logRho[pair[0]]])
                children.append([np.uint32(0) for i in range(0)])
            else: # pair[1] is already aggregated (but not pair[0])
                p_0 = pair[0]
                ids[p_0] = id_1
                aggregations[id_1].append(p_0)
                groups[id_1][2] += 1
                prominences[id_1][1] = max(prominences[id_1][1], logRho[p_0])

        # Check if all points were aggregated together
        aggArr = np.unique(ids)
        emptyIntArr = np.empty(0, dtype = np.uint32)
        ids = emptyIntArr
        if aggArr.size == 1: id_final = aggArr[0]
        else: # If points were not all aggregated together, merge them in order of decreasing size.
            sortedAggregations = sorted(zip([groups[id_i][2] for id_i in aggArr], aggArr))
            _, id_final = sortedAggregations[-1]
            for size_leq, id_leq in sortedAggregations[-2::-1]:
                aggregations[id_final].extend(aggregations[id_leq])
                aggregations[id_leq] = emptyIntList

                # Track complementary group
                groups[id_leq][:2] = groups[id_final][1:]
                prominences[id_leq][0] = prominences[id_final][1]
                
                # Merge
                groups[id_final][2] += size_leq
                children[id_final].append(id_leq)
        aggArr = emptyIntArr

        # Ordered list
        ordering = np.array(aggregations[id_final], dtype = np.uint32)
        aggregations[id_final] = emptyIntList

        # Lists to Arrays
        groups = np.array(groups, dtype = np.uint32)
        prominences = np.array(prominences, dtype = np.float64)

        # Finalise groups and correct for noise
        activeGroups = [id_final]
        while activeGroups:
            id_leq = activeGroups.pop()
            childIDs = children[id_leq]
            if childIDs:
                startAdjust = groups[id_leq][1]
                activeGroups.extend(childIDs)
                noise = 0.0
                for id_geq, childID in enumerate(childIDs):
                    groups[childID][:2] += startAdjust
                    if id_geq > 0: prominences[childID, 0] -= np.sqrt(noise/id_geq)
                    noise += prominences[childID, 1]**2
                prominences[id_leq, 1] -= np.sqrt(noise/(id_geq + 1))
                children[id_leq] = emptyIntList

        # Clean and reorder arrays
        keepBool = groups[:, 2] > 2
        groups = groups[keepBool]
        prominences = prominences[keepBool]
        groups[:, 2] += groups[:, 1]
        reorder = groups[:, 1].argsort()[1:]
        return ordering, groups[reorder], prominences[reorder]

    @staticmethod
    @njit(fastmath = True)
    def _aggregate_njit_float32_uint64(logRho, ordered_pairs):
        # Empty lists and arrays for...
        # ... tracking connected components,
        ids = np.full((logRho.size,), logRho.size, dtype = np.uint64)
        count = 0
        aggregations = [[np.uint64(0) for i in range(0)] for i in range(0)]
        # ... freeing up memory,
        emptyIntList = [np.uint64(0) for i in range(0)]
        # ... and tracking groups.
        groups = [[np.uint64(0) for i in range(0)] for i in range(0)]
        prominences = [[np.float32(0.0) for i in range(0)] for i in range(0)]
        children = [[np.uint64(0) for i in range(0)] for i in range(0)]

        # Kruskal's minimum spanning tree + hierarchy tracking
        for pair in ordered_pairs:
            id_0, id_1 = ids[pair]
            if id_0 != logRho.size: # pair[0] is already aggregated
                if id_0 == id_1: pass # Same group
                elif id_1 == logRho.size: # pair[1] is not yet aggregated
                    p_1 = pair[1]
                    ids[p_1] = id_0
                    aggregations[id_0].append(p_1)
                    groups[id_0][2] += 1
                else: # Different groups -> merge the smaller group into the larger group
                    # Make id_0 correspond to the larger group
                    if groups[id_0][2] < groups[id_1][2]: id_0, id_1 = id_1, id_0
                    
                    # Update the ids of the smaller group
                    for id_i in aggregations[id_1]: ids[id_i] = id_0
                    aggregations[id_0].extend(aggregations[id_1])
                    aggregations[id_1] = emptyIntList
                    currLogRho = logRho[pair[1]]

                    # Merge groups
                    groups[id_1][:2] = groups[id_0][1:]
                    groups[id_0][2] += groups[id_1][2]
                    prominences[id_1][0] = prominences[id_0][1] - currLogRho
                    prominences[id_0][1] = max(prominences[id_0][1], prominences[id_1][1])
                    prominences[id_1][1] -= currLogRho
                    children[id_0].append(id_1)
            elif id_1 == logRho.size: # Neither are aggregated
                ids[pair] = count
                count += 1
                aggregations.append([pair[0], pair[1]])

                # Create new group
                groups.append([np.uint64(0), np.uint64(0), np.uint64(2)])
                prominences.append([np.float32(0.0), logRho[pair[0]]])
                children.append([np.uint64(0) for i in range(0)])
            else: # pair[1] is already aggregated (but not pair[0])
                p_0 = pair[0]
                ids[p_0] = id_1
                aggregations[id_1].append(p_0)
                groups[id_1][2] += 1
                prominences[id_1][1] = max(prominences[id_1][1], logRho[p_0])

        # Check if all points were aggregated together
        aggArr = np.unique(ids)
        emptyIntArr = np.empty(0, dtype = np.uint64)
        ids = emptyIntArr
        if aggArr.size == 1: id_final = aggArr[0]
        else: # If points were not all aggregated together, merge them in order of decreasing size.
            sortedAggregations = sorted(zip([groups[id_i][2] for id_i in aggArr], aggArr))
            _, id_final = sortedAggregations[-1]
            for size_leq, id_leq in sortedAggregations[-2::-1]:
                aggregations[id_final].extend(aggregations[id_leq])
                aggregations[id_leq] = emptyIntList

                # Track complementary group
                groups[id_leq][:2] = groups[id_final][1:]
                prominences[id_leq][0] = prominences[id_final][1]
                
                # Merge
                groups[id_final][2] += size_leq
                children[id_final].append(id_leq)
        aggArr = emptyIntArr

        # Ordered list
        ordering = np.array(aggregations[id_final], dtype = np.uint64)
        aggregations[id_final] = emptyIntList

        # Lists to Arrays
        groups = np.array(groups, dtype = np.uint64)
        prominences = np.array(prominences, dtype = np.float32)

        # Finalise groups and correct for noise
        activeGroups = [id_final]
        while activeGroups:
            id_leq = activeGroups.pop()
            childIDs = children[id_leq]
            if childIDs:
                startAdjust = groups[id_leq][1]
                activeGroups.extend(childIDs)
                noise = 0.0
                for id_geq, childID in enumerate(childIDs):
                    groups[childID][:2] += startAdjust
                    if id_geq > 0: prominences[childID, 0] -= np.sqrt(noise/id_geq)
                    noise += prominences[childID, 1]**2
                prominences[id_leq, 0] -= np.sqrt(noise/(id_geq + 1))
                children[id_leq] = emptyIntList

        # Clean and reorder arrays
        keepBool = groups[:, 2] > 2
        groups = groups[keepBool]
        prominences = prominences[keepBool]
        groups[:, 2] += groups[:, 1]
        reorder = groups[:, 1].argsort()[1:]
        return ordering, groups[reorder], prominences[reorder]

    @staticmethod
    @njit(fastmath = True)
    def _aggregate_njit_float64_uint64(logRho, ordered_pairs):
        # Empty lists and arrays for...
        # ... tracking connected components,
        ids = np.full((logRho.size,), logRho.size, dtype = np.uint64)
        count = 0
        aggregations = [[np.uint64(0) for i in range(0)] for i in range(0)]
        # ... freeing up memory,
        emptyIntList = [np.uint64(0) for i in range(0)]
        # ... and tracking groups.
        groups = [[np.uint64(0) for i in range(0)] for i in range(0)]
        prominences = [[np.float64(0.0) for i in range(0)] for i in range(0)]
        children = [[np.uint64(0) for i in range(0)] for i in range(0)]

        # Kruskal's minimum spanning tree + hierarchy tracking
        for pair in ordered_pairs:
            id_0, id_1 = ids[pair]
            if id_0 != logRho.size: # pair[0] is already aggregated
                if id_0 == id_1: pass # Same group
                elif id_1 == logRho.size: # pair[1] is not yet aggregated
                    p_1 = pair[1]
                    ids[p_1] = id_0
                    aggregations[id_0].append(p_1)
                    groups[id_0][2] += 1
                else: # Different groups -> merge the smaller group into the larger group
                    # Make id_0 correspond to the larger group
                    if groups[id_0][2] < groups[id_1][2]: id_0, id_1 = id_1, id_0
                    
                    # Update the ids of the smaller group
                    for id_i in aggregations[id_1]: ids[id_i] = id_0
                    aggregations[id_0].extend(aggregations[id_1])
                    aggregations[id_1] = emptyIntList
                    currLogRho = logRho[pair[1]]

                    # Merge groups
                    groups[id_1][:2] = groups[id_0][1:]
                    groups[id_0][2] += groups[id_1][2]
                    prominences[id_1][0] = prominences[id_0][1] - currLogRho
                    prominences[id_0][1] = max(prominences[id_0][1], prominences[id_1][1])
                    prominences[id_1][1] -= currLogRho
                    children[id_0].append(id_1)
            elif id_1 == logRho.size: # Neither are aggregated
                ids[pair] = count
                count += 1
                aggregations.append([pair[0], pair[1]])

                # Create new group
                groups.append([np.uint64(0), np.uint64(0), np.uint64(2)])
                prominences.append([np.float64(0.0), logRho[pair[0]]])
                children.append([np.uint64(0) for i in range(0)])
            else: # pair[1] is already aggregated (but not pair[0])
                p_0 = pair[0]
                ids[p_0] = id_1
                aggregations[id_1].append(p_0)
                groups[id_1][2] += 1
                prominences[id_1][1] = max(prominences[id_1][1], logRho[p_0])

        # Check if all points were aggregated together
        aggArr = np.unique(ids)
        emptyIntArr = np.empty(0, dtype = np.uint64)
        ids = emptyIntArr
        if aggArr.size == 1: id_final = aggArr[0]
        else: # If points were not all aggregated together, merge them in order of decreasing size.
            sortedAggregations = sorted(zip([groups[id_i][2] for id_i in aggArr], aggArr))
            _, id_final = sortedAggregations[-1]
            for size_leq, id_leq in sortedAggregations[-2::-1]:
                aggregations[id_final].extend(aggregations[id_leq])
                aggregations[id_leq] = emptyIntList

                # Track complementary group
                groups[id_leq][:2] = groups[id_final][1:]
                prominences[id_leq][0] = prominences[id_final][1]
                
                # Merge
                groups[id_final][2] += size_leq
                children[id_final].append(id_leq)
        aggArr = emptyIntArr

        # Ordered list
        ordering = np.array(aggregations[id_final], dtype = np.uint64)
        aggregations[id_final] = emptyIntList

        # Lists to Arrays
        groups = np.array(groups, dtype = np.uint64)
        prominences = np.array(prominences, dtype = np.float64)

        # Finalise groups and correct for noise
        activeGroups = [id_final]
        while activeGroups:
            id_leq = activeGroups.pop()
            childIDs = children[id_leq]
            if childIDs:
                startAdjust = groups[id_leq][1]
                activeGroups.extend(childIDs)
                noise = 0.0
                for id_geq, childID in enumerate(childIDs):
                    groups[childID][:2] += startAdjust
                    if id_geq > 0: prominences[childID, 0] -= np.sqrt(noise/id_geq)
                    noise += prominences[childID, 1]**2
                prominences[id_leq, 0] -= np.sqrt(noise/(id_geq + 1))
                children[id_leq] = emptyIntList

        # Clean and reorder arrays
        keepBool = groups[:, 2] > 2
        groups = groups[keepBool]
        prominences = prominences[keepBool]
        groups[:, 2] += groups[:, 1]
        reorder = groups[:, 1].argsort()[1:]
        return ordering, groups[reorder], prominences[reorder]

    def compute_significances(self):
        """Computes statistical significances for all groups by fitting a
        descriptive model to their prominences.

        Fits the prominence model (a Beta distribution) by minimising the 
        Wasserstein-1 distance between the numerical and empirical cdfs. The 
        noise model (with model-fitted parameters) is then used alongside the 
        standard normal distribution to transform prominence values into 
        statistical significance values.

        The method requires the `prominences` attribute to have already been 
        created, via the `aggregate` method or otherwise.

        This method generates the `groups_sigs` and `pFit` attributes.
        """

        if self.verbose > 1: self._printFunction('Finding significances...  ')
        start = time.perf_counter()

        # Setup for model fitting
        self.pFit, modelArgs, modelBounds, tol = self._minimize_init_njit(self.prominences[:, 1])

        # Fit model
        sol = minimize(self._wasserstein1_beta, self.pFit, args = tuple(modelArgs),
                       jac = '3-point', bounds = tuple(modelBounds), tol = tol)
        if sol.success: self.pFit = sol.x
        else: self._printFunction('[Warning] Prominence model fitting failed to converge!', returnLine = False, urgent = True)
        del modelArgs, modelBounds, tol

        # Calculate statistical significance values
        self.groups_sigs = norm.isf(beta.sf(self.prominences, self.pFit[0], self.pFit[1]))

        self._regrTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True)
    def _minimize_init_njit(prominences):
        # Sort prominences to help with calculations
        prominence_sorted = np.sort(prominences)

        # Initial guess for Beta model parameters
        binwidth = 2*(prominence_sorted[3*prominences.size//4] - prominence_sorted[prominences.size//4])*prominences.size**(-1/3) # Freedman-Diaconis rule
        mode = binwidth*(np.bincount((prominence_sorted//binwidth).astype(np.int64)).argmax() + 0.5)
        mu, var = prominence_sorted.mean(), prominence_sorted.var()
        term1 = 1 - mu
        term2 = mu*term1/var - 1
        a1, b1 = mu*term2, term1*term2
        a2 = (2*mode - 1)*mu/(term1*mode - (1 - mode)*mu)
        b2 = a2*term1/mu
        modelParams = np.array([0.5*(a1 + a2), 0.5*(b1 + b2)])

        # Precalculate terms for Wasserstein-1 distance with model
        midPoints = np.concatenate((np.zeros(1), prominence_sorted))
        widths = midPoints[1:] - midPoints[:-1]
        midPoints = (midPoints[1:] + midPoints[:-1])/2
        oneMinusMidPoints = 1 - midPoints
        halfInterval = 1/(2*prominences.size)
        quantiles = np.linspace(halfInterval, 1 - halfInterval, prominences.size)
        modelArgs = [midPoints, oneMinusMidPoints, widths, quantiles]

        # Bounds for Beta model parameters
        modelBounds = [(1, np.inf), (1, np.inf)]

        # Tolerance for minimisation
        tol = 10**(-np.ceil(np.log10(prominences.size)) - 3)

        return modelParams, modelArgs, modelBounds, tol

    def _wasserstein1_beta(self, p, midPoints, oneMinusMidPoints, widths, quantiles):
        return self._wasserstein1_beta_njit(p, midPoints, oneMinusMidPoints, widths, quantiles, beta_fun(p[0], p[1]))

    @staticmethod
    @njit()
    def _wasserstein1_beta_njit(p, midPoints, oneMinusMidPoints, widths, quantiles, norm_constant):
        # Calculate numerical cdf
        pdf = midPoints**(p[0] - 1)*oneMinusMidPoints**(p[1] - 1)/norm_constant
        cdf = np.cumsum(pdf*widths)

        # Return Wasserstein-1 distance between the numerical and empirical cdfs
        return np.sum(np.abs(cdf - quantiles)*widths)

    def extract_clusters(self):
        """Classifies groups that have significance of at least `S` as clusters
        and forms the hierarchy according to the parameter `h_style`.

        First classifies any groups that are statistical outliers. Then if
        `h_style` is set to 1, finds the corresponding groups that are also
        statistical outliers. For each of these that are the smallest out of
        those that share the same starting position within the ordered list,
        classify them too as clusters. The input data is always classified as 
        the root cluster in addition to these overdensities. Finally, generate 
        the array of id strings for the clusters.

        This method requires the `groups` attribute to have already been 
        created, via the `aggregate()` method or otherwise. It also requires the 
        `groups_sigs` and `pFit` attributes to have already been created, via
        the `compute_significance` method or otherwise.

        This method generates the 'clusters', `ids`, and `significances`
        attributes.
        """

        if self.verbose > 1: self._printFunction('Finding clusters...       ')
        start = time.perf_counter()

        # Classify clusters as groups that are significant outliers
        if self.S == 'auto': self.S = norm.isf(1 - 0.5**(1/self.prominences.shape[0]))
        sl = self.groups_sigs[:, 1] >= self.S
        self.clusters = self.groups[sl, 1:]
        self.significances = self.groups_sigs[sl, 1]

        # Optional hierarchy correction
        if self.h_style == 1:
            # Retrieve complementary groups whose corresponding subgroup is
            # significantly clustered and who is itself significantly clustered
            sl = np.logical_and(sl, self.groups_sigs[:, 0] >= self.S)
            significances_geq = self.groups_sigs[sl, 0]
            clusters_geq = self.groups[sl, :2]

            # Keep only those complementary groups that are the smallest in their cascade
            sl = np.zeros(clusters_geq.shape[0], dtype = np.bool_)
            cascade_starts_unique = np.unique(clusters_geq[:, 0])
            for cascade_start in cascade_starts_unique:
                sl[np.where(clusters_geq[:, 0] == cascade_start)[0][0]] = 1

            # Merge the hierarchy and clean arrays
            self.clusters = np.vstack((self.clusters, clusters_geq[sl]))
            self.significances = np.concatenate((self.significances, significances_geq[sl]))
            reorder = np.lexsort((np.arange(self.clusters.shape[0]), self.n_samples - self.clusters[:, 1], self.clusters[:, 0]))
            self.clusters = self.clusters[reorder]
            self.significances = self.significances[reorder]

        # Add on root-level cluster
        self.significances = np.concatenate((np.array([np.inf]), self.significances))
        self.clusters = np.vstack((np.array([[0, self.n_samples]], dtype = np.uint32), self.clusters))
        self.ids = ['1']

        # Label clusters according to their hierarchy
        currentParents = [0]
        children = np.zeros(self.clusters.shape[0], dtype = np.uint32)
        for i, clst in enumerate(self.clusters[1:]):
            # Search through hierarchy to find parent cluster
            while clst[0] >= self.clusters[currentParents[-1]][1]:
                currentParents.pop(-1)
            parent = currentParents[-1]
            currentParents.append(i + 1)

            # Assign cluster id to child cluster of parent
            children[parent] += 1
            self.ids.append(f"{self.ids[parent]}-{children[parent]}")
        self.ids = np.array(self.ids)

        self._rejTime = time.perf_counter() - start