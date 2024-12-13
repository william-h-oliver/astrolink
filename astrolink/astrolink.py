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
from scipy.special import erf
from scipy.stats import norm, beta, halfnorm, lognorm
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
    pFit : `numpy.ndarray` of shape (2,) or (3,)
        The model parameters for the model that fits the distribution of 
        `prominences`. If the best-fitting noise model is 'Beta', then  `pFit` 
        has the form `[c, a, b]`, where `a` and `b` are the shape parameters for 
        the Beta distribution. If the best-fitting noise model is 'Half-Normal', 
        then `pFit` has the form `[c, sigma]`, where `sigma` is the standard 
        deviation of the Half-Normal distribution. If the best-fitting noise 
        model is 'Log-Normal', then `pFit` has the form `[c, mu, sigma]`, where 
        `mu` and `sigma` are the mean and standard deviation of the Log-Normal 
        distribution. In all cases, `c` is the cutoff between the noise model 
        and Uniform distribution used to prevent overfitting of the noise model 
        to sufficiently clustered overdensities. The best-fitting noise model, 
        the fitted parameters of all models, the success states of the fitting 
        procedure on each model, and the second-order Akaike Information 
        Criterion (AICc) values for each model can be found with the 
        `_noiseModel`, `_modelParams`, `_modelSuccess`, and `_modelAICc` 
        attibutes respectively.
    """

    def __init__(self, P, weights = None, k_den = 20, adaptive = 1, S = 'auto', k_link = 'auto', h_style = 1, workers = 8, verbose = 0):
        # Input Data
        check_P = isinstance(P, np.ndarray) and len(P.shape) == 2
        assert check_P, "Input data 'P' needs to be a 2D numpy array!"
        self.P = P
        self.n_samples, self.d_features = self.P.shape

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
        if k_link == 'auto': self.k_link = max(int(np.ceil(11.97*self.d_features**(-2.23) - 22.97*self.k_den**(-0.57) + 10.03)), 7) # Fitted params [ 11.97420072  -2.22754059 -22.96660813  -0.56948651   9.03446908]
        else: self.k_link = k_link

        check_h_style = h_style in [0, 1]
        assert check_h_style, "Parameter 'h_style' must be set as either '0' or '1'!"
        self.h_style = h_style

        check_workers = 1 <= workers or workers == -1
        assert check_workers, f"Parameter 'workers' must be set as either '-1' or needs to be an integer that is >= 1 (values > N_cpu will be set to N_cpu)"
        os.environ["OMP_NUM_THREADS"] = f"{min(workers, os.cpu_count())}" if workers != -1 else f"{os.cpu_count()}"
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
        self.logRho = np.empty_like(self.P_transform, shape = (self.n_samples,))
        self.kNN = np.empty((self.n_samples, self.k_link), dtype = np.uint32)
        nbrs = KDTree(self.P_transform)
        working_memory = get_config()["working_memory"]
        chunk_n_rows = max(min(int(working_memory * (2**20) // 16*self.k_den), self.n_samples), 1)
        for sl in gen_batches(self.n_samples, chunk_n_rows):
            sqr_distances, indices = nbrs.query(self.P_transform[sl], k = self.k_den, sqr_dists = True)
            if self.weights is None: self.logRho[sl] = self._compute_logRho(sqr_distances, self.k_den, self.d_features)
            else: self.logRho[sl] = self._compute_weightedLogRho(sqr_distances, self.weights[indices], self.d_features)
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
    @njit()
    def _compute_weightedLogRho(sqr_distances, weights, d_features):
        coreSqrDist = sqr_distances[:, -1]
        return np.log((weights.sum(axis = 1) - (weights*sqr_distances).sum(axis = 1)/coreSqrDist)/coreSqrDist**(d_features/2))

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
        if self.logRho.dtype == np.float64:
            ordered_pairs = self._order_pairs_njit_float64(self.logRho, self.kNN)
            del self.kNN
            self.ordering, self.groups, self.prominences = self._aggregate_njit_float64(self.logRho, ordered_pairs)
        else:
            ordered_pairs = self._order_pairs_njit_float32(self.logRho, self.kNN)
            del self.kNN
            self.ordering, self.groups, self.prominences = self._aggregate_njit_float32(self.logRho, ordered_pairs)
        self._aggregateTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _order_pairs_njit_float64(logRho, kNN):
        # Create empty arrays
        n_samples, k_link = kNN.shape
        graphShape = n_samples*(k_link - 1)
        edges = np.empty(graphShape, dtype = np.float64)
        ordered_pairs = np.empty((graphShape, 2), dtype = np.uint32)

        # For each pair of vertices find the adjoining edge weight
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

        # Order points by descending edge weight
        return ordered_pairs[edges.argsort()[::-1]]

    @staticmethod
    @njit(fastmath = True)
    def _aggregate_njit_float64(logRho, ordered_pairs):
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
                    groups[childID][0] += startAdjust
                    groups[childID][1] += startAdjust
                    if id_geq > 0: prominences[childID, 0] -= np.sqrt(noise/id_geq)
                    noise += prominences[childID, 1]**2
                prominences[id_leq, 1] -= np.sqrt(noise/(id_geq + 1))
                children[id_leq] = emptyIntList

        # Clean and reorder arrays
        groups[:, 2] += groups[:, 1]
        reorder = groups[:, 1].argsort()[1:]
        return ordering, groups[reorder], prominences[reorder]

    @staticmethod
    @njit(fastmath = True, parallel = True)
    def _order_pairs_njit_float32(logRho, kNN):
        # Create empty arrays
        n_samples, k_link = kNN.shape
        graphShape = n_samples*(k_link - 1)
        edges = np.empty(graphShape, dtype = np.float32)
        ordered_pairs = np.empty((graphShape, 2), dtype = np.uint32)

        # For each pair of vertices find the adjoining edge weight
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
                    
        # Order points by descending edge weight
        return ordered_pairs[edges.argsort()[::-1]]

    @staticmethod
    @njit(fastmath = True)
    def _aggregate_njit_float32(logRho, ordered_pairs):
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
                    groups[childID][0] += startAdjust
                    groups[childID][1] += startAdjust
                    if id_geq > 0: prominences[childID, 0] -= np.sqrt(noise/id_geq)
                    noise += prominences[childID, 1]**2
                prominences[id_leq, 0] -= np.sqrt(noise/(id_geq + 1))
                children[id_leq] = emptyIntList

        # Clean and reorder arrays
        groups[:, 2] += groups[:, 1]
        reorder = groups[:, 1].argsort()[1:]
        return ordering, groups[reorder], prominences[reorder]

    def compute_significances(self):
        """Computes statistical significances for all groups by fitting a
        descriptive model to the prominences of groups.

        Constructs the subgroup prominence model (a combination of a noise model 
        and a Uniform distribution) and fits it by minimising the negative
        log-likelihood of the resulting probability distribution. The noise 
        model (with model-fitted parameters) is then used alongside the standard 
        normal distribution to transform prominence values into statistical 
        significance values. The noise model is chosen from among the Beta, 
        Half-normal, and Log-normal distributions using the second-order Akaike 
        Information Criterion (AICc).

        The method requires the `prominences` attribute to have already been 
        created, via the `aggregate` method or otherwise.

        This method generates the `group_sigs` and `pFit` attributes.
        """

        if self.verbose > 1: self._printFunction('Finding significances...  ')
        start = time.perf_counter()

        # Setup for model fitting
        self._modelParams, modelArgs, modelBounds, tol, self._modelAICc = self._minimize_init(self.prominences[:, 1])

        # Fit models
        modelNegLLs = [self._negLL_beta, self._negLL_halfnormal, self._negLL_lognormal]
        self._modelSuccess = np.ones(self._modelAICc.size, dtype = np.bool_)
        for i, (negLL, params, args, bounds) in enumerate(zip(modelNegLLs, self._modelParams, modelArgs, modelBounds)):
            sol = minimize(negLL, params, args = tuple(args), jac = '3-point', bounds = tuple(bounds), tol = tol)
            if sol.success: self._modelParams[i] = sol.x
            else: self._modelSuccess[i] = False
            self._modelAICc[i] += 2*sol.fun
        del modelNegLLs, modelArgs, modelBounds, tol

        if not self._modelSuccess.any(): self._printFunction('[Warning] Prominence model may be incorrectly fitted!', returnLine = False, urgent = True)

        # Choose the best model and calculate statistical significance values
        bestModel = np.argmin(self._modelAICc)
        self.pFit = self._modelParams[bestModel]
        noiseModels = ['Beta', 'Half-normal', 'Log-normal']
        self._noiseModel = noiseModels[bestModel]
        if self._noiseModel == 'Beta':
            self.groups_sigs = norm.isf(beta.sf(self.prominences, self.pFit[1], self.pFit[2]))
        elif self._noiseModel == 'Half-normal':
            self.groups_sigs = norm.isf(halfnorm.sf(self.prominences, 0, self.pFit[1]))
        elif self._noiseModel == 'Log-normal':
            self.groups_sigs = norm.isf(lognorm.sf(self.prominences, self.pFit[2], scale = np.exp(self.pFit[1])))

        self._regrTime = time.perf_counter() - start

    @staticmethod
    @njit(fastmath = True)
    def _minimize_init(prominences):
        # Precalculated properties of the prominence values
        promOrd = np.sort(prominences)
        cutOff = promOrd[int(0.99*promOrd.size)]
        mu, var = promOrd.mean(), promOrd.var()
        tol = 1e-5
        
        # Precalculated transforms of the prominence values
        promOrdOpenBorder = promOrd[np.logical_and(promOrd > 0, promOrd < 1)]
        lnx = np.log(promOrdOpenBorder)
        lnx_cumsum = lnx.cumsum()
        lnx_sqrd_cumsum = (lnx**2).cumsum()
        ln_1_minus_x_cumsum = np.log(1 - promOrdOpenBorder).cumsum()
        x_sqrd_cumsum = (promOrd**2).cumsum()
        
        # For Beta model
        term1 = 1 - mu
        term2 = mu*term1/var - 1
        modelParams = [np.array([cutOff, term2*mu, term2*term1])]
        modelArgs = [[promOrdOpenBorder, lnx_cumsum, ln_1_minus_x_cumsum]]
        modelBounds = [[(tol, 1 - tol), (1 + tol, np.inf), (1, np.inf)]]
        
        # For Half-normal model
        modelParams.append(np.array([cutOff, np.sqrt(np.pi/2)*mu]))
        modelArgs.append([promOrd, x_sqrd_cumsum])
        modelBounds.append([(tol, 1 - tol), (tol, np.inf)])
        
        # For log-normal model
        muSqr = mu**2
        modelParams.append(np.array([cutOff, np.log(muSqr/np.sqrt(muSqr + var)), np.sqrt(np.log(1 + var/muSqr))]))
        modelArgs.append([promOrdOpenBorder, lnx_cumsum, lnx_sqrd_cumsum])
        modelBounds.append([(tol, 1 - tol), (-np.inf, np.inf), (tol, np.inf)])

        # Second-order correction of the Akaike information criterion
        modelAICc = np.zeros(len(modelParams))
        for i, (params, args) in enumerate(zip(modelParams, modelArgs)):
            modelAICc[i] = 2*params.size + 2*params.size*(params.size + 1)/(args[0].size - params.size - 1)

        return modelParams, modelArgs, modelBounds, tol, modelAICc

    def _negLL_beta(self, p, promOrdOpenBorder, lnx_cumsum, ln_1_minus_x_cumsum):
        return self._negLL_beta_njit(p, promOrdOpenBorder, lnx_cumsum, ln_1_minus_x_cumsum, beta_fun(p[1], p[2])*betainc_fun(p[1], p[2], p[0]))

    @staticmethod
    @njit()
    def _negLL_beta_njit(p, promOrdOpenBorder, lnx_cumsum, ln_1_minus_x_cumsum, norm_constant):
        c, a, b = p
        n = promOrdOpenBorder.size
        transitionPoint, right = 0, lnx_cumsum.size - 1
        while right - transitionPoint > 1:
            middle = (right - transitionPoint)//2 + transitionPoint
            if promOrdOpenBorder[middle] < c: transitionPoint = middle
            else: right = middle
        uniform_term = (c**(a - 1))*((1 - c)**(b - 1))
        normalisationFactor = norm_constant + uniform_term*(1 - c)
        return n*np.log(normalisationFactor) - (a - 1)*lnx_cumsum[transitionPoint] - (b - 1)*ln_1_minus_x_cumsum[transitionPoint] - (n - transitionPoint - 1)*np.log(uniform_term)

    def _negLL_halfnormal(self, p, promOrd, x_sqrd_cumsum):
        return self._negLL_halfnormal_njit(p, promOrd, x_sqrd_cumsum, erf(p[0]/(np.sqrt(2)*p[1])))

    @staticmethod
    @njit()
    def _negLL_halfnormal_njit(p, promOrd, x_sqrd_cumsum, erfTerm):
        c, sigma = p
        cSqr, twoSigmaSqr = c**2, 2*sigma**2
        n = promOrd.size
        # Find the index of the first prominence that is greater than c
        transitionPoint, right = 0, n - 1
        while right - transitionPoint > 1:
            middle = (right - transitionPoint)//2 + transitionPoint
            if promOrd[middle] < c: transitionPoint = middle
            else: right = middle
        return n*np.log(np.sqrt(np.pi/2)*sigma*erfTerm + (1 - c)*np.exp(-cSqr/twoSigmaSqr)) + (x_sqrd_cumsum[transitionPoint] + (n - transitionPoint - 1)*cSqr)/twoSigmaSqr

    def _negLL_lognormal(self, p, promOrdOpenBorder, lnx_cumsum, lnx_sqrd_cumsum):
        return self._negLL_lognormal_njit(p, promOrdOpenBorder, lnx_cumsum, lnx_sqrd_cumsum, erf((np.log(p[0]) - p[1])/(np.sqrt(2)*p[2])))

    @staticmethod
    @njit()
    def _negLL_lognormal_njit(p, promOrdOpenBorder, lnx_cumsum, lnx_sqrd_cumsum, erfTerm):
        c, mu, sigma = p
        logc, twoSigmaSqr = np.log(c), 2*sigma**2
        n = promOrdOpenBorder.size
        # Find the index of the first prominence that is greater than c
        transitionPoint, right = 0, n - 1
        while right - transitionPoint > 1:
            middle = (right - transitionPoint)//2 + transitionPoint
            if promOrdOpenBorder[middle] < c: transitionPoint = middle
            else: right = middle
        uniformTerm = np.exp(-(logc - mu)**2/twoSigmaSqr)/c
        normalisationFactor = np.sqrt(np.pi/2)*sigma*(1 + erfTerm) + (1 - c)*uniformTerm
        return n*(np.log(normalisationFactor) + mu**2/twoSigmaSqr) + (1 - 2*mu/twoSigmaSqr)*(lnx_cumsum[transitionPoint] + (n - transitionPoint - 1)*logc) + (lnx_sqrd_cumsum[transitionPoint] + (n - transitionPoint - 1)*logc**2)/twoSigmaSqr

    def extract_clusters(self, rootID = '1'):
        """Classifies groups that have significance of at least `S` as clusters
        and forms the hierarchy according to the parameter `h_style`.

        First classifies any groups that are statistical outliers. Then if
        `h_style` is set to 1, finds the corresponding groups that are also
        statistical outliers. For each of these that are the smallest out of
        those that share the same starting position within the ordered list,
        classify them too as clusters. Following this, if rootID is not set as
        `None` then also classify the input data the root cluster.
        Finally, generate the array of id strings for the clusters.

        This method requires the `groups` attribute to have already been 
        created, via the `aggregate()` method or otherwise. It also requires the 
        `group_sigs` and `pFit` attributes to have already been created, via the 
        `compute_significance` method or otherwise.

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
        if self.S == 'auto':
            if self._noiseModel == 'Beta': self.S = norm.isf(beta.sf(self.pFit[0], self.pFit[1], self.pFit[2]))
            elif self._noiseModel == 'Half-normal': self.S = norm.isf(halfnorm.sf(self.pFit[0], 0, self.pFit[1]))
            elif self._noiseModel == 'Log-normal': self.S = norm.isf(lognorm.sf(self.pFit[0], self.pFit[2], scale = np.exp(self.pFit[1])))
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
            reorder = np.array(sorted(np.arange(self.clusters.shape[0]), key = lambda i: [self.clusters[i, 0], self.n_samples - self.clusters[i, 1]]), dtype = np.uint32)
            self.clusters = self.clusters[reorder]
            self.significances = self.significances[reorder]

        # Add on root-level cluster
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
            else: # First root cluster
                self.ids.append(rootID)
        self.ids = np.array(self.ids)
        self._rejTime = time.perf_counter() - start