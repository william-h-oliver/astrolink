import numpy as np
from astrolink import AstroLink


def test_astrolink():
    # Run with float32 data in 1-dimension
    gauss2D_1 = np.random.normal(0, 1, (10**4, 1)) # Two gaussian blobs
    gauss2D_2 = np.random.normal(10, 1, (10**4, 1))
    P = np.concatenate((gauss2D_1, gauss2D_2), axis = 0).astype(np.float32)
    
    try: clusterer = AstroLink(P, S = 5, k_link = 20)
    except: assert False, "AstroLink class could not be instantiated"

    try: clusterer.run()
    except: assert False, "AstroLink.run() could not be run with input-data type np.float32"

    # Run with float64 data in 2-dimensions
    gauss2D_1 = np.random.normal(0, 1, (10**4, 2)) # Two gaussian blobs
    gauss2D_2 = np.random.normal([10, 0], 1, (10**4, 2))
    P = np.concatenate((gauss2D_1, gauss2D_2), axis = 0)
    
    try: clusterer = AstroLink(P, verbose = 2)
    except: assert False, "AstroLink class could not be instantiated"

    try: clusterer.run()
    except: assert False, "AstroLink.run() could not be run with input-data type np.float64"





    # Run jitted methods in python mode for increased test coverage, these are tested in the runs above and in their salient outputs below
    # _rescale()
    arr = clusterer._rescale_njit.py_func(clusterer.P)

    # _compute_logRho()
    sqr_distances = np.sort(np.random.uniform(0, 1, (clusterer.n_samples, clusterer.k_den)), axis = 1)
    arr = clusterer._compute_logRho_njit.py_func(sqr_distances, clusterer.k_den, clusterer.d_features)
    arr = clusterer._compute_weighted_logRho_njit.py_func(sqr_distances, np.ones_like(sqr_distances), clusterer.d_features)

    # _normalise()
    x = np.random.uniform(0, 1, 100)
    arr = clusterer._normalise_njit.py_func(x)

    # _make_graph_njit()
    c = AstroLink(P.astype(np.float32))
    c.transform_data()
    c.estimate_density_and_kNN()
    pairs, edges = c._make_graph_njit.py_func(c.logRho, c.kNN)
    ordered_pairs = pairs[edges.argsort()[::-1]]

    # _aggregate_njit_floatXX_uintXX()
    arr1, arr2, arr3 = c._aggregate_njit_float32_uint32.py_func(c.logRho.astype(np.float32), ordered_pairs.astype(np.uint32))
    arr1, arr2, arr3 = c._aggregate_njit_float32_uint64.py_func(c.logRho.astype(np.float32), ordered_pairs.astype(np.uint64))
    arr1, arr2, arr3 = c._aggregate_njit_float64_uint32.py_func(c.logRho.astype(np.float64), ordered_pairs.astype(np.uint32))
    arr1, arr2, arr3 = c._aggregate_njit_float64_uint64.py_func(c.logRho.astype(np.float64), ordered_pairs.astype(np.uint64))
    del c, arr1, arr2, arr3

    # _minimize_init()
    modelParams, modelArgs, _, _, _ = clusterer._minimize_init_njit.py_func(clusterer.prominences[:, 0])

    # _negLL_**_njit() functions
    modelNegLLs = [clusterer._negLL_beta_njit.py_func, clusterer._negLL_truncatednormal_njit.py_func, clusterer._negLL_lognormal_njit.py_func]
    for negLL, params, args in zip(modelNegLLs, modelParams, modelArgs):
        num = negLL(params, *args, 0.5)
    del num, modelNegLLs, modelParams, modelArgs








    # Test properties
    # S
    val = clusterer.S
    assert isinstance(val, float), "S has not been calculated properly from S = 'auto'"

    # k_link
    val = clusterer.k_link
    assert isinstance(val, int), "k_link has not been calculated properly from k_link = 'auto'"

    # logRho
    arr = clusterer.logRho
    assert isinstance(arr, np.ndarray), "logRho must be a numpy array"
    assert arr.shape == (clusterer.P.shape[0],), "logRho does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "logRho does not have the correct dtype"
    assert np.logical_and(arr >= 0.0, arr <= 1.0).all(), "logRho is not bounded between 0 and 1"

    # ordering
    arr = clusterer.ordering
    assert isinstance(arr, np.ndarray), "ordering must be a numpy array"
    assert arr.shape == (clusterer.P.shape[0],), "ordering does not have the correct shape"
    assert arr.dtype == np.dtype('uint32'), "ordering does not have the correct dtype"
    assert np.unique(arr).size == clusterer.P.shape[0], "ordering does not contain all the indices of P exactly once"

    # clusters
    arr = clusterer.clusters
    assert isinstance(arr, np.ndarray), "clusters must be a numpy array"
    assert len(arr.shape) == 2 and arr.shape[1] == 2, "clusters does not have the correct shape"
    assert arr.dtype == np.dtype('uint32'), "clusters does not have the correct dtype"
    assert (arr[:, 1] > arr[:, 0]).all() and arr.min() >= 0 and arr.max() <= clusterer.P.shape[0], "clusters contains invalid indices"

    # ids
    arr = clusterer.ids
    assert isinstance(arr, np.ndarray), "ids must be a numpy array"
    assert arr.shape == (clusterer.clusters.shape[0],), "ids does not have the correct shape"
    assert arr.dtype.type == np.str_, "ids does not have the correct dtype"
    assert np.char.isdigit(np.char.replace(arr, '-', '')).all(), "ids does not contain only digits and dashes"
    def check_valid_hierarchy(ids):
        for i, id in enumerate(ids):
            j = i - 1
            status = False
            while j >= 0:
                firstChildID = ids[j] + '-1'
                id_split = ids[j].split('-')
                parentID = '-'.join(id_split[:-1])
                if parentID != '': nextSiblingID = parentID + '-' + str(int(id_split[-1]) + 1)
                else: nextSiblingID = str(int(id_split[-1]) + 1)
                if id in [firstChildID, nextSiblingID]:
                    status = True
                    break
                else: j -= 1
            if i > 0 and not status: return False
        return True
    assert check_valid_hierarchy(arr), "ids does not specify a valid hierarchy"

    # significances
    arr = clusterer.significances
    assert isinstance(arr, np.ndarray), "significances must be a numpy array"
    assert arr.shape == (clusterer.clusters.shape[0],), "significances does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "significances does not have the correct dtype"
    assert (arr >= clusterer.S).all(), "significances contains invalid values"

    # groups
    arr = clusterer.groups
    assert isinstance(arr, np.ndarray), "groups must be a numpy array"
    assert len(arr.shape) == 2 and arr.shape[1] == 3, "groups does not have the correct shape"
    assert arr.dtype == np.dtype('uint32'), "groups does not have the correct dtype"
    assert (arr[:, 1] > arr[:, 0]).all() and (arr[:, 2] > arr[:, 1]).all() and arr.min() >= 0 and arr.max() <= clusterer.P.shape[0], "groups contains invalid indices"

    # prominences
    arr = clusterer.prominences
    assert isinstance(arr, np.ndarray), "prominences must be a numpy array"
    assert arr.shape == (clusterer.groups.shape[0], 2), "prominences does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "prominences does not have the correct dtype"
    assert np.logical_and(arr >= 0.0, arr <= 1.0).all(), "prominences is not bounded between 0 and 1"

    # groups_sigs
    arr = clusterer.groups_sigs
    assert isinstance(arr, np.ndarray), "groups_sigs must be a numpy array"
    assert arr.shape == (clusterer.groups.shape[0], 2), "groups_sigs does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "groups_sigs does not have the correct dtype"

    # pFit
    arr = clusterer.pFit
    assert isinstance(arr, np.ndarray), "pFit must be a numpy array"
    assert (arr.shape == (2,)) or (arr.shape == (3,)), "pFit does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "pFit does not have the correct dtype"
    assert (arr >= 0.0).all() and arr[0] <= 1.0, "pFit contains invalid values"