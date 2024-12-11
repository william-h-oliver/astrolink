import numpy as np
from scipy.special import beta as beta_fun
from scipy.special import betainc as betainc_fun
from scipy.special import erf
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
    arr = clusterer._rescale.py_func(clusterer.P)

    # _compute_logRho()
    sqr_distances = np.random.uniform(0, 1, (clusterer.n_samples, clusterer.k_den))
    arr = clusterer._compute_logRho.py_func(sqr_distances, clusterer.k_den, clusterer.d_features)
    arr = clusterer._compute_weightedLogRho.py_func(sqr_distances, np.ones_like(sqr_distances), clusterer.d_features)

    # _normalise()
    x = np.random.uniform(0, 1, 100)
    arr = clusterer._normalise.py_func(x)

    # _normalise()
    x = np.random.uniform(0, 1, 100)
    arr = clusterer._normalise.py_func(x)

    # order_pairs_njit_float32() and aggregate_njit_float32()
    c = AstroLink(P.astype(np.float32))
    c.transform_data()
    c.estimate_density_and_kNN()
    ordered_pairs = c._order_pairs_njit_float32(c.logRho, c.kNN)
    arr1, arr2, arr3, arr4, arr5 = c._aggregate_njit_float32.py_func(c.logRho, ordered_pairs)

    # order_pairs_njit_float64() and aggregate_njit_float64()
    c = AstroLink(P)
    c.transform_data()
    c.estimate_density_and_kNN()
    ordered_pairs = c._order_pairs_njit_float64(c.logRho, c.kNN)
    arr1, arr2, arr3, arr4, arr5 = c._aggregate_njit_float64.py_func(c.logRho, ordered_pairs)
    del c, arr1, arr2, arr3, arr4, arr5

    # _minimize_init()
    _proms_ordered, _lnx_cumsum, _ln_1_minus_x_cumsum, p_beta, _x_sqrd_cumsum, p_halfnormal = clusterer._minimize_init.py_func(clusterer.prominences_leq)

    # _negLL_beta_njit()
    num = clusterer._negLL_beta_njit.py_func(p_beta, _proms_ordered, _lnx_cumsum, _ln_1_minus_x_cumsum, beta_fun(p_beta[1], p_beta[2])*betainc_fun(p_beta[1], p_beta[2], p_beta[0]))
    del num, _lnx_cumsum, _ln_1_minus_x_cumsum, p_beta

    # _negLL_halfnormal_njit()
    num = clusterer._negLL_half_normal_njit.py_func(p_halfnormal, _proms_ordered, _x_sqrd_cumsum, erf(p_halfnormal[0]/(np.sqrt(2)*p_halfnormal[1])))
    del num, _x_sqrd_cumsum, p_halfnormal








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

    # groups_leq
    arr = clusterer.groups_leq
    assert isinstance(arr, np.ndarray), "groups_leq must be a numpy array"
    assert len(arr.shape) == 2 and arr.shape[1] == 2, "groups_leq does not have the correct shape"
    assert arr.dtype == np.dtype('uint32'), "groups_leq does not have the correct dtype"
    assert (arr[:, 1] > arr[:, 0]).all() and arr.min() >= 0 and arr.max() <= clusterer.P.shape[0], "groups_leq contains invalid indices"

    # prominences_leq
    arr = clusterer.prominences_leq
    assert isinstance(arr, np.ndarray), "prominences_leq must be a numpy array"
    assert arr.shape == (clusterer.groups_leq.shape[0],), "prominences_leq does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "prominences_leq does not have the correct dtype"
    assert np.logical_and(arr >= 0.0, arr <= 1.0).all(), "prominences_leq is not bounded between 0 and 1"

    # groups_leq_sigs
    arr = clusterer.groups_leq_sigs
    assert isinstance(arr, np.ndarray), "groups_leq_sigs must be a numpy array"
    assert arr.shape == (clusterer.groups_leq.shape[0],), "groups_leq_sigs does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "groups_leq_sigs does not have the correct dtype"

    # groups_geq
    arr = clusterer.groups_geq
    assert isinstance(arr, np.ndarray), "groups_geq must be a numpy array"
    assert arr.shape == clusterer.groups_leq.shape, "groups_geq does not have the correct shape"
    assert arr.dtype == np.dtype('uint32'), "groups_geq does not have the correct dtype"
    assert (arr[:, 1] > arr[:, 0]).all() and arr.min() >= 0 and arr.max() <= clusterer.P.shape[0], "groups_geq contains invalid indices"

    # prominences_geq
    arr = clusterer.prominences_geq
    assert isinstance(arr, np.ndarray), "prominences_geq must be a numpy array"
    assert arr.shape == (clusterer.groups_geq.shape[0],), "prominences_geq does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "prominences_geq does not have the correct dtype"
    assert np.logical_and(arr >= 0.0, arr <= 1.0).all(), "prominences_geq is not bounded between 0 and 1"

    # groups_geq_sigs
    arr = clusterer.groups_geq_sigs
    assert isinstance(arr, np.ndarray), "groups_geq_sigs must be a numpy array"
    assert arr.shape == (clusterer.groups_geq.shape[0],), "groups_geq_sigs does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "groups_geq_sigs does not have the correct dtype"

    # pFit
    arr = clusterer.pFit
    assert isinstance(arr, np.ndarray), "pFit must be a numpy array"
    assert arr.shape == (3,), "pFit does not have the correct shape"
    assert arr.dtype in [np.dtype('float32'), np.dtype('float64')], "pFit does not have the correct dtype"
    assert (arr >= 0.0).all() and arr[0] <= 1.0, "pFit contains invalid values"