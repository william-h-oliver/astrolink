import numpy as np
from astrolink import AstroLink
from astrolink import plotting



def test_plotting():
    # Run with float64 data in 2-dimensions
    gauss2D_1 = np.random.normal(0, 1, (10**4, 2)) # Two gaussian blobs
    gauss2D_2 = np.random.normal([10, 0], 1, (10**4, 2))
    P = np.concatenate((gauss2D_1, gauss2D_2), axis = 0)
    
    try: clusterer = AstroLink(P, verbose = 2)
    except: assert False, "AstroLink class could not be instantiated"

    try: clusterer.run()
    except: assert False, "AstroLink.run() could not be run with input-data type np.float64"

    # Plotting
    try: line, polyCollections = plotting.plotOrderedDensity(clusterer)
    except: assert False, "AstroLink.plotOrderedDensity() could not be run"

    try: plotting.plotAggregationTree(clusterer)
    except: assert False, "AstroLink.plotAggregationTree() could not be run"

    try: h, bins, patches, line, lineCollection = plotting.plotProminenceModel(clusterer)
    except: assert False, "AstroLink.plotProminenceModel() could not be run"

    try: h, bins, patches, line, lineCollection = plotting.plotSignificanceModel(clusterer)
    except: assert False, "AstroLink.plotSignificanceModel() could not be run"