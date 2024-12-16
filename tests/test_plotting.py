import numpy as np
from astrolink import AstroLink
from astrolink import plot



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
    try: line, polyCollections = plot.orderedDensity(clusterer)
    except: assert False, "plot.orderedDensity() could not be run"

    try: plot.aggregationTree(clusterer)
    except: assert False, "plot.aggregationTree() could not be run"

    try: h, bins, patches, line, lineCollection = plot.prominenceModel(clusterer)
    except: assert False, "plot.prominenceModel() could not be run"

    try: h, bins, patches, line, lineCollection = plot.significanceModel(clusterer)
    except: assert False, "plot.significanceModel() could not be run"

    try: plot.logRhoOnX(clusterer, clusterer.P)
    except: assert False, "plot.logRhoOnX() could not be run"

    try: plot.labelsOnX(clusterer, clusterer.P)
    except: assert False, "plot.labelsOnX() could not be run"