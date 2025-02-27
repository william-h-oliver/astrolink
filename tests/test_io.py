import os
import numpy as np
from astrolink import AstroLink
from astrolink import io


def test_io():
    P = np.random.normal(0, 1, (10**4, 2))
    clusterer = AstroLink(P)
    clusterer.run()

    try: io.saveAstroLinkObject(clusterer, 'test_io.npz')
    except: assert False, "AstroLink class instance could not be saved"

    try: clusterer = io.loadAstroLinkObject('test_io.npz')
    except: assert False, "AstroLink class instance could not be loaded"

    os.remove('test_io.npz')