"""
Loading and saving routines to be used with the AstroLink class.

Author: William H. Oliver <william.hardie.oliver@gmail.com>
License: MIT
"""

# Third-party libraries
import numpy as np

def saveAstroLinkObject(clusterer, fileName):
    """
    Save an AstroLink instance.

    Parameters
    ----------
    clusterer : AstroLink
        The AstroLink instance to save.
    fileName : str
        The name of the file to save the instance to.
    """

    # Save the AstroLink instance
    np.savez(fileName, clusterer = clusterer)

def loadAstroLinkObject(fileName):
    """
    Load a saved AstroLink instance.

    Parameters
    ----------
    fileName : str
        The name of the file to load the instance from.

    Returns
    -------
    clusterer : AstroLink
        The loaded AstroLink instance.
    """

    if not fileName.endswith('.npz'):
        fileName += '.npz'

    # Load the AstroLink instance
    clusterer = np.load(fileName, allow_pickle = True)['clusterer'].item()

    return clusterer