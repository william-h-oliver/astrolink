# The version file is generated automatically by setuptools_scm
from astrolink._version import version as __version__
from .astrolink import AstroLink

# Force JIT compilation on import
import numpy as np
c = AstroLink(np.random.uniform(0, 1, (500, 3)))
c.run()