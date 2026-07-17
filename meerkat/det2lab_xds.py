"""Deprecated location. Use meerkat.xds.geometry.

The geometry moved to meerkat/xds/ so that it -- and XPARM parsing -- stay importable
without fabio and h5py, which is what lets the meerkat-ewald GUI depend on meerkat
without inheriting its heavy dependencies.

`from meerkat.det2lab_xds import det2lab_xds` still works: meerkat 0.3.8 is on PyPI
and this module path is part of what shipped. Removed in 0.5.
"""

import warnings

from .xds.geometry import det2lab_xds, rotvec2mat

__all__ = ["det2lab_xds", "rotvec2mat"]

warnings.warn(
    "meerkat.det2lab_xds has moved to meerkat.xds.geometry and this alias will be "
    "removed in 0.5. Import from meerkat or meerkat.xds instead.",
    DeprecationWarning,
    stacklevel=2,
)
