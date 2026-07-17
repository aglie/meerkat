"""XDS file formats and diffraction geometry. numpy-only.

This subpackage is the dependency surface for the separate meerkat-ewald GUI: it
must stay importable with numpy alone, so that a Qt viewer parsing a text file does
not pull in fabio and h5py. Enforced by ruff's banned-api rules (see pyproject.toml)
and by tests/test_lazy_import.py, which checks it in a subprocess.

Nothing here may import from meerkat.recon, meerkat.io, meerkat.config, or
meerkat.legacy.
"""

from .geometry import det2lab_xds, rotvec2mat
from .spot import read_spot_xds
from .xds_inp import params_from_xds_inp
from .xparm import (
    CRYSTAL_KEYS,
    INSTRUMENT_KEYS,
    cell2vecs,
    read_xparm,
    vecs2cell,
    write_xparm,
)

__all__ = [
    "CRYSTAL_KEYS",
    "INSTRUMENT_KEYS",
    "cell2vecs",
    "det2lab_xds",
    "params_from_xds_inp",
    "read_spot_xds",
    "read_xparm",
    "rotvec2mat",
    "vecs2cell",
    "write_xparm",
]
