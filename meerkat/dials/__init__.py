"""Convert a DIALS-processed experiment directly to XDS format.

Needs dxtbx (part of a DIALS install) -- imported lazily inside functions, so
`import meerkat` never requires DIALS, matching the rest of the package.
"""

from __future__ import annotations

from .to_xds import (
    DETECTOR_TABLE,
    convert,
    read_dials_geometry,
    rotation_aligning,
    write_spot_xds,
    write_xds_inp,
)

__all__ = [
    "DETECTOR_TABLE",
    "convert",
    "read_dials_geometry",
    "rotation_aligning",
    "write_spot_xds",
    "write_xds_inp",
]
