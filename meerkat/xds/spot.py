"""Reading XDS SPOT.XDS files."""

from __future__ import annotations

import numpy as np

__all__ = ["XDS_SPOT_OFFSET", "read_spot_xds"]

# XDS reports spot positions with 1-based pixel indices and the frame number offset by
# half an oscillation relative to what det2lab_xds expects. Add this to (x, y, frame)
# to move a SPOT.XDS row into the det2lab_xds convention.
XDS_SPOT_OFFSET = np.array([-1.0, -1.0, 0.5])


def read_spot_xds(path):
    """Load spots from SPOT.XDS.

    Returns an (N, 4) array of x, y, frame, intensity. XDS writes 7 columns when the
    spots are indexed (x y z intensity h k l) and 4 when they are not; only the first
    four are read, which is what the refinement needs and what improve_orientation
    has always done.

    Pixel coordinates are 1-based and the frame number is offset by half an
    oscillation relative to the det2lab_xds convention -- callers wanting the
    det2lab_xds convention must add XDS_SPOT_OFFSET. That correction is deliberately
    NOT applied here: it is a convention shift the caller must opt into, and burying
    it in the reader is how it ends up applied twice or not at all.
    """
    spots = []
    with open(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 4:
                continue
            spots.append([float(x) for x in fields[:4]])

    if not spots:
        raise ValueError(f"{path} contains no spots")

    return np.array(spots)
