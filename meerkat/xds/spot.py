"""Reading XDS SPOT.XDS files."""

from __future__ import annotations

import numpy as np

__all__ = ["read_spot_xds"]


def read_spot_xds(path):
    """Load spots from SPOT.XDS.

    Returns an (N, 4) array of x, y, frame, intensity. XDS writes 7 columns when the
    spots are indexed (x y z intensity h k l) and 4 when they are not; only the first
    four are read, which is what the refinement needs and what improve_orientation
    has always done.

    Pixel coordinates are 1-based and the frame number is offset by half an
    oscillation relative to the det2lab_xds convention -- callers wanting the
    det2lab_xds convention must add [-1, -1, +0.5]. That correction is deliberately
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
