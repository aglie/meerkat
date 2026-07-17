"""Reading instrument geometry from an XDS.INP file.

Consolidated from two verbatim copies: evaldpy/evald_qt5_v2d.py:32 and
evaldpy/inp2xparm.py:10. The Ewald GUI's own header already asked for this
("Code improvement: shares code with xdsinp2xparm, possibly make a separate shared
library file").

XDS.INP is NOT a substitute for XPARM.XDS. It carries no orientation matrix, so the
returned dict has no unit_cell_vectors, no cell, and no space_group_nr -- which is
exactly why the Ewald viewer built on it can only show lab-frame q-vectors and never
indexed hkl.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["params_from_xds_inp"]

# Matches KEYWORD=value. The character class must include '-' (as a literal, at the
# end) because real XDS keywords contain it: DIRECTION_OF_DETECTOR_X-AXIS,
# X-RAY_WAVELENGTH.
#
# The original was r'([A-Z_-a-z]+)=...', where `_-a` is parsed as a character RANGE
# (_ to a, i.e. _ ` a). It happened to work -- it matched '-' and every real keyword
# -- but only by luck, and it also matched a backtick.
_PARAM_RE = re.compile(r"([A-Za-z_-]+)=([^=\n]*\w)\b(?!=)")


def params_from_xds_inp(filename):
    """Parse XDS.INP into meerkat's instrument-parameter naming.

    Note the two synthesized values: starting_frame and starting_angle are set to 0
    because XDS.INP does not record them. XPARM.XDS does, so the two sources disagree
    on the absolute phi origin -- do not mix them for the same experiment.
    """
    with open(filename) as f:
        text = f.read()

    text = re.sub(r"!.*", "", text)  # XDS comments run to end of line
    xds_params = dict(_PARAM_RE.findall(text))

    def get(key):
        try:
            raw = xds_params[key]
        except KeyError:
            raise KeyError(f"{filename} has no {key!r} entry") from None
        # np.fromstring is deprecated for binary input; the text path with sep=' '
        # still works, but this is the form that is not on a deprecation path.
        return np.array(raw.split(), dtype=float)

    detector_x = get("DIRECTION_OF_DETECTOR_X-AXIS")
    detector_y = get("DIRECTION_OF_DETECTOR_Y-AXIS")

    wavelength = get("X-RAY_WAVELENGTH")

    return {
        # XDS.INP does not record these; XPARM.XDS does. See docstring.
        "starting_frame": np.array([0.0]),
        "starting_angle": np.array([0.0]),
        "oscillation_angle": get("OSCILLATION_RANGE"),
        "rotation_axis": get("ROTATION_AXIS"),
        "wavelength": wavelength,
        "wavevector": get("INCIDENT_BEAM_DIRECTION") / wavelength,
        "NX": get("NX"),
        "NY": get("NY"),
        "pixelsize_x": get("QX"),
        "pixelsize_y": get("QY"),
        "distance_to_detector": get("DETECTOR_DISTANCE"),
        "x_center": get("ORGX"),
        "y_center": get("ORGY"),
        "detector_x": detector_x,
        "detector_y": detector_y,
        # Derived, because XDS.INP does not state it. inp2xparm.py hardcoded [0,0,1]
        # instead, which is only right for an untilted detector -- this is the more
        # general form and matches what the Ewald viewer already did.
        "detector_normal": np.cross(detector_x, detector_y),
    }
