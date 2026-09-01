"""The Ewald viewer's data model -- everything that needs neither Qt nor OpenGL.

Kept separate from the widgets on purpose. The geometry is the part that can be
wrong in a way you would not notice by looking at it, and it is the part CI can
test: CI has numpy, no display and no Qt.

Ported from evaldpy/evald_qt5_v2d.py, which carried its own copy of the XDS.INP
parser. That copy is gone; this reads through `meerkat.xds`, which is the same
parser `inp2xparm` used and is covered by tests.
"""

from __future__ import annotations

import os

import numpy as np

from ..xds import (
    XDS_SPOT_OFFSET,
    det2lab_xds,
    params_from_xds_inp,
    read_spot_xds,
    read_xparm,
)

__all__ = [
    "PARAMETER_ORDER",
    "Experiment",
    "filter_spots",
    "load_experiment",
    "locate_experiment",
    "spot_q_vectors",
    "top_n_reflections",
]

# The order the instrument parameters are laid out in the control panel. The
# original sorted the keys alphabetically, which put NX next to distance and
# scattered the three detector vectors -- grouping them by what they describe
# makes the panel usable. Anything not listed is appended, sorted, so a key
# added to the parser still shows up.
PARAMETER_ORDER = (
    "wavelength",
    "wavevector",
    "rotation_axis",
    "oscillation_angle",
    "starting_frame",
    "starting_angle",
    "distance_to_detector",
    "x_center",
    "y_center",
    "detector_x",
    "detector_y",
    "detector_normal",
    "pixelsize_x",
    "pixelsize_y",
    "NX",
    "NY",
)

# Parameters that are not part of the diffraction geometry and only clutter the
# panel. read_xparm returns them; XDS.INP does not.
_HIDDEN_PARAMETERS = frozenset(
    {
        "unit_cell_vectors",
        "cell",
        "space_group_nr",
        "number_of_detector_segments",
        "detector_segment_geometry",
        "detector_segment_crossection",
    }
)

_XPARM_NAMES = ("XPARM.XDS", "GXPARM.XDS")
_INP_NAMES = ("XDS.INP",)


def ordered_parameters(params):
    """Parameter names in display order, hiding the non-geometric ones."""
    shown = [k for k in params if k not in _HIDDEN_PARAMETERS]
    known = [k for k in PARAMETER_ORDER if k in shown]
    return known + sorted(set(shown) - set(known))


def locate_experiment(path):
    """Resolve a user-supplied path to (parameter file, SPOT.XDS).

    `path` may be an XPARM.XDS, a GXPARM.XDS, an XDS.INP, or a directory holding
    one of them. A directory prefers XPARM over XDS.INP: XPARM records the phi
    origin (starting_frame and starting_angle) and the orientation matrix, while
    XDS.INP records neither, so the same data opened through XDS.INP sits half an
    oscillation away from where it really is.
    """
    path = os.fspath(path)

    if os.path.isdir(path):
        folder = path
        for name in _XPARM_NAMES + _INP_NAMES:
            candidate = os.path.join(folder, name)
            if os.path.exists(candidate):
                parameter_file = candidate
                break
        else:
            raise FileNotFoundError(
                f"{folder} contains none of " + ", ".join(_XPARM_NAMES + _INP_NAMES)
            )
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        parameter_file = path
        folder = os.path.dirname(os.path.abspath(path))

    spot_file = os.path.join(folder, "SPOT.XDS")
    if not os.path.exists(spot_file):
        raise FileNotFoundError(f"no SPOT.XDS next to {parameter_file}")

    return parameter_file, spot_file


def read_parameters(parameter_file):
    """Instrument parameters from an XPARM.XDS, a GXPARM.XDS or an XDS.INP.

    Dispatch is by file name, not by content: XDS names these files by
    convention and a file called XPARM.XDS that is not one is a mistake worth
    reporting as such.
    """
    name = os.path.basename(parameter_file).upper()
    if name in _XPARM_NAMES:
        return read_xparm(parameter_file)
    if name in _INP_NAMES:
        return params_from_xds_inp(parameter_file)
    raise ValueError(
        f"do not know how to read {os.path.basename(parameter_file)}; "
        "expected " + ", ".join(_XPARM_NAMES + _INP_NAMES)
    )


class Experiment:
    """An opened experiment: instrument parameters plus the spots XDS found.

    `spots` is (N, 4) -- x, y, frame, intensity -- in XDS's own convention,
    exactly as SPOT.XDS states it. The convention shift to det2lab_xds is applied
    in q_vectors() and nowhere else, so it cannot be applied twice.
    """

    def __init__(self, params, spots, parameter_file=None, spot_file=None):
        self.params = dict(params)
        self.spots = np.asarray(spots, dtype=float)
        self.parameter_file = parameter_file
        self.spot_file = spot_file

    @property
    def has_orientation(self):
        """True when the parameters came from an XPARM, which carries the cell."""
        return "unit_cell_vectors" in self.params

    def q_vectors(self, spots=None):
        """Lab-frame q for each spot, as an (N, 3) array."""
        return spot_q_vectors(self.spots if spots is None else spots, self.params)


def load_experiment(path):
    """Open an experiment from a parameter file or a folder. See locate_experiment."""
    parameter_file, spot_file = locate_experiment(path)
    return Experiment(
        read_parameters(parameter_file),
        read_spot_xds(spot_file),
        parameter_file=parameter_file,
        spot_file=spot_file,
    )


def spot_q_vectors(spots, params):
    """Lab-frame q-vectors for (N, >=3) spots in XDS convention, as (N, 3).

    XDS_SPOT_OFFSET is what makes this differ from the 2021--2025 viewer, which
    passed SPOT.XDS coordinates to det2lab_xds untouched. SPOT.XDS is 1-based in
    x and y and its frame number sits half an oscillation from what det2lab_xds
    expects, so the whole point cloud was displaced by a pixel and rotated by
    half a frame -- small, but this is a tool for judging exactly that kind of
    error, and improve-orientation had the same bug until 0.4.0.
    """
    spots = np.asarray(spots, dtype=float)
    if spots.size == 0:
        return np.zeros((0, 3))

    xyf = spots[:, :3] + XDS_SPOT_OFFSET
    q, _, _ = det2lab_xds(xyf[:, :2], xyf[:, 2], **params)
    return q.T


def filter_spots(spots, imin=None, imax=None, frame_min=None, frame_max=None):
    """Spots whose intensity and frame number are within the given bounds.

    Bounds are inclusive and `None` means unbounded.
    """
    spots = np.asarray(spots, dtype=float)
    if spots.size == 0:
        return spots

    keep = np.ones(len(spots), dtype=bool)
    if imin is not None:
        keep &= spots[:, 3] >= imin
    if imax is not None:
        keep &= spots[:, 3] <= imax
    if frame_min is not None:
        keep &= spots[:, 2] >= frame_min
    if frame_max is not None:
        keep &= spots[:, 2] <= frame_max
    return spots[keep]


def top_n_reflections(spots, n):
    """The n most intense spots. Ties are kept, so this can return more than n."""
    spots = np.asarray(spots, dtype=float)
    if len(spots) <= n:
        return spots
    threshold = np.sort(spots[:, 3])[-n]
    return spots[spots[:, 3] >= threshold]
