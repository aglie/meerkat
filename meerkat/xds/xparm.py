"""Reading and writing XDS XPARM.XDS / GXPARM.XDS files.

THE canonical write_xparm lives here. Before this module it existed in four places
-- improve_orientation_v0.82.py:277, xparm_transform_0.21.py:12, evaldpy/inp2xparm.py:49
(commented "copy-pasted from improve_orientation_v0.5py") -- and they had drifted:

  * xparm_transform's copy hardcoded a Pilatus 6M (1475x1679 @ 0.172 mm) instead of
    using the params it was handed, which is why that script printed "THIS SCRIPT IS
    BROKEN / PIXEL SIZE IS WRONG" on every run.
  * improve_orientation's copy passed params["x_center"] to %f WITHOUT the [0] its
    sibling fields use. It survived only because refining 'xycenter' happened to
    replace the array with a scalar first; running --refine cell left it an array,
    and float(1-element-array) has been deprecated since numpy 1.25.

Both classes of bug are fixed structurally rather than by care: the detector is
always taken from params, and _scalar() coerces every scalar field, so the function
accepts read_xparm's 1-element arrays and plain floats alike.

Convention, verified against real XPARM files and asserted in tests:
unit_cell_vectors holds the REAL-space cell vectors a, b, c as ROWS. So
hkl = dot(unit_cell_vectors, q), with no transpose.

numpy-only by design -- this module is the dependency surface for the separate
meerkat-ewald GUI package, which must not drag in fabio or h5py to parse a text file.
"""

from __future__ import annotations

import os
import re

import numpy as np

__all__ = [
    "INSTRUMENT_KEYS",
    "CRYSTAL_KEYS",
    "read_xparm",
    "write_xparm",
    "vecs2cell",
    "cell2vecs",
]

# Which parameters describe the *instrument* rather than this crystal or this scan.
#
# Two consumers need exactly this split, which is why it lives here rather than in
# either of them:
#   * improve-orientation --instrument-xparm, importing geometry refined on a
#     standard sample;
#   * the Ewald GUI, which builds a spin box per editable scalar/3-vector and would
#     choke on a (3,3) unit_cell_vectors.
INSTRUMENT_KEYS = (
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
    "rotation_axis",
)

# Crystal- and scan-specific: importing these from another experiment is always wrong.
CRYSTAL_KEYS = (
    "unit_cell_vectors",
    "cell",
    "space_group_nr",
    "starting_frame",
    "starting_angle",
    "oscillation_angle",
)

# Order matters: read_xparm consumes the file positionally, so this table is both the
# parse order and the field widths.
_FIELDS = (
    ("starting_frame", 1),
    ("starting_angle", 1),
    ("oscillation_angle", 1),
    ("rotation_axis", 3),
    ("wavelength", 1),
    ("wavevector", 3),
    ("space_group_nr", 1),
    ("cell", 6),
    ("unit_cell_vectors", 9),
    ("number_of_detector_segments", 1),
    ("NX", 1),
    ("NY", 1),
    ("pixelsize_x", 1),
    ("pixelsize_y", 1),
    ("x_center", 1),
    ("y_center", 1),
    ("distance_to_detector", 1),
    ("detector_x", 3),
    ("detector_y", 3),
    ("detector_normal", 3),
    ("detector_segment_crossection", 5),
    ("detector_segment_geometry", 9),
)


def _scalar(x) -> float:
    """Coerce a 1-element array or a plain number to a float.

    read_xparm returns 1-element numpy arrays for scalar fields (a shape that
    det2lab_xds's broadcasting depends on), but refinement replaces some of them with
    plain floats. Rather than requiring every caller to remember which is which --
    the exact mistake that produced the x_center bug -- coerce at the boundary.
    """
    return float(np.ravel(np.asarray(x))[0])


def _get_numbers(matches, count):
    return np.array([float(next(matches).group()) for _ in range(count)])


def read_xparm(path_to_xparm="."):
    """Load instrument geometry from XPARM.XDS or GXPARM.XDS.

    Given a directory, prefers GXPARM.XDS (refined) over XPARM.XDS (initial).

    Returns a dict whose scalar entries are 1-element numpy arrays. That is load
    bearing, not an accident: det2lab_xds relies on it for broadcasting, and it is
    the shape the published 0.3.x API returns. Do not "clean it up".
    """
    path_to_xparm = os.fspath(path_to_xparm)

    if not os.path.exists(path_to_xparm):
        raise FileNotFoundError(f"path {path_to_xparm} does not exist")

    if os.path.isdir(path_to_xparm):
        for candidate in ("GXPARM.XDS", "XPARM.XDS"):
            full = os.path.join(path_to_xparm, candidate)
            if os.path.isfile(full):
                path_to_xparm = full
                break
        else:
            raise FileNotFoundError(
                f"neither GXPARM.XDS nor XPARM.XDS found in {path_to_xparm}"
            )

    with open(path_to_xparm) as f:
        f.readline()  # header line -- contains a date, so it must not be scraped
        text = f.read()

    # No exponent support on purpose: this matches what XDS actually writes, and
    # broadening it would silently mis-parse "1.0e-3" as 1.0 followed by 3.
    matches = re.compile(r"-?\d+\.?\d*").finditer(text)

    try:
        result = {name: _get_numbers(matches, width) for name, width in _FIELDS}
    except StopIteration:
        raise ValueError(f"{path_to_xparm} is truncated or not an XPARM file") from None

    result["unit_cell_vectors"] = np.reshape(result["unit_cell_vectors"], (3, 3))

    try:
        next(matches)
    except StopIteration:
        pass
    else:
        raise ValueError(f"{path_to_xparm} has trailing data; not an XPARM file")

    return result


def _vec2str(v) -> str:
    return " ".join(str(el) for el in np.ravel(np.asarray(v)))


def write_xparm(output_filename, params) -> None:
    """Write params to an XPARM.XDS-format file.

    Every scalar goes through _scalar(), and the detector geometry comes from params
    -- see the module docstring for the two bugs that makes structurally impossible.
    """
    uc = np.asarray(params["unit_cell_vectors"])
    if uc.shape != (3, 3):
        raise ValueError(f"unit_cell_vectors must be 3x3, got {uc.shape}")

    ra = np.ravel(np.asarray(params["rotation_axis"]))
    wv = np.ravel(np.asarray(params["wavevector"]))
    nx, ny = int(_scalar(params["NX"])), int(_scalar(params["NY"]))

    lines = [
        " XPARM.XDS    VERSION Jan 31, 2020  BUILT=meerkat generated",
        "    %i        %g    %g %f %f %f"
        % (
            int(_scalar(params["starting_frame"])),
            _scalar(params["starting_angle"]),
            _scalar(params["oscillation_angle"]),
            ra[0],
            ra[1],
            ra[2],
        ),
        "%f %f %f %f" % (_scalar(params["wavelength"]), wv[0], wv[1], wv[2]),
        "    %i     %s" % (int(_scalar(params["space_group_nr"])), _vec2str(params["cell"])),
    ]
    lines += [_vec2str(row) for row in uc]
    lines += [
        "         1      %i      %i    %f    %f"
        % (nx, ny, _scalar(params["pixelsize_x"]), _scalar(params["pixelsize_y"])),
        "%f %f %f"
        % (
            _scalar(params["x_center"]),
            _scalar(params["y_center"]),
            _scalar(params["distance_to_detector"]),
        ),
    ]
    lines += [_vec2str(params[k]) for k in ("detector_x", "detector_y", "detector_normal")]
    lines += [
        "         1         1      %i         1      %i" % (nx, ny),
        "    0.00    0.00    0.00  1.00000  0.00000  0.00000  0.00000  1.00000  0.00000",
    ]

    with open(output_filename, "w") as f:
        f.write("\n".join(lines) + "\n")


def vecs2cell(vectors):
    """Cell parameters (a, b, c, alpha, beta, gamma) from cell vectors given as ROWS."""
    vectors = np.asarray(vectors)
    metric = np.dot(vectors, vectors.T)
    a, b, c = np.sqrt(np.diag(metric))
    alpha = np.rad2deg(np.arccos(metric[1, 2] / (b * c)))
    beta = np.rad2deg(np.arccos(metric[0, 2] / (a * c)))
    gamma = np.rad2deg(np.arccos(metric[0, 1] / (a * b)))
    return np.array([a, b, c, alpha, beta, gamma])


def cell2vecs(cell):
    """Cell vectors as ROWS, in the standard setting with a along x and b in the xy plane.

    Inverse of vecs2cell up to an overall rotation: vecs2cell(cell2vecs(p)) == p, but
    cell2vecs(vecs2cell(v)) recovers v only if v was already in the standard setting.
    """
    a, b, c, alpha, beta, gamma = (float(x) for x in np.ravel(np.asarray(cell)))
    al, be, ga = np.deg2rad([alpha, beta, gamma])
    volume_factor = np.sqrt(
        1
        - np.cos(al) ** 2
        - np.cos(be) ** 2
        - np.cos(ga) ** 2
        + 2 * np.cos(al) * np.cos(be) * np.cos(ga)
    )
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * np.cos(ga), b * np.sin(ga), 0.0],
            [
                c * np.cos(be),
                c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga),
                c * volume_factor / np.sin(ga),
            ],
        ]
    )
