"""Deterministic synthetic experiment: a small detector, a small grid, real files.

reconstruct_data reads frames through fabio.open(), so a fake in-memory frame source
is not enough -- we write real EDF files. Everything here is seeded, so the golden
regression test is reproducible on any machine and any OS.

Kept deliberately small (100x120 detector, 20 frames, 31^3 grid) so the whole suite
runs in seconds and the golden file stays ~100 KB rather than gigabytes.
"""

from __future__ import annotations

import numpy as np

# A TRICLINIC cell. This is deliberate and load-bearing.
#
# The obvious choice -- a cubic cell -- makes unit_cell_vectors a multiple of the
# identity, which is symmetric. A symmetric matrix is its own transpose, so
# `dot(uc, q)` and `dot(uc.T, q)` are identical and the golden test cannot detect a
# transposed cell. Verified by mutation testing: with a cubic cell, transposing
# unit_cell_vectors in reconstruct_data left the entire suite green.
#
# No two cell lengths or angles are equal either, so a permuted or partially
# transposed cell also shows up.
SYNTHETIC_CELL = (10.0, 11.0, 12.0, 85.0, 95.0, 100.0)  # a, b, c, alpha, beta, gamma
NX, NY = 100, 120
FIRST_FRAME, LAST_FRAME = 1, 20

# 5 deg/frame over 20 frames = a 100 deg sweep. Chosen empirically together with
# MAXIND: this small detector only reaches |q| ~ 0.15 A^-1, i.e. hkl ~ +-1.5, so a
# larger grid would be almost entirely NaN and the golden would pin nothing. These
# values fill ~33% of the grid, which is enough to be a real regression test.
OSCILLATION_ANGLE = 5.0
MAXIND = 1.5
NUMBER_OF_PIXELS = [31, 31, 31]


def cell_to_vectors(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Real-space cell vectors a, b, c as ROWS, in the standard XDS-like setting.

    Lower-triangular by construction, hence emphatically not symmetric -- see the
    note on SYNTHETIC_CELL above for why that matters.
    """
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


def write_synthetic_xparm(path) -> None:
    """Write a small but structurally real XPARM.XDS.

    Number formatting matters: read_XPARM scrapes numbers with the regex
    r"-?\\d+\\.?\\d*", which has no exponent support -- it would parse "1.0e-3" as
    "1.0" followed by "3". So everything must be plain decimal notation.
    """
    uc = cell_to_vectors(*SYNTHETIC_CELL)
    lines = [
        " XPARM.XDS    VERSION Jun 17, 2015",
        # starting_frame, starting_angle, oscillation_angle, rotation_axis(3)
        "%6d %13.4f %9.4f %9.6f %9.6f %9.6f" % (1, 0.0, OSCILLATION_ANGLE, 1.0, 0.0, 0.0),
        # wavelength, wavevector(3)  -- wavevector is the incident beam, |k| = 1/lambda
        "%15.6f %14.6f %14.6f %14.6f" % (0.7, 0.0, 0.0, 1.0 / 0.7),
        # space_group_nr, cell(6)
        "%6d %11.4f %11.4f %11.4f %7.3f %7.3f %7.3f" % ((1,) + SYNTHETIC_CELL),
    ]
    for row in uc:  # a, b, c as ROWS
        lines.append("%15.6f %14.6f %14.6f" % tuple(row))
    lines += [
        # n_segments, NX, NY, pixelsize_x, pixelsize_y
        "%10d %9d %9d %11.6f %11.6f" % (1, NX, NY, 0.172, 0.172),
        # x_center, y_center, distance_to_detector
        "%15.6f %14.6f %14.6f" % (NX / 2.0, NY / 2.0, 100.0),
        "%15.6f %14.6f %14.6f" % (1.0, 0.0, 0.0),  # detector_x
        "%15.6f %14.6f %14.6f" % (0.0, 1.0, 0.0),  # detector_y
        "%15.6f %14.6f %14.6f" % (0.0, 0.0, 1.0),  # detector_normal
        "%10d %9d %9d %9d %9d" % (1, 1, NX, 1, NY),  # segment crossection
        "%8.2f %7.2f %7.2f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f"
        % (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),  # segment geometry
    ]
    path.write_text("\n".join(lines) + "\n")


def frame_data(frame_number: int) -> np.ndarray:
    """Deterministic pseudo-random counts for one frame.

    Seeded per frame rather than per run, so a frame's content does not depend on
    which frames were generated before it -- that keeps partial/parallel runs
    comparable.
    """
    rng = np.random.default_rng(seed=frame_number)
    data = rng.integers(0, 1000, size=(NY, NX)).astype(np.int32)
    # A few permanently negative pixels, so `measured_pixels = image >= 0` has
    # something to mask. Real detectors mark untrusted pixels this way.
    data[0, 0] = -1
    data[5, 7] = -1
    data[-1, -1] = -1
    return data


def write_synthetic_frames(directory, template="frame_%05i.edf") -> str:
    """Write EDF frames and return the printf-style template reconstruct_data wants."""
    import fabio

    for n in range(FIRST_FRAME, LAST_FRAME + 1):
        img = fabio.edfimage.EdfImage(data=frame_data(n))
        img.write(str(directory / (template % n)))
    return str(directory / template)


def build_experiment(directory):
    """Materialize the whole synthetic experiment. Returns (xparm_path, template)."""
    xparm = directory / "XPARM.XDS"
    write_synthetic_xparm(xparm)
    template = write_synthetic_frames(directory)
    return xparm, template


def run_reference_reconstruction(directory, **overrides):
    """Run the synthetic reconstruction with the canonical parameters.

    Both the golden generator and the golden test call this, so they cannot drift
    apart -- a golden that was generated with different parameters than it is
    checked against is worse than no golden at all.
    """
    import contextlib
    import io
    import warnings

    from meerkat import reconstruct_data

    xparm, template = build_experiment(directory)
    kwargs = dict(
        filename_template=template,
        first_image=FIRST_FRAME,
        last_image=LAST_FRAME,
        maxind=[MAXIND] * 3,
        number_of_pixels=NUMBER_OF_PIXELS,
        path_to_XPARM=str(xparm),
        output_filename=None,
        all_in_memory=True,
    )
    kwargs.update(overrides)

    # reconstruct_data prints a line per frame and divides 0/0 for unmeasured
    # voxels. Both are current, intended behaviour; silence them here rather than
    # letting them pollute test output.
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return reconstruct_data(**kwargs)
