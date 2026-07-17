"""Shared fixtures.

The data under tests/data/ is an *independent oracle*: the Bragg peak positions and
their hkl assignments were produced by XDS, not by meerkat. A test built on them can
fail in a way that a golden-file regression test cannot -- a golden file only pins
what meerkat currently does, right or wrong.
"""

from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def xparm_path():
    """Path to a real-format XPARM.XDS for the PdCPTN-like test dataset."""
    return DATA / "XPARM.XDS"


@pytest.fixture(scope="session")
def spot_path():
    """Path to SPOT.XDS holding the same 22 spots, in XDS's COLSPOT format."""
    return DATA / "SPOT.XDS"


@pytest.fixture(scope="session")
def xparm(xparm_path):
    """Instrument parameters parsed from XPARM.XDS by meerkat's own reader.

    Deliberately parsed rather than hand-typed: a hand-typed fixture can encode a
    transposed unit_cell_vectors and then be 'corrected' by a .T at the call site,
    which is exactly what the old in-module test did -- it passed while being unable
    to detect a real transpose bug.
    """
    from meerkat import read_XPARM

    return read_XPARM(str(xparm_path))


@pytest.fixture(scope="session")
def bragg_peaks():
    """XDS-indexed Bragg peaks: (hkl, pixel_xy, frame_number).

    Columns of the file: h k l I sigma x y frame rlp peak corr psi
    """
    b = np.loadtxt(DATA / "bragg_peaks.txt")
    return b[:, 0:3], b[:, 5:7], b[:, 7]
