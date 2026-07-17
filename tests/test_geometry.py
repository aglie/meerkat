"""Geometry tests, built on XDS's own output as ground truth.

Replaces the in-module test_det2lab_xds() that used to live in meerkat/det2lab_xds.py.
That test hand-typed unit_cell_vectors with a, b, c in *columns* and compensated with
a .T at the call site. The two were a matched pair, so it passed -- but it could not
have detected a real transpose bug, and it taught the wrong convention to anyone who
read it. Parsing the XPARM makes the transpose unrepresentable.
"""

import numpy as np

from meerkat import det2lab_xds, rotvec2mat


def test_xparm_rows_are_abc(xparm):
    """XPARM stores the unit cell vectors as ROWS a, b, c.

    This is the single most expensive fact in this codebase to relearn, so assert it
    directly: the row norms must reproduce the cell lengths a, b, c.
    """
    row_norms = np.linalg.norm(xparm["unit_cell_vectors"], axis=1)
    np.testing.assert_allclose(row_norms, xparm["cell"][:3], atol=1e-3)


def test_bragg_peaks_index_to_integers(xparm, bragg_peaks):
    """Predicted hkl for XDS-indexed spots must land near integers.

    Note there is no .T on unit_cell_vectors -- rows are a, b, c, so dot(uc, lab) is
    the correct contraction. The 0.15 tolerance reflects genuine spot-position scatter
    in the measured data (observed max ~0.126), not slack for a convention error.
    """
    hkl, xy, frame = bragg_peaks
    lab = det2lab_xds(xy, frame, **xparm)[0]
    fractional = np.dot(xparm["unit_cell_vectors"], lab)
    deviation = np.abs(fractional.T - hkl)
    assert np.all(deviation < 0.15), f"max deviation {deviation.max():.4f}"


def test_transposed_cell_vectors_would_fail(xparm, bragg_peaks):
    """Guard on the guard.

    If someone reintroduces the spurious .T, test_bragg_peaks_index_to_integers must
    fail rather than quietly pass. Pin that: the transposed contraction is off by
    ~8 r.l.u., nowhere near the tolerance.
    """
    hkl, xy, frame = bragg_peaks
    lab = det2lab_xds(xy, frame, **xparm)[0]
    wrong = np.abs(np.dot(xparm["unit_cell_vectors"].T, lab).T - hkl)
    assert wrong.max() > 1.0, "a transposed cell must not be mistaken for a correct one"


def test_rotvec2mat_is_orthogonal():
    m = rotvec2mat(np.array([1.0, 2.0, 3.0]), np.deg2rad(37.0))
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(m), 1.0, atol=1e-12)


def test_rotvec2mat_rejects_zero_axis():
    """A zero-length axis has no rotation defined; it must raise, not return garbage.

    The Ewald GUI lets the user spin the rotation axis through (0, 0, 0) via spin
    boxes, so this path is reachable from the UI.
    """
    import pytest

    with pytest.raises(Exception, match="zero"):
        rotvec2mat(np.array([0.0, 0.0, 0.0]), 1.0)
