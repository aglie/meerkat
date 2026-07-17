"""Correctness of the pixel -> hkl -> voxel chain, against an independent oracle.

The golden test says "unchanged". This one says "right". The difference matters:
every geometry convention in this package (rows-are-abc, the phi sign, the rotation
sense, the index origin) could be consistently wrong and the golden would pass
forever.

Ground truth: 22 Bragg peaks that XDS located on the detector and *indexed*. Their
hkl were produced by XDS, not by meerkat, so this test can fail in a way no golden
file can.

Scope note -- why this stops short of reconstruct_data(): the peaks span frames
27.9 to 1783.7 on a 2463x2527 detector. Synthesizing frames for a real end-to-end run
would mean ~1756 images of ~25 MB, i.e. ~43 GB, to light up 22 pixels. So we drive
the same arithmetic reconstruct_data drives -- det2lab_xds, the dot with
unit_cell_vectors, and the to_index formula from meerkat.py:388-391 -- without the
frames. The fabio decode and accumulate_intensity steps that this leaves out are
covered by test_golden.py's synthetic experiment, which uses a 100x120 detector.
"""

import numpy as np
import pytest

MAXIND = 6.0  # covers the peak list (max |l| = 6)
NPIX = 13  # -> step of exactly 1 r.l.u., so integer hkl sit on voxel centres


def to_index(fractional, maxind, number_of_pixels):
    """The index mapping exactly as meerkat.py:385-391 computes it."""
    maxind = np.array([maxind] * 3, dtype=np.float64)
    n = np.array([number_of_pixels] * 3)
    step_size_inv = 1.0 * (n - 1) / maxind / 2
    return np.around(
        step_size_inv[:, np.newaxis] * (fractional + maxind[:, np.newaxis])
    ).astype(np.int64)


@pytest.fixture(scope="module")
def predicted(xparm, bragg_peaks):
    """Run the real chain: detector pixel -> lab q-vector -> fractional hkl."""
    from meerkat import det2lab_xds

    hkl, xy, frame = bragg_peaks
    lab = det2lab_xds(xy, frame, **xparm)[0]
    fractional = np.dot(xparm["unit_cell_vectors"], lab)  # rows are a, b, c
    return hkl, fractional


def test_predicted_hkl_are_near_integers(predicted):
    """XDS says these spots are at integer hkl. meerkat's geometry must agree."""
    hkl, fractional = predicted
    deviation = np.abs(fractional.T - hkl)
    assert np.all(deviation < 0.15), f"max deviation {deviation.max():.4f} r.l.u."


def test_every_peak_bins_to_its_own_hkl_voxel(predicted):
    """The full chain must place each peak in the voxel its known hkl demands.

    With maxind=6 and N=13 the step is exactly 1 r.l.u. and lower_limits is -6, so
    the voxel for hkl is exactly hkl + 6. No rounding slack: a failure here is a real
    misplacement, not a near-miss.
    """
    hkl, fractional = predicted
    actual = to_index(fractional, MAXIND, NPIX)
    expected = (hkl + MAXIND).T.astype(np.int64)
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg="a Bragg peak binned to the wrong voxel -- geometry convention is wrong",
    )


def test_all_peaks_land_inside_the_grid(predicted):
    """The bounds filter (meerkat.py:228-229) must not be dropping real data."""
    _, fractional = predicted
    idx = to_index(fractional, MAXIND, NPIX)
    assert np.all(idx >= 0) and np.all(idx < NPIX)


def test_transposed_cell_would_misplace_peaks(xparm, bragg_peaks):
    """Guard on the guard: the wrong convention must fail loudly, not subtly.

    If a future refactor reintroduces a .T on unit_cell_vectors, the tests above have
    to break. Confirm they would.
    """
    from meerkat import det2lab_xds

    hkl, xy, frame = bragg_peaks
    lab = det2lab_xds(xy, frame, **xparm)[0]
    wrong = np.dot(xparm["unit_cell_vectors"].T, lab)
    assert np.abs(wrong.T - hkl).max() > 1.0


def test_reversed_rotation_would_misplace_peaks(xparm, bragg_peaks):
    """The sign of phi is a convention that is easy to flip and hard to notice."""
    from meerkat import det2lab_xds

    hkl, xy, frame = bragg_peaks
    flipped = dict(xparm)
    flipped["oscillation_angle"] = -np.asarray(xparm["oscillation_angle"])
    lab = det2lab_xds(xy, frame, **flipped)[0]
    wrong = np.dot(flipped["unit_cell_vectors"], lab)
    assert np.abs(wrong.T - hkl).max() > 0.15, "a flipped rotation sense went undetected"
