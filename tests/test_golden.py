"""Golden regression: the reconstruction must not change unless we mean it to.

Generated from the pre-modernization code (Stage 1), so it pins current behaviour
including its warts. Stages 2-4 must leave it bit-for-bit identical; Stage 5
(float32 -> float64) changes it deliberately and regenerates it with the diff
enumerated in the commit message.

This is a characterization test, not a correctness test -- it says "still the same",
not "still right". test_bragg_end_to_end.py is the one that says "right".
"""

from pathlib import Path

import numpy as np
import pytest
from synthetic import (
    NUMBER_OF_PIXELS,
    SYNTHETIC_CELL,
    cell_to_vectors,
    run_reference_reconstruction,
)

GOLDEN = Path(__file__).parent / "data" / "golden_31.npz"


@pytest.fixture(scope="module")
def reconstruction(tmp_path_factory):
    return run_reference_reconstruction(tmp_path_factory.mktemp("golden"))


def test_synthetic_cell_is_not_symmetric():
    """The fixture's own premise -- guard it, because it failed once already.

    An earlier version of this fixture used a cubic cell, making unit_cell_vectors a
    multiple of the identity. Symmetric matrices equal their own transpose, so
    transposing the cell inside reconstruct_data was a no-op and the entire suite
    stayed green through a mutation that would have corrupted every real
    reconstruction. A triclinic cell is what gives the golden test its teeth.
    """
    uc = cell_to_vectors(*SYNTHETIC_CELL)
    assert not np.allclose(uc, uc.T), (
        "cell became symmetric -- the golden can no longer detect a transpose"
    )
    a, b, c, alpha, beta, gamma = SYNTHETIC_CELL
    assert len({a, b, c}) == 3, "cell lengths must differ, else axis permutations hide"
    assert len({alpha, beta, gamma}) == 3, "cell angles must differ"


def test_data_matches_golden(reconstruction):
    """Bit-for-bit. Not allclose -- a refactor has no business moving any bit.

    NaN-to-NaN comparison matters here: ~66% of this grid is unmeasured, and those
    voxels are NaN by design (see test_unmeasured_voxels_are_nan).
    """
    expected = np.load(GOLDEN)["data"]
    actual = reconstruction["data"]
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_grid_metadata_matches_golden(reconstruction):
    g = np.load(GOLDEN)
    np.testing.assert_array_equal(np.asarray(reconstruction["step_sizes"]), g["step_sizes"])
    np.testing.assert_array_equal(np.asarray(reconstruction["lower_limits"]), g["lower_limits"])
    np.testing.assert_allclose(np.asarray(reconstruction["metric_tensor"]), g["metric_tensor"])


def test_lower_limits_is_currently_float32(reconstruction):
    """Pins the float32 regression so Stage 5 has to be a deliberate act.

    meerkat.py casts maxind to float32, and it leaks all the way into the output
    metadata: lower_limits = -maxind. np.float_ (float64) was the original dtype
    until commit 7c19787 swapped it for float32 while fixing numpy-2 aliases.
    Stage 5 restores float64 and flips this assertion.
    """
    assert np.asarray(reconstruction["lower_limits"]).dtype == np.float32


def test_unmeasured_voxels_are_nan(reconstruction):
    """0/0 -> NaN is the "no data here" signal, and Yell reads it. Do not "fix" it.

    reconstruct_data divides the summed intensity by the pixel count with no guard,
    so voxels the Ewald sphere never swept become NaN. That is load-bearing output,
    not an accident to be cleaned up behind a np.errstate.
    """
    data = reconstruction["data"]
    assert np.isnan(data).any(), "expected unmeasured voxels in this geometry"
    assert np.isfinite(data).any(), "expected measured voxels too"
    # Guard the fixture itself: if coverage drifts far from ~33%, the golden has
    # stopped testing what it was built to test.
    fraction = np.isfinite(data).mean()
    assert 0.2 < fraction < 0.5, f"coverage {fraction:.1%} -- retune the synthetic geometry"


def test_shape_follows_number_of_pixels(reconstruction):
    assert reconstruction["data"].shape == tuple(NUMBER_OF_PIXELS)


def test_yell_format_metadata(reconstruction):
    """The keys Yell reads. Renaming or dropping one silently breaks Yell."""
    assert reconstruction["format"] == "Yell 1.0"
    assert reconstruction["is_direct"] is False or reconstruction["is_direct"] == 0
    for key in ("space_group_nr", "unit_cell", "metric_tensor", "step_sizes", "lower_limits"):
        assert key in reconstruction, f"Yell requires {key!r}"
