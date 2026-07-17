"""Pin the index mapping: hkl -> voxel.

The index formula, from meerkat.py:388-391:
    step_size_inv = (number_of_pixels - 1) / maxind / 2
    index         = around(step_size_inv * (c + maxind))
so index 0 sits at c = -maxind and index N-1 at c = +maxind: voxel *centres* land on
integer indices.

Tie-breaking is deliberately untested. np.around is half-to-even and floor(u + 0.5)
is half-up, but they differ only when a coordinate lands exactly on a voxel boundary,
which real-valued data does not do. Either is reasonable; the implementation keeps
np.around because it is already there and already handles negatives correctly.

What IS tested here: the grid endpoints, rejection of out-of-range coordinates (where
Meerkat2's C++ int() truncation gets it wrong), and the float32 cast of maxind that
Stage 5 reverts.
"""

import numpy as np
import pytest


def to_index(c, maxind, number_of_pixels, dtype=np.float32):
    """The mapping exactly as meerkat.py:385-391 computes it."""
    maxind = np.array(maxind, dtype=dtype)
    number_of_pixels = np.array(number_of_pixels)
    step_size_inv = 1.0 * (number_of_pixels - 1) / maxind / 2
    c = np.atleast_2d(np.asarray(c, dtype=float))
    return np.around(step_size_inv[:, np.newaxis] * (c + maxind[:, np.newaxis])).astype(np.int64)


class TestGridEndpoints:
    def test_lower_limit_maps_to_zero(self):
        idx = to_index([[-5.0], [-5.0], [-5.0]], [5, 5, 5], [11, 11, 11])
        np.testing.assert_array_equal(idx.ravel(), [0, 0, 0])

    def test_upper_limit_maps_to_last_voxel(self):
        idx = to_index([[5.0], [5.0], [5.0]], [5, 5, 5], [11, 11, 11])
        np.testing.assert_array_equal(idx.ravel(), [10, 10, 10])

    def test_origin_maps_to_centre(self):
        idx = to_index([[0.0], [0.0], [0.0]], [5, 5, 5], [11, 11, 11])
        np.testing.assert_array_equal(idx.ravel(), [5, 5, 5])


class TestNegativeCoordinates:
    """Coordinates below the lower limit must produce negative indices, so the
    caller's `indices >= 0` filter (meerkat.py:228-229) drops them.

    This is where Meerkat2's C++ to_index differs: it uses int(), which truncates
    toward zero, so u=-0.6 becomes 0 and the out-of-range point is wrongly binned
    onto the h=0 face. np.around floors it to -1 and it gets rejected. Meerkat is
    correct here; pin it.
    """

    def test_below_lower_limit_gives_negative_index(self):
        # maxind=5, N=11 -> step_size_inv=1; c=-5.6 -> u=-0.6 -> around -> -1
        idx = to_index([[-5.6], [-5.6], [-5.6]], [5, 5, 5], [11, 11, 11])
        assert np.all(idx < 0), f"expected negative, got {idx.ravel()}"

    def test_truncation_toward_zero_would_be_wrong(self):
        """Guard on the guard: int() would accept a point that must be rejected."""
        u = -0.6
        assert int(np.around(u)) == -1, "around floors this -> rejected. Correct."
        assert int(u + 0.5) == 0, "C++-style int(u+0.5) accepts it -> wrong. Do not adopt."


class TestFloat32MaxindRegression:
    """Stage 5 territory: maxind is cast to float32 at meerkat.py:385.

    np.float_ (float64) was the original dtype; commit 7c19787 changed it to float32
    while fixing numpy-2 aliases -- an unintentional precision regression.

    The blast radius is SMALLER than it first looks, and these tests record why, so
    nobody re-inflates the claim later. step_size_inv is computed as
        1.0 * (number_of_pixels - 1) / maxind / 2
    where number_of_pixels is an int64 array; numpy promotes int64/float32 to
    float64, so step_size_inv is ALREADY float64. The cast therefore degrades only
    `maxind` itself -- the additive offset and the value written to lower_limits --
    and not the reciprocal step used for the multiply.

    Consequence, measured over 500k random coordinates on an 801^3 grid:
      maxind = 7.0, 5.0, 1.5  (exactly representable) -> 0 voxels differ
      maxind = 7.3, 6.7, 0.1  (not representable)     -> ~0.001-0.002% differ
    So the fix is still right -- it is a one-line revert of an accident, and
    lower_limits is written to the output file where Yell and downstream tools read
    it -- but it is a precision hygiene fix, not a correctness emergency.
    """

    def test_step_size_inv_is_already_float64(self):
        """The cast does not reach the reciprocal step. Documented, not assumed."""
        maxind = np.array([7.0, 7.0, 7.0], dtype=np.float32)
        npix = np.array([801, 801, 801])
        step_size_inv = 1.0 * (npix - 1) / maxind / 2
        assert step_size_inv.dtype == np.float64

    @staticmethod
    def _voxel_boundary_coords(mv, npix=801):
        """Coordinates sitting exactly on voxel boundaries under float64 maxind.

        Adversarial by construction rather than random: the float32 error is ~1e-5
        voxels, so a uniform random sample hits a boundary about once in 1e5 draws
        and the test would pass or fail on the seed. Landing every sample exactly on
        a tie makes the comparison deterministic.
        """
        maxind64 = np.array([mv] * 3, dtype=np.float64)
        n = np.array([npix] * 3)
        step_size_inv = 1.0 * (n - 1) / maxind64 / 2
        k = np.arange(0, npix - 1)
        c = (k + 0.5) / step_size_inv[0] - maxind64[0]
        return np.broadcast_to(c, (3, c.size))

    @pytest.mark.parametrize("mv", [7.0, 5.0, 1.5])
    def test_representable_maxind_is_unaffected(self, mv):
        """For the common case the float32 cast changes precisely nothing.

        maxind is exactly representable, so both the offset and step_size_inv come
        out bit-identical -- even for coordinates sitting exactly on a boundary.
        """
        c = self._voxel_boundary_coords(mv)
        i32 = to_index(c, [mv] * 3, [801] * 3, dtype=np.float32)
        i64 = to_index(c, [mv] * 3, [801] * 3, dtype=np.float64)
        np.testing.assert_array_equal(i32, i64)

    @pytest.mark.parametrize("mv", [7.3, 6.7])
    def test_unrepresentable_maxind_shifts_boundary_voxels(self, mv):
        """maxind that float32 cannot hold shifts coordinates across boundaries.

        On boundary coordinates ~half flip (the float32 offset breaks the tie one
        way or the other). On *random* coordinates the rate is only ~1e-5, since a
        1e-5-voxel error only matters that close to a boundary -- so this is a
        precision hygiene fix, not a correctness emergency.
        """
        c = self._voxel_boundary_coords(mv)
        i32 = to_index(c, [mv] * 3, [801] * 3, dtype=np.float32)
        i64 = to_index(c, [mv] * 3, [801] * 3, dtype=np.float64)
        fraction = (i32 != i64).any(axis=0).mean()
        assert fraction > 0.1, f"expected many boundary flips, got {fraction:.1%}"

    def test_lower_limits_loses_precision(self):
        """The always-present artifact: this value is written to the output file.

        lower_limits = -maxind, and Yell reads it to place the grid in reciprocal
        space. A float32 -7.3 is off by 1.9e-7 r.l.u.
        """
        assert float(np.float32(-7.3)) != -7.3
        assert abs(float(np.float32(-7.3)) + 7.3) == pytest.approx(1.9e-7, rel=0.1)

    def test_current_code_uses_float32(self):
        """Fails the moment Stage 5 lands, which is the point."""
        import inspect

        from meerkat.meerkat import reconstruct_data

        src = inspect.getsource(reconstruct_data)
        assert "np.array(maxind, dtype=np.float32)" in src, (
            "maxind dtype changed -- if this is Stage 5, invert this test and "
            "regenerate the golden"
        )
