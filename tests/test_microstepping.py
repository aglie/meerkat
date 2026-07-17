"""Phi microstepping: subdividing each frame's rotation.

This feature had no test, and on that basis the modernization plan proposed deleting
it as dead code. It is not dead -- it works, and its sub-step angles agree exactly
with Meerkat2's Microstep class. It nearly got deleted because nothing demonstrated
otherwise, so: demonstrate it.

`microsteps` is a triple [x, y, phi] and the three entries are in completely different
states. Only phi > 1 works; see test_x_y_microstepping_is_not_implemented and
test_frame_decimation.
"""

import numpy as np
import pytest
from synthetic import run_reference_reconstruction


def sub_step_offsets(oscillation, microsteps, frame_number=10):
    """The angle offsets, relative to the frame centre, that meerkat.py:474 produces."""
    micro = oscillation / microsteps
    angles = [
        ((frame_number - 0.5) * microsteps + m + 0.5) * micro for m in range(microsteps)
    ]
    return np.array(angles) - frame_number * oscillation


class TestSubStepAngles:
    """The maths, checked independently of any reconstruction."""

    def test_single_step_sits_at_the_frame_centre(self):
        np.testing.assert_allclose(sub_step_offsets(1.0, 1), [0.0])

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_offsets_are_symmetric_about_the_frame_centre(self, n):
        offsets = sub_step_offsets(1.0, n)
        np.testing.assert_allclose(offsets, -offsets[::-1], atol=1e-12)

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_offsets_span_exactly_one_oscillation(self, n):
        """n sub-steps must tile the frame, not overlap it or leave a gap."""
        offsets = sub_step_offsets(1.0, n)
        spacing = 1.0 / n
        np.testing.assert_allclose(np.diff(offsets), spacing, atol=1e-12)
        # First and last sit half a spacing inside the frame edges.
        assert offsets[0] == pytest.approx(-0.5 + spacing / 2)
        assert offsets[-1] == pytest.approx(0.5 - spacing / 2)

    def test_matches_meerkat2_microstep(self):
        """Meerkat2's misc.h Microstep(n): inc = 1/n, start = (-1 + inc)/2.

        The two codebases must not disagree about where a sub-step lands.
        """
        for n in (1, 2, 3, 4, 8):
            inc = 1.0 / n
            start = (-1.0 + inc) / 2
            expected = start + inc * np.arange(n)
            np.testing.assert_allclose(sub_step_offsets(1.0, n), expected, atol=1e-12)


class TestPhiMicrosteppingRuns:
    @pytest.mark.parametrize("n", [2, 4])
    def test_it_reconstructs(self, tmp_path, n):
        result = run_reference_reconstruction(tmp_path, microsteps=[1, 1, n])
        assert np.isfinite(result["data"]).any()

    def test_it_fills_more_voxels_than_no_microstepping(self, tmp_path):
        """The physical point of the feature.

        Subdividing the rotation places each frame's intensity at several angles
        instead of one, which closes gaps between frames. If this ever stops being
        true, the feature has stopped doing anything.
        """
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        plain = run_reference_reconstruction(tmp_path / "a")
        stepped = run_reference_reconstruction(tmp_path / "b", microsteps=[1, 1, 4])
        assert np.isfinite(stepped["data"]).sum() > np.isfinite(plain["data"]).sum()

    def test_microsteps_none_is_the_same_as_no_microstepping(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        default = run_reference_reconstruction(tmp_path / "a")
        explicit = run_reference_reconstruction(tmp_path / "b", microsteps=None)
        np.testing.assert_array_equal(default["data"], explicit["data"])


class TestFrameDecimation:
    """phi < 1 is frame decimation, not microstepping: 0.1 means every 10th frame.

    It was broken: 1/0.1 is a float, so np.arange produced a float array and
    scale[frame_number - first_image] raised IndexError. The path could never run.
    Fixed with int(round(...)).
    """

    @pytest.mark.parametrize("step, expected_frames", [(0.5, 10), (0.25, 5)])
    def test_reconstructs_a_subset_of_frames(self, tmp_path, step, expected_frames):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        every = run_reference_reconstruction(tmp_path / "a")
        some = run_reference_reconstruction(tmp_path / "b", microsteps=[1, 1, step])
        # Fewer frames -> strictly less of reciprocal space swept.
        assert np.isfinite(some["data"]).sum() < np.isfinite(every["data"]).sum()

    def test_finer_decimation_uses_fewer_frames(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        half = run_reference_reconstruction(tmp_path / "a", microsteps=[1, 1, 0.5])
        quarter = run_reference_reconstruction(tmp_path / "b", microsteps=[1, 1, 0.25])
        assert np.isfinite(quarter["data"]).sum() < np.isfinite(half["data"]).sum()

    def test_non_reciprocal_integer_is_refused(self, tmp_path):
        """0.3 does not mean 'every 3.33 frames'. Refuse rather than guess."""
        with pytest.raises(AssertionError, match="1/N"):
            run_reference_reconstruction(tmp_path, microsteps=[1, 1, 0.3])


class TestUnimplementedDirections:
    """The other two entries of the triple. Pinned so their state is unambiguous."""

    @pytest.mark.parametrize("microsteps", [[2, 1, 1], [1, 2, 1], [2, 2, 1]])
    def test_x_y_microstepping_is_refused(self, tmp_path, microsteps):
        """Sub-pixel microstepping has never executed: an assert blocks it, and the
        upsampling branch behind it had no return, so it could only produce None.

        meerkat/meerkat.py carries a comment at the assert describing where it would
        land and what it would cost, should anyone want it.
        """
        with pytest.raises(AssertionError, match="x and y are not implemented"):
            run_reference_reconstruction(tmp_path, microsteps=microsteps)

    def test_non_integer_x_y_is_refused(self, tmp_path):
        with pytest.raises(AssertionError, match="should be integer"):
            run_reference_reconstruction(tmp_path, microsteps=[1.5, 1, 1])

    def test_wrong_length_is_refused(self, tmp_path):
        with pytest.raises(AssertionError, match="three values"):
            run_reference_reconstruction(tmp_path, microsteps=[1, 1])
