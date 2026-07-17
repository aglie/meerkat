"""XPARM read/write round-tripping, and regressions for the two drifted copies.

write_xparm existed in four places and had diverged. These tests pin the behaviour
that the surviving copy must have.
"""

import numpy as np
import pytest

from meerkat.xds import cell2vecs, read_xparm, vecs2cell, write_xparm
from meerkat.xds.xparm import _scalar


def test_roundtrip_preserves_every_field(xparm_path, tmp_path):
    original = read_xparm(str(xparm_path))
    out = tmp_path / "XPARM.XDS"
    write_xparm(str(out), original)
    reread = read_xparm(str(out))

    assert set(reread) == set(original)
    for key, value in original.items():
        np.testing.assert_allclose(
            reread[key], value, rtol=1e-6, atol=1e-6, err_msg=f"field {key!r} did not survive"
        )


def test_write_is_idempotent(xparm_path, tmp_path):
    """write(read(write(read(f)))) == write(read(f)), byte for byte."""
    first = tmp_path / "a.XDS"
    second = tmp_path / "b.XDS"
    write_xparm(str(first), read_xparm(str(xparm_path)))
    write_xparm(str(second), read_xparm(str(first)))
    assert first.read_text() == second.read_text()


def test_detector_is_not_hardcoded(xparm_path, tmp_path):
    """Regression: xparm_transform_0.21.py:25 hardcoded a Pilatus 6M.

    It wrote "1475 1679 0.172000 0.172000" regardless of the params handed to it,
    which is why the script announced "THIS SCRIPT IS BROKEN / PIXEL SIZE IS WRONG".
    Use a detector that is nothing like a Pilatus so a regression is unmistakable.
    """
    params = read_xparm(str(xparm_path))
    params["NX"] = np.array([2463.0])
    params["NY"] = np.array([2527.0])
    params["pixelsize_x"] = np.array([0.05])
    params["pixelsize_y"] = np.array([0.06])

    out = tmp_path / "XPARM.XDS"
    write_xparm(str(out), params)

    text = out.read_text()
    assert "1475" not in text, "Pilatus 6M NX leaked into the output"
    assert "1679" not in text, "Pilatus 6M NY leaked into the output"

    reread = read_xparm(str(out))
    assert _scalar(reread["NX"]) == 2463
    assert _scalar(reread["NY"]) == 2527
    assert _scalar(reread["pixelsize_x"]) == pytest.approx(0.05)
    assert _scalar(reread["pixelsize_y"]) == pytest.approx(0.06)


def test_accepts_scalars_where_read_xparm_returns_arrays(xparm_path, tmp_path):
    """Regression: improve_orientation_v0.82.py:291 passed x_center to %f WITHOUT [0].

    That worked only because refining 'xycenter' replaced the 1-element array with a
    plain float first. With --refine cell it stayed an array, and float(1-element
    ndarray) is deprecated since numpy 1.25 and slated to raise. The params dict is
    genuinely type-unstable, so write_xparm must accept both.
    """
    import warnings

    params = read_xparm(str(xparm_path))
    # Exactly what refinement leaves behind: some fields scalars, others arrays.
    params["x_center"] = 1214.768921
    params["y_center"] = 1261.04126
    params["distance_to_detector"] = np.array([200.057556])

    out = tmp_path / "XPARM.XDS"
    with warnings.catch_warnings():
        # Verified on numpy 2.3.4: the old '%f' % np.array([...]) raises here.
        warnings.simplefilter("error", DeprecationWarning)
        write_xparm(str(out), params)

    reread = read_xparm(str(out))
    assert _scalar(reread["x_center"]) == pytest.approx(1214.768921)
    assert _scalar(reread["y_center"]) == pytest.approx(1261.04126)


def test_write_never_formats_a_bare_array(xparm_path, tmp_path):
    """The mechanism behind the x_center bug, guarded directly.

    Every scalar field must go through _scalar(). If any of them is passed to %f as a
    1-element array, numpy emits a DeprecationWarning (1.25+) that will become an
    error -- so turn it into an error here and write with the all-arrays dict that
    read_xparm actually returns.
    """
    import warnings

    params = read_xparm(str(xparm_path))  # every scalar is a 1-element array
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        write_xparm(str(tmp_path / "XPARM.XDS"), params)


def test_scalar_coercion_accepts_both_shapes():
    assert _scalar(np.array([3.5])) == 3.5
    assert _scalar(3.5) == 3.5
    assert _scalar(np.float64(3.5)) == 3.5


def test_write_rejects_a_malformed_cell(xparm_path, tmp_path):
    params = read_xparm(str(xparm_path))
    params["unit_cell_vectors"] = np.eye(2)
    with pytest.raises(ValueError, match="3x3"):
        write_xparm(str(tmp_path / "bad.XDS"), params)


def test_read_prefers_gxparm_over_xparm(xparm_path, tmp_path):
    """GXPARM.XDS is XDS's *refined* output; it must win when both are present."""
    params = read_xparm(str(xparm_path))
    write_xparm(str(tmp_path / "XPARM.XDS"), params)

    refined = dict(params)
    refined["distance_to_detector"] = np.array([999.0])
    write_xparm(str(tmp_path / "GXPARM.XDS"), refined)

    assert _scalar(read_xparm(str(tmp_path))["distance_to_detector"]) == pytest.approx(999.0)


def test_read_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        read_xparm("/nonexistent/XPARM.XDS")


def test_read_directory_without_xparm_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="XPARM"):
        read_xparm(str(tmp_path))


def test_read_rejects_a_non_xparm_file(tmp_path):
    junk = tmp_path / "junk.txt"
    junk.write_text("header\n1 2 3\n")
    with pytest.raises(ValueError):
        read_xparm(str(junk))


class TestCellConversions:
    def test_vecs2cell_matches_the_xparm_cell(self, xparm):
        """The file states its own cell; deriving it from the vectors must agree.

        This is a second, independent check of the rows-are-abc convention: with a
        transposed matrix the derived cell would not match what XDS wrote.
        """
        derived = vecs2cell(xparm["unit_cell_vectors"])
        np.testing.assert_allclose(derived, xparm["cell"], atol=1e-3)

    def test_cell2vecs_inverts_vecs2cell(self):
        cell = np.array([10.0, 11.0, 12.0, 85.0, 95.0, 100.0])
        np.testing.assert_allclose(vecs2cell(cell2vecs(cell)), cell, atol=1e-9)

    def test_cell2vecs_rows_have_the_right_lengths(self):
        cell = np.array([10.0, 11.0, 12.0, 85.0, 95.0, 100.0])
        vecs = cell2vecs(cell)
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), cell[:3], atol=1e-9)
