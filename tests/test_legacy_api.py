"""Pin the public API that already shipped on PyPI.

meerkat 0.3.8 is published, so `from meerkat import ...` and reconstruct_data's
keyword signature are a contract with existing user scripts. Stage 4 replaces the
engine underneath; these tests are what let that happen without breaking anyone.

The return contract is genuinely incoherent (see test_return_contract_*). It is
pinned bug-for-bug on purpose: "fixing" it is a breaking change that this
modernization has not agreed to make.
"""

import inspect

import numpy as np
import pytest
from synthetic import run_reference_reconstruction


def test_public_names_are_importable():
    """The names other code actually imports. improve_orientation uses the first two."""
    import meerkat

    for name in ("read_XPARM", "det2lab_xds", "rotvec2mat", "reconstruct_data"):
        assert hasattr(meerkat, name), f"meerkat.{name} disappeared -- breaking change"


def test_star_import_exposes_public_names():
    """`from meerkat import *` is how the README tells people to use this."""
    ns = {}
    exec("from meerkat import *", ns)
    for name in ("read_XPARM", "det2lab_xds", "reconstruct_data"):
        assert name in ns, f"{name} vanished from `from meerkat import *`"


def test_version_is_exposed():
    import meerkat

    assert isinstance(meerkat.__version__, str)


def test_reconstruct_data_signature():
    """Keyword names and defaults are the contract. Reordering breaks positional calls."""
    sig = inspect.signature(
        __import__("meerkat", fromlist=["reconstruct_data"]).reconstruct_data
    )
    params = list(sig.parameters)
    assert params[:5] == [
        "filename_template",
        "first_image",
        "last_image",
        "maxind",
        "number_of_pixels",
    ], "the five positional parameters changed -- breaking change"
    for name in (
        "reconstruct_in_orthonormal_basis",
        "measured_pixels",
        "microsteps",
        "unit_cell_transform_matrix",
        "polarization_plane_normal",
        "polarization_factor",
        "medium",
        "path_to_XPARM",
        "output_filename",
        "size_of_cache",
        "all_in_memory",
        "override",
        "scale",
        "keep_number_of_pixels",
    ):
        assert name in sig.parameters, f"keyword {name!r} disappeared -- breaking change"


def test_return_contract_dict_when_in_memory_and_no_file(tmp_path):
    """output_filename=None + all_in_memory=True -> a plain dict."""
    result = run_reference_reconstruction(tmp_path)
    assert isinstance(result, dict)
    assert "data" in result


def test_return_contract_none_when_writing_a_file(tmp_path):
    """Writing to a file returns None, not the file. Pinned bug-for-bug."""
    out = tmp_path / "out.h5"
    result = run_reference_reconstruction(
        tmp_path, output_filename=str(out), all_in_memory=True
    )
    assert result is None, "reconstruct_data returns None when it writes a file"
    assert out.exists()


def test_out_of_core_requires_an_output_filename(tmp_path):
    """all_in_memory=False with no filename raises rather than crashing obscurely."""
    with pytest.raises(Exception, match="output filename"):
        run_reference_reconstruction(tmp_path, output_filename=None, all_in_memory=False)


def test_refuses_to_overwrite_without_override(tmp_path):
    out = tmp_path / "twice.h5"
    run_reference_reconstruction(tmp_path, output_filename=str(out), all_in_memory=True)
    with pytest.raises(Exception, match="already exists"):
        run_reference_reconstruction(tmp_path, output_filename=str(out), all_in_memory=True)


def test_keep_number_of_pixels_selects_yell_09(tmp_path):
    """keep_number_of_pixels=True keeps both datasets and tags the older format."""
    result = run_reference_reconstruction(tmp_path, keep_number_of_pixels=True)
    assert result["format"] == "Yell 0.9"
    assert "rebinned_data" in result
    assert "number_of_pixels_rebinned" in result
    assert "data" not in result


def test_out_of_core_matches_in_memory(tmp_path):
    """The h5py low-level read-modify-write path must agree with the in-memory one.

    accumulate_intensity's out-of-core branch (meerkat.py:245-261) reaches into h5py
    internals (._id.read / ._id.write) because the high-level API has no +=. It is
    the most fragile code in the package and has never had a test. It stays
    load-bearing on a 16 GB machine, where 801^3 grids do not fit comfortably in RAM.
    """
    import h5py

    a = tmp_path / "a"
    a.mkdir()
    in_memory = run_reference_reconstruction(a)["data"]

    b = tmp_path / "b"
    b.mkdir()
    out = tmp_path / "ooc.h5"
    run_reference_reconstruction(b, output_filename=str(out), all_in_memory=False)
    with h5py.File(out, "r") as f:
        on_disk = f["data"][:]

    np.testing.assert_array_equal(on_disk, in_memory)
