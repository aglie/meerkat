"""The meerkat-ewald contract: meerkat.xds must be importable with numpy alone.

The Qt viewer needs XPARM parsing and det2lab_xds. It does not need fabio (image
decoding) or h5py (reconstruction output), and Qt/OpenGL dependencies are painful
enough on their own without inheriting a scientific image stack to read a text file.

Before this split, `import meerkat` ran `from .meerkat import *`, which imports fabio
and h5py at module scope -- so there was no way to touch read_XPARM without them.

Checked in a subprocess because sys.modules is process-global: by the time this test
runs, pytest has already imported fabio via other tests, and an in-process check
would pass vacuously.
"""

import subprocess
import sys
import textwrap

HEAVY = ("fabio", "h5py", "scipy")


def _run(code):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout.strip()


def test_importing_meerkat_xds_does_not_import_fabio_or_h5py():
    _run(
        f"""
        import sys
        import meerkat.xds
        loaded = {HEAVY!r}
        offenders = sorted(m for m in loaded if m in sys.modules)
        assert not offenders, (
            "importing meerkat.xds pulled in " + repr(offenders) +
            " -- this breaks the meerkat-ewald dependency contract"
        )
        print("ok")
        """
    )


def test_importing_meerkat_xds_xparm_does_not_import_fabio_or_h5py():
    """The narrowest import, and the one meerkat-ewald actually uses."""
    _run(
        f"""
        import sys
        import meerkat.xds.xparm
        offenders = sorted(m for m in {HEAVY!r} if m in sys.modules)
        assert not offenders, "meerkat.xds.xparm pulled in " + repr(offenders)
        print("ok")
        """
    )


def test_xds_can_parse_an_xparm_without_heavy_deps():
    """Not just importable -- actually usable. An import guard that never runs the
    code would not notice a lazy `import fabio` inside a function."""
    _run(
        f"""
        import sys
        from pathlib import Path
        from meerkat.xds import det2lab_xds, read_xparm

        p = read_xparm(str(Path("tests/data/XPARM.XDS")))
        assert p["unit_cell_vectors"].shape == (3, 3)
        det2lab_xds(__import__("numpy").array([[100.0, 100.0]]), 5.0, **p)

        offenders = sorted(m for m in {HEAVY!r} if m in sys.modules)
        assert not offenders, "using meerkat.xds pulled in " + repr(offenders)
        print("ok")
        """
    )


def test_importing_meerkat_itself_stays_light():
    """`import meerkat` must not eagerly pull the reconstruction stack either.

    This is what the PEP 562 __getattr__ in meerkat/__init__.py buys: the names still
    resolve, but only when touched.
    """
    _run(
        f"""
        import sys
        import meerkat
        offenders = sorted(m for m in {HEAVY!r} if m in sys.modules)
        assert not offenders, (
            "`import meerkat` eagerly imported " + repr(offenders) +
            " -- the lazy __getattr__ has been defeated"
        )
        print("ok")
        """
    )


def test_touching_reconstruct_data_does_import_fabio():
    """The other half of lazy: the heavy path must still work on demand."""
    _run(
        """
        import sys
        import meerkat
        assert "fabio" not in sys.modules
        meerkat.reconstruct_data           # touching it resolves the import
        assert "fabio" in sys.modules, "reconstruct_data did not pull in fabio"
        print("ok")
        """
    )
