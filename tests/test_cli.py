"""The `meerkat` command line.

The load-bearing test here is test_cli_reconstruction_matches_golden: this stage adds
interfaces, so the CLI must be a faithful translation onto the existing engine and
nothing may move. If that passes, the two driving interfaces are provably the same
computation.
"""

import numpy as np
import pytest
from synthetic import build_experiment

from meerkat.cli.main import main

h5py = pytest.importorskip("h5py")

GOLDEN = "tests/data/golden_31.npz"


def write_config(directory, xparm, template, **overrides):
    settings = {
        "DATA_FILE_TEMPLATE": template,
        "XPARM_FILE": str(xparm),
        "FIRST_FRAME": 1,
        "LAST_FRAME": 20,
        "NUMBER_OF_PIXELS": "31 31 31",
        "LOWER_LIMITS": "-1.5 -1.5 -1.5",
        "OUTPUT_FILENAME": str(directory / "out.h5"),
    }
    settings.update(overrides)
    lines = [f"{k} {v}" for k, v in settings.items()] + ["SYMMETRIC_LIMITS", "ALL_IN_MEMORY"]
    path = directory / "recon.mrk"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def experiment(tmp_path):
    xparm, template = build_experiment(tmp_path)
    return tmp_path, xparm, template


def test_version():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_no_command_prints_help():
    assert main([]) == 1


def _subcommands():
    """Discover subcommands from the parser rather than hardcoding them, so a new one
    is covered the moment it is added."""
    import argparse

    from meerkat.cli.main import build_parser

    for action in build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    return []


@pytest.mark.parametrize("command", _subcommands())
def test_subcommand_help_renders(command, capsys):
    """`meerkat <cmd> --help` must not crash.

    Regression: it did. argparse runs every help string through `help % params` in
    HelpFormatter._expand_help, and DATA_FILE_TEMPLATE's help necessarily contains a
    literal '%05i' -- which raised TypeError. Nothing else exercises --help, so this
    shipped-looking CLI was broken on its most basic invocation.
    """
    with pytest.raises(SystemExit) as e:
        main([command, "--help"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip()


def test_top_level_help_renders(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for command in _subcommands():
        assert command in out


def test_help_shows_the_template_example_literally(capsys):
    """The %%-escaping must not leak into what the user reads."""
    with pytest.raises(SystemExit):
        main(["reconstruct", "--help"])
    assert "%05i" in capsys.readouterr().out


def test_help_does_not_leak_the_suppress_sentinel(capsys):
    with pytest.raises(SystemExit):
        main(["reconstruct", "--help"])
    assert "SUPPRESS" not in capsys.readouterr().out


def test_cli_reconstruction_matches_golden(experiment, capsys):
    """The CLI must compute exactly what the library computes. Bit for bit."""
    directory, xparm, template = experiment
    config = write_config(directory, xparm, template)

    assert main(["reconstruct", str(config)]) == 0

    with h5py.File(directory / "out.h5", "r") as f:
        data = f["data"][:]
    expected = np.load(GOLDEN)["data"]
    np.testing.assert_array_equal(data, expected)


def test_dry_run_computes_nothing(experiment, capsys):
    directory, xparm, template = experiment
    config = write_config(directory, xparm, template)

    assert main(["reconstruct", str(config), "--dry-run"]) == 0
    assert not (directory / "out.h5").exists()

    out = capsys.readouterr().out
    assert "DATA_FILE_TEMPLATE" in out
    # SYMMETRIC_LIMITS must have been resolved into explicit limits.
    assert "UPPER_LIMITS 1.5 1.5 1.5" in out
    assert "STEP_SIZES 0.1 0.1 0.1" in out


def test_cli_flag_overrides_config_file(experiment, capsys):
    directory, xparm, template = experiment
    config = write_config(directory, xparm, template, POLARIZATION_FACTOR=1.0)

    main(["reconstruct", str(config), "--polarization-factor", "0.5", "--dry-run"])
    assert "POLARIZATION_FACTOR 0.5" in capsys.readouterr().out


def test_config_file_beats_defaults(experiment, capsys):
    directory, xparm, template = experiment
    config = write_config(directory, xparm, template, MEDIUM="Helium")

    main(["reconstruct", str(config), "--dry-run"])
    assert "MEDIUM Helium" in capsys.readouterr().out


def test_pure_cli_run_needs_no_config_file(experiment, capsys):
    """The LLM-friendly path: everything as flags, no file."""
    directory, xparm, template = experiment
    assert (
        main(
            [
                "reconstruct",
                "--data-file-template", template,
                "--xparm-file", str(xparm),
                "--first-frame", "1",
                "--last-frame", "20",
                "--number-of-pixels", "31", "31", "31",
                "--lower-limits", "-1.5", "-1.5", "-1.5",
                "--symmetric-limits",
                "--output-filename", str(directory / "flags.h5"),
                "--all-in-memory",
            ]
        )
        == 0
    )
    with h5py.File(directory / "flags.h5", "r") as f:
        np.testing.assert_array_equal(f["data"][:], np.load(GOLDEN)["data"])


def test_dump_config_round_trips_to_identical_data(experiment, tmp_path):
    """Dump the resolved config, re-run from it, and get the same bits.

    This is the property provenance depends on. It is checked here on the config
    itself; Stage 7 embeds the same text in the output file.
    """
    directory, xparm, template = experiment
    config = write_config(directory, xparm, template)
    redo = directory / "redo.mrk"

    main(["reconstruct", str(config), "--dump-config", str(redo), "--dry-run"])
    assert main(["reconstruct", str(config)]) == 0
    assert main(["reconstruct", str(redo), "--output-filename", str(directory / "b.h5")]) == 0

    with h5py.File(directory / "out.h5", "r") as f:
        first = f["data"][:]
    with h5py.File(directory / "b.h5", "r") as f:
        second = f["data"][:]
    np.testing.assert_array_equal(first, second)


def test_asymmetric_grid_is_refused_clearly(experiment, capsys):
    """The 0.3.x engine only takes a symmetric maxind. Say so, rather than silently
    reconstructing something else."""
    directory, xparm, template = experiment
    config = write_config(
        directory, xparm, template, LOWER_LIMITS="-1.5 -1.5 -1.5", UPPER_LIMITS="3.0 3.0 3.0"
    )
    # SYMMETRIC_LIMITS is appended by write_config and would contradict; drop it.
    config.write_text(config.read_text().replace("SYMMETRIC_LIMITS\n", ""))

    with pytest.raises(SystemExit, match="asymmetric"):
        main(["reconstruct", str(config)])


def test_bad_config_reports_the_line(experiment):
    directory, xparm, template = experiment
    config = directory / "bad.mrk"
    config.write_text("FIRST_FRAME 1\nNOT_A_KEYWORD 3\n")
    with pytest.raises(SystemExit, match="NOT_A_KEYWORD"):
        main(["reconstruct", str(config)])


class TestTransformXparm:
    def test_identity_transform_preserves_the_cell(self, xparm_path, tmp_path):
        from meerkat.xds import read_xparm

        out = tmp_path / "out.XDS"
        assert main(
            ["transform-xparm", "-i", str(xparm_path), "-o", str(out),
             "-t", "1", "0", "0", "0", "1", "0", "0", "0", "1"]
        ) == 0
        before, after = read_xparm(str(xparm_path)), read_xparm(str(out))
        np.testing.assert_allclose(after["cell"], before["cell"], atol=1e-3)
        np.testing.assert_allclose(
            after["unit_cell_vectors"], before["unit_cell_vectors"], atol=1e-5
        )

    def test_swap_a_and_b(self, xparm_path, tmp_path):
        from meerkat.xds import read_xparm

        out = tmp_path / "out.XDS"
        main(["transform-xparm", "-i", str(xparm_path), "-o", str(out),
              "-t", "0", "1", "0", "1", "0", "0", "0", "0", "-1"])
        before, after = read_xparm(str(xparm_path)), read_xparm(str(out))
        assert after["cell"][0] == pytest.approx(before["cell"][1], abs=1e-3)
        assert after["cell"][1] == pytest.approx(before["cell"][0], abs=1e-3)

    def test_detector_is_preserved(self, xparm_path, tmp_path):
        """The bug that made xparm_transform_0.21.py print 'THIS SCRIPT IS BROKEN'."""
        from meerkat.xds import read_xparm
        from meerkat.xds.xparm import _scalar

        out = tmp_path / "out.XDS"
        main(["transform-xparm", "-i", str(xparm_path), "-o", str(out),
              "-t", "1", "0", "0", "0", "1", "0", "0", "0", "1"])
        after = read_xparm(str(out))
        assert int(_scalar(after["NX"])) == 2463  # not 1475
        assert int(_scalar(after["NY"])) == 2527  # not 1679

    def test_singular_transform_is_refused(self, xparm_path, tmp_path):
        with pytest.raises(SystemExit, match="singular"):
            main(["transform-xparm", "-i", str(xparm_path), "-o", str(tmp_path / "o.XDS"),
                  "-t", "1", "0", "0", "1", "0", "0", "0", "0", "1"])

    def test_volume_changing_transform_needs_force(self, xparm_path, tmp_path):
        args = ["transform-xparm", "-i", str(xparm_path), "-o", str(tmp_path / "o.XDS"),
                "-t", "2", "0", "0", "0", "1", "0", "0", "0", "1"]
        with pytest.raises(SystemExit, match="volume"):
            main(args)
        assert main(args + ["--force"]) == 0


class TestInfo:
    def test_info_on_xparm(self, xparm_path, capsys):
        assert main(["info", str(xparm_path)]) == 0
        out = capsys.readouterr().out
        assert "8.2287" in out
        assert "2463 x 2527" in out
        assert "rows are a, b, c" in out

    def test_info_on_reconstruction(self, experiment, capsys):
        directory, xparm, template = experiment
        config = write_config(directory, xparm, template)
        main(["reconstruct", str(config)])
        capsys.readouterr()

        assert main(["info", str(directory / "out.h5")]) == 0
        out = capsys.readouterr().out
        assert "Yell 1.0" in out
        assert "(31, 31, 31)" in out
