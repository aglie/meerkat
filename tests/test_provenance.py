"""Provenance recorded in the reconstruction itself.

The acceptance criterion is test_round_trip_reproduces_the_data_exactly: given only a
.h5, recover a config and reproduce the file. Everything else here supports that.
"""

import json

import numpy as np
import pytest
from synthetic import build_experiment

from meerkat.cli.main import main
from meerkat.io import PROVENANCE_GROUP, read_provenance

h5py = pytest.importorskip("h5py")

CONFIG = """\
# a comment that must survive into the output
DATA_FILE_TEMPLATE {template}
XPARM_FILE {xparm}
FIRST_FRAME 1
LAST_FRAME 20
NUMBER_OF_PIXELS 31 31 31
LOWER_LIMITS -1.5 -1.5 -1.5
SYMMETRIC_LIMITS
OUTPUT_FILENAME {output}
ALL_IN_MEMORY
"""


@pytest.fixture
def reconstructed(tmp_path):
    xparm, template = build_experiment(tmp_path)
    output = tmp_path / "a.h5"
    config = tmp_path / "recon.mrk"
    config.write_text(CONFIG.format(template=template, xparm=xparm, output=output))
    assert main(["reconstruct", str(config)]) == 0
    return tmp_path, config, output


class TestYellCompatibility:
    """Provenance must be invisible to Yell.

    Yell's entire HDF5 read is file.openDataSet("data") -- by name
    (Yell/src/IntensityMap.cpp:20) -- and there is no root enumeration anywhere in
    Yell/src or FTL/src. Checked before this was built, because if Yell had walked
    the root the whole design would have had to be attributes instead.
    """

    def test_provenance_lives_in_one_group(self, reconstructed):
        _, _, output = reconstructed
        with h5py.File(output, "r") as f:
            assert isinstance(f[PROVENANCE_GROUP], h5py.Group)

    def test_yell_datasets_are_untouched(self, reconstructed):
        _, _, output = reconstructed
        with h5py.File(output, "r") as f:
            for key in ("data", "format", "unit_cell", "metric_tensor", "step_sizes",
                        "lower_limits", "is_direct"):
                assert key in f, f"Yell requires {key!r}"
            assert f["data"][:].shape == (31, 31, 31)

    def test_no_new_root_datasets(self, reconstructed):
        """Only ONE new root member, and it is a group."""
        _, _, output = reconstructed
        with h5py.File(output, "r") as f:
            extra = {k for k in f if not isinstance(f[k], h5py.Group)}
            expected = {"data", "format", "space_group_nr", "unit_cell", "metric_tensor",
                        "step_sizes", "lower_limits", "is_direct"}
            assert extra <= expected, f"unexpected root datasets: {extra - expected}"


class TestRoundTrip:
    def test_round_trip_reproduces_the_data_exactly(self, reconstructed, capsys):
        """The whole point: a .h5 is enough to reproduce itself."""
        tmp_path, _, output = reconstructed

        assert main(["dump-config", str(output)]) == 0
        recovered = tmp_path / "redo.mrk"
        recovered.write_text(capsys.readouterr().out)

        again = tmp_path / "b.h5"
        assert main(["reconstruct", str(recovered), "--output-filename", str(again)]) == 0

        with h5py.File(output, "r") as f:
            first = f["data"][:]
        with h5py.File(again, "r") as f:
            second = f["data"][:]
        np.testing.assert_array_equal(first, second)

    def test_resolved_config_needs_no_inference(self, reconstructed):
        """All four grid quantities written explicitly, so a re-run derives nothing."""
        _, _, output = reconstructed
        text = read_provenance(output)["config_resolved"]
        for keyword in ("LOWER_LIMITS", "UPPER_LIMITS", "STEP_SIZES", "NUMBER_OF_PIXELS"):
            assert keyword in text
        assert "SYMMETRIC_LIMITS" not in text, "would double-mirror on re-read"

    def test_original_config_is_kept_verbatim(self, reconstructed):
        """Comments and all -- the resolved config cannot carry intent."""
        _, config, output = reconstructed
        assert read_provenance(output)["config_text"] == config.read_text()
        assert "a comment that must survive" in read_provenance(output)["config_text"]


class TestRecordContents:
    def test_records_the_xparm_verbatim_and_hashed(self, reconstructed):
        """So the geometry is recoverable even if the XPARM moves or is re-refined."""
        tmp_path, _, output = reconstructed
        record = read_provenance(output)
        assert record["xparm_text"].startswith(" XPARM.XDS")
        assert len(record["xparm_sha256"]) == 64
        assert record["xparm_path"].endswith("XPARM.XDS")

    def test_records_the_command_line(self, reconstructed):
        _, _, output = reconstructed
        assert "reconstruct" in read_provenance(output)["command_line"]

    def test_records_versions(self, reconstructed):
        _, _, output = reconstructed
        versions = json.loads(read_provenance(output)["versions"])
        assert "meerkat" in versions
        assert "numpy" in versions

    def test_records_which_frames_were_used(self, reconstructed):
        _, _, output = reconstructed
        frames = read_provenance(output)["frames_used"]
        np.testing.assert_array_equal(frames, np.arange(1, 21))

    def test_records_when_and_where(self, reconstructed):
        _, _, output = reconstructed
        record = read_provenance(output)
        assert record["timestamp_utc"].startswith("20")
        assert record["hostname"]
        assert record["cwd"]

    def test_frame_checksums_are_off_by_default(self, reconstructed):
        """Hashing 100 GB of frames to write a 1 GB file is rarely the trade wanted."""
        _, _, output = reconstructed
        inputs = json.loads(read_provenance(output)["input_files"])
        assert not [i for i in inputs if i["role"] == "frame"]

    def test_frame_checksums_can_be_requested(self, tmp_path):
        xparm, template = build_experiment(tmp_path)
        output = tmp_path / "c.h5"
        config = tmp_path / "c.mrk"
        config.write_text(CONFIG.format(template=template, xparm=xparm, output=output))
        main(["reconstruct", str(config), "--checksum-frames"])

        inputs = json.loads(read_provenance(output)["input_files"])
        frames = [i for i in inputs if i["role"] == "frame"]
        assert len(frames) == 20
        assert all(len(f["sha256"]) == 64 for f in frames)


class TestSidecar:
    def test_sidecar_is_written_next_to_the_output(self, reconstructed):
        """The trace someone will actually find. The .h5 copy is the backup."""
        _, _, output = reconstructed
        sidecar = output.with_suffix(output.suffix + ".mrk")
        assert sidecar.exists()
        assert "DATA_FILE_TEMPLATE" in sidecar.read_text()

    def test_sidecar_matches_the_embedded_resolved_config(self, reconstructed):
        _, _, output = reconstructed
        sidecar = output.with_suffix(output.suffix + ".mrk")
        assert sidecar.read_text() == read_provenance(output)["config_resolved"]

    def test_sidecar_can_be_disabled(self, tmp_path):
        xparm, template = build_experiment(tmp_path)
        output = tmp_path / "d.h5"
        config = tmp_path / "d.mrk"
        config.write_text(CONFIG.format(template=template, xparm=xparm, output=output))
        main(["reconstruct", str(config), "--no-sidecar"])
        assert not output.with_suffix(output.suffix + ".mrk").exists()


class TestOptOut:
    def test_provenance_can_be_disabled(self, tmp_path):
        xparm, template = build_experiment(tmp_path)
        output = tmp_path / "e.h5"
        config = tmp_path / "e.mrk"
        config.write_text(CONFIG.format(template=template, xparm=xparm, output=output))
        main(["reconstruct", str(config), "--no-provenance"])
        assert read_provenance(output) == {}

    def test_dump_config_on_a_file_without_provenance_says_so(self, tmp_path):
        """meerkat < 0.4 wrote no provenance; do not pretend otherwise."""
        path = tmp_path / "old.h5"
        with h5py.File(path, "w") as f:
            f["data"] = np.zeros((3, 3, 3), dtype="float32")
        with pytest.raises(SystemExit, match="no provenance"):
            main(["dump-config", str(path)])

    def test_dump_config_original_on_a_flags_only_run(self, tmp_path):
        xparm, template = build_experiment(tmp_path)
        output = tmp_path / "f.h5"
        main([
            "reconstruct",
            "--data-file-template", template,
            "--xparm-file", str(xparm),
            "--first-frame", "1", "--last-frame", "20",
            "--number-of-pixels", "31", "31", "31",
            "--lower-limits", "-1.5", "-1.5", "-1.5",
            "--symmetric-limits", "--all-in-memory",
            "--output-filename", str(output),
        ])
        # There was no config file, so --original has nothing to show and must say so
        # rather than print an empty string.
        with pytest.raises(SystemExit, match="entirely by command-line flags"):
            main(["dump-config", str(output), "--original"])
        # ...but the resolved config still reproduces the run.
        assert main(["dump-config", str(output)]) == 0


def test_info_reports_provenance_presence(reconstructed, capsys):
    _, _, output = reconstructed
    main(["info", str(output)])
    assert "provenance      : present" in capsys.readouterr().out
