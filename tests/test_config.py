"""The .mrk config format and the parameter spec."""

from dataclasses import fields

import numpy as np
import pytest

from meerkat.config import (
    PARAMETER_SPEC,
    ConfigError,
    ReconstructionParameters,
    dump_mrk,
    parse_mrk,
    resolve_grid,
)

MINIMAL = """
DATA_FILE_TEMPLATE frames/img_%05i.cbf
FIRST_FRAME 1
LAST_FRAME 360
NUMBER_OF_PIXELS 501 501 501
LOWER_LIMITS -7 -7 -7
SYMMETRIC_LIMITS
"""


def test_spec_matches_dataclass():
    """One definition per parameter, enforced.

    This one-line test is load-bearing: it is the only thing stopping the CLI, the
    config parser, and the dumper from drifting apart. If they drift, the provenance
    record silently misreports what was run -- which defeats the purpose of having
    one.
    """
    assert {f.name for f in fields(ReconstructionParameters)} == {s.name for s in PARAMETER_SPEC}


def test_keywords_are_unique():
    keywords = [s.keyword for s in PARAMETER_SPEC]
    assert len(keywords) == len(set(keywords))


class TestParsing:
    def test_minimal_config(self):
        v = parse_mrk(MINIMAL)
        assert v["data_file_template"] == "frames/img_%05i.cbf"
        assert v["first_frame"] == 1
        assert v["last_frame"] == 360
        assert v["number_of_pixels"] == [501, 501, 501]
        assert v["lower_limits"] == [-7.0, -7.0, -7.0]
        assert v["symmetric_limits"] is True

    def test_returns_only_what_was_set(self):
        """Not-mentioned must be distinguishable from set-to-the-default, or the
        defaults < config < CLI precedence collapses."""
        v = parse_mrk(MINIMAL)
        assert "polarization_factor" not in v
        assert "medium" not in v

    @pytest.mark.parametrize("comment", ["#", "!"])
    def test_comments(self, comment):
        v = parse_mrk(f"FIRST_FRAME 1 {comment} start here\n{comment} whole line\nLAST_FRAME 9\n")
        assert v == {"first_frame": 1, "last_frame": 9}

    def test_blank_lines_and_indentation(self):
        assert parse_mrk("\n\n   FIRST_FRAME    1   \n\n") == {"first_frame": 1}

    def test_keywords_are_case_insensitive(self):
        assert parse_mrk("first_frame 1\n") == {"first_frame": 1}


class TestParsingErrors:
    def test_unknown_keyword_reports_line_and_suggests(self):
        with pytest.raises(ConfigError) as e:
            parse_mrk("FIRST_FRAME 1\nNUMBER_OF_PIXEL 5 5 5\n", filename="t.mrk")
        msg = str(e.value)
        assert "t.mrk:2" in msg
        assert "NUMBER_OF_PIXELS" in msg  # the suggestion

    def test_duplicate_keyword_is_an_error(self):
        """Meerkat2 silently takes the last one. In a tool whose output is meant to be
        a provenance record, a config that means something other than it reads is the
        worst failure mode available."""
        with pytest.raises(ConfigError, match="already given on line 1"):
            parse_mrk("FIRST_FRAME 1\nFIRST_FRAME 2\n")

    def test_wrong_number_of_values(self):
        with pytest.raises(ConfigError, match="exactly 3 values"):
            parse_mrk("NUMBER_OF_PIXELS 5 5\n")

    def test_flag_takes_no_values(self):
        with pytest.raises(ConfigError, match="flag"):
            parse_mrk("SYMMETRIC_LIMITS yes\n")

    def test_non_numeric_value(self):
        with pytest.raises(ConfigError, match="not a valid"):
            parse_mrk("FIRST_FRAME banana\n")


class TestResolveGrid:
    def test_symmetric_from_lower(self):
        g = resolve_grid(lower=[-7, -7, -7], n=[501, 501, 501], symmetric=True)
        np.testing.assert_allclose(g.upper_limits, [7, 7, 7])
        np.testing.assert_allclose(g.step_sizes, 14.0 / 500)
        assert g.is_symmetric

    def test_symmetric_from_upper(self):
        g = resolve_grid(upper=[7, 7, 7], n=[501, 501, 501], symmetric=True)
        np.testing.assert_allclose(g.lower_limits, [-7, -7, -7])

    def test_derive_n_from_limits_and_step(self):
        g = resolve_grid(lower=[-1, -1, -1], upper=[1, 1, 1], step=[0.1, 0.1, 0.1])
        np.testing.assert_array_equal(g.number_of_pixels, [21, 21, 21])

    def test_derive_step_from_limits_and_n(self):
        g = resolve_grid(lower=[-1, -1, -1], upper=[1, 1, 1], n=[21, 21, 21])
        np.testing.assert_allclose(g.step_sizes, 0.1)

    def test_derive_upper(self):
        g = resolve_grid(lower=[-1, -1, -1], step=[0.1, 0.1, 0.1], n=[21, 21, 21])
        np.testing.assert_allclose(g.upper_limits, [1, 1, 1])

    def test_over_determined_but_consistent_is_accepted(self):
        """dump_mrk writes all four explicitly, so this path must work -- it is what
        makes a dumped config re-runnable without re-deriving anything."""
        g = resolve_grid(lower=[-1, -1, -1], upper=[1, 1, 1], step=[0.1, 0.1, 0.1], n=[21, 21, 21])
        np.testing.assert_allclose(g.step_sizes, 0.1)

    def test_over_determined_and_inconsistent_is_rejected(self):
        with pytest.raises(ConfigError, match="over-determined and inconsistent"):
            resolve_grid(lower=[-1, -1, -1], upper=[1, 1, 1], step=[0.5, 0.5, 0.5], n=[21, 21, 21])

    def test_under_determined_names_the_missing_keywords(self):
        with pytest.raises(ConfigError, match="under-determined"):
            resolve_grid(lower=[-1, -1, -1], n=[21, 21, 21])

    def test_non_integer_grid_size_is_rejected(self):
        with pytest.raises(ConfigError, match="non-integer"):
            resolve_grid(lower=[0, 0, 0], upper=[1, 1, 1], step=[0.3, 0.3, 0.3])

    def test_symmetric_needs_something_to_mirror(self):
        with pytest.raises(ConfigError, match="needs LOWER_LIMITS or UPPER_LIMITS"):
            resolve_grid(n=[21, 21, 21], step=[0.1, 0.1, 0.1], symmetric=True)

    def test_symmetric_contradicted_by_explicit_limits(self):
        with pytest.raises(ConfigError, match="not the negative"):
            resolve_grid(lower=[-1, -1, -1], upper=[2, 2, 2], n=[21, 21, 21], symmetric=True)

    def test_no_nan_sentinels(self):
        """Meerkat2 guards with `lower_limits[0] == NAN`, which is dead code: NaN
        never compares equal. A missing LOWER_LIMITS must be an error, not a NaN that
        propagates into to_index."""
        with pytest.raises(ConfigError):
            resolve_grid(n=[21, 21, 21])

    def test_asymmetric_grid_resolves(self):
        g = resolve_grid(lower=[-2, -1, -1], upper=[6, 1, 1], n=[81, 21, 21])
        assert not g.is_symmetric
        np.testing.assert_allclose(g.step_sizes, 0.1)


class TestRoundTrip:
    def _params(self):
        return ReconstructionParameters(
            data_file_template="frames/img_%05i.cbf",
            first_frame=1,
            last_frame=360,
            number_of_pixels=[501, 501, 501],
            lower_limits=[-7.0, -7.0, -7.0],
            symmetric_limits=True,
        ).validated()

    def test_dump_then_parse_gives_the_same_grid(self):
        original = self._params()
        reparsed = ReconstructionParameters(**parse_mrk(dump_mrk(original))).validated()

        a, b = original.grid(), reparsed.grid()
        np.testing.assert_allclose(a.lower_limits, b.lower_limits)
        np.testing.assert_allclose(a.upper_limits, b.upper_limits)
        np.testing.assert_allclose(a.step_sizes, b.step_sizes)
        np.testing.assert_array_equal(a.number_of_pixels, b.number_of_pixels)

    def test_dump_is_idempotent(self):
        first = dump_mrk(self._params())
        second = dump_mrk(ReconstructionParameters(**parse_mrk(first)).validated())
        assert first == second

    def test_dump_writes_the_resolved_grid_explicitly(self):
        """So a re-run infers nothing -- the point of the provenance trace."""
        text = dump_mrk(self._params())
        for keyword in ("LOWER_LIMITS", "UPPER_LIMITS", "STEP_SIZES", "NUMBER_OF_PIXELS"):
            assert keyword in text
        assert "SYMMETRIC_LIMITS" not in text, "would double-mirror on re-read"


class TestValidation:
    def _base(self, **kw):
        args = dict(
            data_file_template="f_%05i.cbf",
            first_frame=1,
            last_frame=10,
            number_of_pixels=[11, 11, 11],
            lower_limits=[-1.0, -1.0, -1.0],
            symmetric_limits=True,
        )
        args.update(kw)
        return ReconstructionParameters(**args)

    def test_valid(self):
        assert self._base().validated() is not None

    @pytest.mark.parametrize("missing", ["data_file_template", "first_frame", "last_frame"])
    def test_required_fields(self, missing):
        with pytest.raises(ConfigError, match="required"):
            self._base(**{missing: None}).validated()

    def test_frames_out_of_order(self):
        with pytest.raises(ConfigError, match="before FIRST_FRAME"):
            self._base(first_frame=10, last_frame=1).validated()

    def test_template_without_a_format_specifier(self):
        with pytest.raises(ConfigError, match="printf-style"):
            self._base(data_file_template="frames.cbf").validated()

    def test_bad_medium(self):
        with pytest.raises(ConfigError, match="Air or Helium"):
            self._base(medium="Nitrogen").validated()

    def test_bad_output_format(self):
        with pytest.raises(ConfigError, match="YELL"):
            self._base(output_format="Yell 1.0").validated()

    def test_microstep_frames_is_accepted(self):
        """phi microstepping works and is supported. Only x/y sub-pixel stepping is
        unimplemented, and that is not reachable from the config at all."""
        assert self._base(microstep_frames=4).validated() is not None

    def test_microstep_frames_must_be_at_least_one(self):
        with pytest.raises(ConfigError, match="at least 1"):
            self._base(microstep_frames=0).validated()

    def test_every_nth_frame_is_accepted(self):
        assert self._base(reconstruct_every_nth_frame=10).validated() is not None

    def test_every_nth_frame_must_be_at_least_one(self):
        with pytest.raises(ConfigError, match="at least 1"):
            self._base(reconstruct_every_nth_frame=0).validated()

    def test_microstepping_and_decimation_cannot_be_combined(self):
        """The legacy engine encodes both on the same axis of the microsteps triple."""
        with pytest.raises(ConfigError, match="cannot be combined"):
            self._base(microstep_frames=2, reconstruct_every_nth_frame=2).validated()
