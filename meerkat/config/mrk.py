"""The .mrk reconstruction config file: `KEYWORD value...`, one per line.

Format follows Meerkat2's, because that is the idiom this audience already lives in
(XDS.INP, SPOT.XDS, GXPARM.XDS are all keyword files) and because a plain text file
sitting next to the data is a provenance trace that survives being emailed around.
Compatibility with Meerkat2 is intentional but not strict -- the keyword sets differ
where the two programs differ.

Deliberate divergences from Meerkat2's parser, all bug fixes:
  * A duplicate keyword is an error rather than last-wins. See parse_mrk.
  * The grid is resolved and cross-checked rather than overwritten. See resolve_grid.
  * Missing values are None, not NaN sentinels that never compare equal.
"""

from __future__ import annotations

import numpy as np

from .params import (
    PARAMETER_SPEC,
    ConfigError,
    ReconstructionParameters,
    spec_by_keyword,
)

__all__ = ["dump_mrk", "parse_mrk", "read_mrk"]

_COMMENT_CHARS = "#!"


def _strip_comment(line: str) -> str:
    for i, ch in enumerate(line):
        if ch in _COMMENT_CHARS:
            return line[:i]
    return line


def _convert(spec, tokens, filename, lineno, line):
    if spec.nargs == 0:  # a flag
        if tokens:
            raise ConfigError(
                f"{spec.keyword} is a flag and takes no values, got {' '.join(tokens)}",
                filename, lineno, line,
            )
        return True

    if spec.nargs is None:
        if len(tokens) != 1:
            raise ConfigError(
                f"{spec.keyword} takes exactly one value, got {len(tokens)}",
                filename, lineno, line,
            )
        try:
            return spec.type(tokens[0])
        except ValueError:
            raise ConfigError(
                f"{spec.keyword}: {tokens[0]!r} is not a valid "
                f"{getattr(spec.type, '__name__', spec.type)}",
                filename, lineno, line,
            ) from None

    if len(tokens) != spec.nargs:
        raise ConfigError(
            f"{spec.keyword} takes exactly {spec.nargs} values, got {len(tokens)}",
            filename, lineno, line,
        )
    try:
        return [spec.type(t) for t in tokens]
    except ValueError:
        raise ConfigError(
            f"{spec.keyword}: {' '.join(tokens)!r} are not all valid "
            f"{getattr(spec.type, '__name__', spec.type)}",
            filename, lineno, line,
        ) from None


def parse_mrk(text: str, filename: str = "<config>") -> dict:
    """Parse .mrk text into a dict of parameter name -> value.

    Returns only what the file actually set, so callers can distinguish "not
    mentioned" from "set to the default" -- which is what makes the
    defaults < config < CLI precedence work.
    """
    values: dict = {}
    seen_at: dict = {}

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw).strip()
        if not line:
            continue

        tokens = line.split()
        keyword = tokens[0].upper()
        spec = spec_by_keyword(keyword, filename, lineno, raw)

        if spec.name in seen_at:
            raise ConfigError(
                f"{keyword} was already given on line {seen_at[spec.name]}. "
                f"Duplicate keywords are an error: a config that silently means "
                f"something other than it reads is worse than no config at all.",
                filename, lineno, raw,
            )
        seen_at[spec.name] = lineno
        values[spec.name] = _convert(spec, tokens[1:], filename, lineno, raw)

    return values


def read_mrk(path) -> dict:
    with open(path) as f:
        return parse_mrk(f.read(), filename=str(path))


def _format_value(spec, value) -> str | None:
    if value is None:
        return None
    if spec.nargs == 0:
        return spec.keyword if value else None
    if spec.nargs is None:
        return f"{spec.keyword} {value}"
    flat = np.ravel(np.asarray(value)).tolist()
    return spec.keyword + " " + " ".join(_format_scalar(v) for v in flat)


def _format_scalar(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return f"{v:.10g}" if isinstance(v, float) else str(v)


def dump_mrk(params: ReconstructionParameters, resolved_grid=True) -> str:
    """Render params back to .mrk text.

    With resolved_grid=True (the default) all four of lower/upper/step/n are written
    explicitly. That makes the output re-runnable with no inference: reading it back
    exercises resolve_grid's over-determined-but-consistent path, which is precisely
    why that path must be accepted rather than rejected.
    """
    grid = params.grid() if resolved_grid else None

    lines = []
    for spec in PARAMETER_SPEC:
        if spec.name == "symmetric_limits" and resolved_grid:
            # Redundant once explicit limits are written, and would double-mirror.
            continue

        value = getattr(params, spec.name)
        grid_fields = ("lower_limits", "upper_limits", "step_sizes", "number_of_pixels")
        if grid is not None and spec.name in grid_fields:
            value = getattr(grid, spec.name)

        rendered = _format_value(spec, value)
        if rendered is not None:
            lines.append(rendered)

    return "\n".join(lines) + "\n"
