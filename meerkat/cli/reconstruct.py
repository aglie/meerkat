"""`meerkat reconstruct` -- config file and/or command-line driven.

Precedence: built-in defaults < config file < command-line flags.

The mechanism that makes this work is `default=argparse.SUPPRESS`: unspecified flags
are simply absent from the parsed namespace, so "the user typed --polarization-factor 1"
is distinguishable from "1 happens to be the default". Without that, every default
would silently override the config file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..config import PARAMETER_SPEC, ConfigError, ReconstructionParameters, dump_mrk, read_mrk

__all__ = ["add_arguments", "resolve", "run"]


def add_arguments(parser):
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        metavar="CONFIG.mrk",
        help="reconstruction parameter file. Every setting can also be given as a "
        "flag below; flags win over the file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fully resolved configuration and exit without reconstructing",
    )
    parser.add_argument(
        "--dump-config",
        metavar="FILE",
        default=None,
        help="write the resolved configuration to FILE (use '-' for stdout)",
    )
    parser.add_argument(
        "--no-provenance",
        action="store_true",
        help="do not record how this reconstruction was made in the output file",
    )
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="do not write the <output>.mrk sidecar next to the output file",
    )
    parser.add_argument(
        "--checksum-frames",
        action="store_true",
        help="also record a sha256 of every input frame. Off by default: hashing "
        "100 GB of frames to write a 1 GB output is rarely worth it",
    )

    group = parser.add_argument_group("reconstruction parameters")
    for spec in PARAMETER_SPEC:
        if spec.nargs == 0:
            group.add_argument(
                spec.flag,
                action="store_true",
                default=argparse.SUPPRESS,
                help=_help(spec),
            )
        else:
            group.add_argument(
                spec.flag,
                nargs=spec.nargs,
                type=spec.type,
                default=argparse.SUPPRESS,
                metavar=_metavar(spec),
                help=_help(spec),
            )
    return parser


def _metavar(spec):
    if spec.nargs is None:
        return spec.name.split("_")[-1].upper()
    return tuple(spec.keyword.split("_")[-1][:1].upper() + str(i) for i in range(spec.nargs))


def _help(spec):
    """Render help text for argparse, with the default appended.

    The escaping is not optional: argparse runs every help string through
    `help % params` in HelpFormatter._expand_help, regardless of formatter class. Our
    DATA_FILE_TEMPLATE help necessarily contains a literal '%05i', which raises
    TypeError there unless the percent signs are doubled.
    """
    text = spec.help if spec.default is None else f"{spec.help} (default: {spec.default})"
    return text.replace("%", "%%")


def resolve(args) -> ReconstructionParameters:
    """Merge defaults < config file < CLI flags into validated parameters."""
    from_file = read_mrk(args.config) if args.config else {}

    from_cli = {
        spec.name: getattr(args, spec.name)
        for spec in PARAMETER_SPEC
        if hasattr(args, spec.name)
    }

    merged = {**from_file, **from_cli}
    return ReconstructionParameters(**merged).validated()


def run(args) -> int:
    try:
        params = resolve(args)
    except ConfigError as exc:
        raise SystemExit(f"error: {exc}") from None

    text = dump_mrk(params)

    if args.dump_config:
        if args.dump_config == "-":
            print(text, end="")
        else:
            with open(args.dump_config, "w") as f:
                f.write(text)

    if args.dry_run:
        if args.dump_config != "-":
            print(text, end="")
        return 0

    config_text = Path(args.config).read_text() if args.config else ""

    _reconstruct(params)

    output = Path(params.output_filename)

    if not args.no_sidecar:
        # A text file beside the data is the trace someone will actually find in two
        # years. The copy inside the .h5 is for when this one gets lost.
        sidecar = output.with_suffix(output.suffix + ".mrk")
        sidecar.write_text(text)
        print(f"wrote {sidecar}")

    if not args.no_provenance:
        from ..io import write_provenance

        write_provenance(
            output,
            params,
            config_text=config_text,
            argv=["meerkat", "reconstruct", *(sys.argv[2:] if len(sys.argv) > 2 else [])],
            checksum_frames=args.checksum_frames,
        )
        print(f"recorded provenance in {output}")

    return 0


def _reconstruct(params: ReconstructionParameters) -> None:
    """Drive the existing engine.

    Deliberately a translation layer rather than a new engine: this stage adds the
    interfaces and leaves the reconstruction maths untouched, so the golden test still
    proves nothing moved. The engine decomposition is a separate, reviewable step.
    """
    from ..meerkat import reconstruct_data

    grid = params.grid()

    if not grid.is_symmetric:
        raise SystemExit(
            "error: asymmetric grids are not supported yet.\n"
            f"  LOWER_LIMITS {list(grid.lower_limits)} is not the negative of "
            f"UPPER_LIMITS {list(grid.upper_limits)}.\n"
            "  The 0.3.x engine only accepts a symmetric half-width (maxind); "
            "general limits arrive with the engine decomposition."
        )

    kwargs = dict(
        filename_template=params.data_file_template,
        first_image=params.first_frame,
        last_image=params.last_frame,
        maxind=list(grid.maxind),
        number_of_pixels=list(grid.number_of_pixels),
        path_to_XPARM=params.xparm_file,
        output_filename=params.output_filename,
        polarization_factor=params.polarization_factor,
        polarization_plane_normal=list(params.polarization_plane_normal),
        medium=params.medium,
        reconstruct_in_orthonormal_basis=params.reconstruct_in_orthonormal_basis,
        all_in_memory=params.all_in_memory,
        override=params.overwrite,
        size_of_cache=params.size_of_cache,
        keep_number_of_pixels=(params.output_format == "YELL_0.9"),
    )

    # The legacy engine takes one `microsteps` triple [x, y, phi], where phi > 1
    # subdivides each frame's rotation and phi < 1 (as 1/N) skips frames. The config
    # splits those into two honest keywords; map them back. x/y sub-pixel
    # microstepping is not implemented -- see the comment at meerkat.py's assert.
    if params.microstep_frames is not None:
        kwargs["microsteps"] = [1, 1, params.microstep_frames]
    elif params.reconstruct_every_nth_frame is not None:
        kwargs["microsteps"] = [1, 1, 1.0 / params.reconstruct_every_nth_frame]

    if params.unit_cell_transform is not None:
        kwargs["unit_cell_transform_matrix"] = np.asarray(
            params.unit_cell_transform, dtype=float
        ).reshape(3, 3)

    if params.mask is not None:
        import fabio

        kwargs["measured_pixels"] = fabio.open(params.mask).data >= 0

    if params.scales is not None:
        kwargs["scale"] = np.loadtxt(params.scales)

    reconstruct_data(**kwargs)
