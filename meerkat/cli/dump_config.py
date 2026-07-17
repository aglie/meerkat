"""`meerkat dump-config` -- recover the config that made a reconstruction.

The point of the provenance record: given only a .h5, get back a .mrk you can run.
"""

from __future__ import annotations

import json

__all__ = ["add_arguments", "run"]


def add_arguments(parser):
    parser.add_argument("file", help="a reconstruction .h5 written by meerkat >= 0.4")
    parser.add_argument(
        "--original",
        action="store_true",
        help="print the config file as the user originally wrote it, comments and all, "
        "instead of the resolved one. Empty if the run was driven purely by flags.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="print everything recorded: command line, XPARM, versions, host, inputs",
    )
    return parser


def run(args) -> int:
    from ..io import read_provenance

    record = read_provenance(args.file)
    if not record:
        raise SystemExit(
            f"error: {args.file} carries no provenance.\n"
            "  It was written by meerkat < 0.4, by another program, or with "
            "--no-provenance."
        )

    if args.all:
        return _print_everything(record)

    if args.original:
        text = record.get("config_text", "")
        if not text.strip():
            raise SystemExit(
                "error: this run was driven entirely by command-line flags, so there "
                "is no original config file.\n"
                "  Use `meerkat dump-config` without --original for the resolved one, "
                "or --all to see the command line."
            )
        print(text, end="")
        return 0

    print(record.get("config_resolved", ""), end="")
    return 0


def _print_everything(record) -> int:
    def section(title, body):
        if body is None or not str(body).strip():
            return
        print(f"--- {title} " + "-" * max(0, 68 - len(title)))
        print(str(body).rstrip())
        print()

    section("when", record.get("timestamp_utc"))
    section(
        "where",
        f"{record.get('username', '?')}@{record.get('hostname', '?')}:{record.get('cwd', '?')}",
    )
    section("command line", record.get("command_line"))
    section("resolved config (re-runnable)", record.get("config_resolved"))
    section("original config file", record.get("config_text"))

    xparm_path = record.get("xparm_path")
    if xparm_path:
        section("xparm", f"{xparm_path}\nsha256: {record.get('xparm_sha256', '?')}")
        section("xparm contents", record.get("xparm_text"))

    versions = record.get("versions")
    if versions:
        try:
            section(
                "versions",
                "\n".join(f"  {k:10s} {v}" for k, v in json.loads(versions).items()),
            )
        except (json.JSONDecodeError, AttributeError):
            section("versions", versions)

    inputs = record.get("input_files")
    if inputs:
        try:
            parsed = json.loads(inputs)
            if parsed:
                section(
                    "input files",
                    "\n".join(
                        f"  {i['role']:6s} {i['path']}\n         {i['sha256']}"
                        for i in parsed
                    ),
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            section("input files", inputs)

    frames = record.get("frames_used")
    if frames is not None and len(frames):
        section("frames used", f"{len(frames)} frames, {frames[0]} .. {frames[-1]}")

    return 0
