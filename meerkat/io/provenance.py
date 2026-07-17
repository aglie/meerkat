"""Record how a reconstruction was made, inside the reconstruction.

The problem this solves: a .h5 on a colleague's disk, three years old, and nobody can
say which frames, which XPARM, or which corrections produced it. meerkat 0.3.x wrote
seven metadata keys, none of which were reconstruction parameters.

Everything lands in ONE group, /meerkat_provenance/. That is deliberate: Yell reads
this format, so the only question that matters is whether it tolerates an extra root
member. Checked before writing any of this -- Yell's entire HDF5 read is
`file.openDataSet("data")` (Yell/src/IntensityMap.cpp:20), by name, and there is no
root enumeration anywhere in Yell/src or FTL/src. Confined to a group, provenance
cannot collide with a name Yell wants, and cannot be reached by an enumeration Yell
does not perform.

Written by reopening the finished file rather than by threading a parameter through
the engine: provenance is a property of a *run*, the reconstruction maths does not
care, and this keeps the 141-line engine untouched.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import shlex
import socket
import sys
from datetime import datetime, timezone

import numpy as np

__all__ = ["PROVENANCE_GROUP", "read_provenance", "write_provenance"]

PROVENANCE_GROUP = "meerkat_provenance"


def _sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        from .._version import __version__

        out["meerkat"] = __version__
    except Exception:
        pass
    for name in ("numpy", "h5py", "fabio", "scipy"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", getattr(module, "version", "?"))
        except Exception:
            pass
    return out


def frames_used(params):
    """The frame numbers actually reconstructed.

    Cheap forensics: with the template and this list you can find every input file,
    which is what people actually reach for. Hashing 100 GB of frames to write a 1 GB
    output is not a trade anyone wants, so per-frame checksums are opt-in.
    """
    step = params.reconstruct_every_nth_frame or 1
    return np.arange(params.first_frame, params.last_frame + 1, step, dtype=np.int64)


def write_provenance(h5_path, params, config_text=None, argv=None, checksum_frames=False):
    """Add /meerkat_provenance/ to a finished reconstruction.

    config_resolved is the load-bearing entry: it is a complete, re-runnable .mrk with
    the grid written out explicitly, so reproducing the run needs no inference. The
    round trip is tested end to end (see tests/test_provenance.py).
    """
    import h5py

    from ..config import dump_mrk

    entries = {
        # The verbatim file the user wrote, comments and all. Empty for a pure-CLI run.
        "config_text": config_text or "",
        # The one that matters: fully resolved, re-runnable.
        "config_resolved": dump_mrk(params),
        "command_line": shlex.join(argv if argv is not None else sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "versions": json.dumps(_versions(), indent=2, sort_keys=True),
    }

    try:
        entries["username"] = getpass.getuser()
    except Exception:
        # getuser() raises when there is no passwd entry, e.g. in some containers.
        entries["username"] = "unknown"

    xparm_path = _resolve_xparm(params.xparm_file)
    if xparm_path is not None:
        entries["xparm_path"] = os.path.abspath(xparm_path)
        entries["xparm_sha256"] = _sha256(xparm_path)
        with open(xparm_path) as f:
            # Verbatim, so the geometry is recoverable even if the file moves or is
            # re-refined. An XPARM is under 1 KB; there is no reason not to.
            entries["xparm_text"] = f.read()

    inputs = []
    for label in ("mask", "scales"):
        path = getattr(params, label, None)
        if path and os.path.exists(path):
            inputs.append({"role": label, "path": os.path.abspath(path), "sha256": _sha256(path)})

    frames = frames_used(params)

    if checksum_frames:
        for frame in frames:
            name = params.data_file_template % frame
            if os.path.exists(name):
                inputs.append(
                    {"role": "frame", "path": os.path.abspath(name), "sha256": _sha256(name)}
                )

    entries["input_files"] = json.dumps(inputs, indent=2)

    with h5py.File(h5_path, "a") as f:
        if PROVENANCE_GROUP in f:
            del f[PROVENANCE_GROUP]
        group = f.create_group(PROVENANCE_GROUP)
        for key, value in entries.items():
            # Variable-length UTF-8 datasets, not attributes: HDF5 attributes have a
            # 64 KB ceiling in the compact layout, and a config plus an XPARM plus a
            # frame checksum list will find it.
            group.create_dataset(key, data=value, dtype=h5py.string_dtype())
        group.create_dataset("frames_used", data=frames)


def _resolve_xparm(path):
    """Mirror read_xparm's GXPARM-over-XPARM preference, so we record what was used."""
    if path is None or not os.path.exists(path):
        return None
    if os.path.isdir(path):
        for candidate in ("GXPARM.XDS", "XPARM.XDS"):
            full = os.path.join(path, candidate)
            if os.path.isfile(full):
                return full
        return None
    return path


def read_provenance(h5_path):
    """Read /meerkat_provenance/ back as a dict. Returns {} if absent."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        if PROVENANCE_GROUP not in f:
            return {}
        group = f[PROVENANCE_GROUP]
        out = {}
        for key in group:
            value = group[key][()]
            out[key] = value.decode() if isinstance(value, bytes) else value
        return out
