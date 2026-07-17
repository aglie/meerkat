"""
A python library for performing reciprocal space reconstruction from single crystal
x-ray measurements.
"""

import importlib

from ._version import __version__ as __version__

# Lazy attribute access (PEP 562). The point is that `import meerkat.xds.xparm` must
# not drag in fabio and h5py: importing any submodule executes this file first, so a
# module-level `from .meerkat import *` here would defeat the whole meerkat.xds split
# no matter how clean that subpackage is. meerkat-ewald depends on that split.
#
# Everything below still resolves exactly as it did in 0.3.x, including
# `from meerkat import *` -- it just resolves on first touch instead of on import.

_ORIGIN = {
    # numpy-only. Importing these pulls in nothing heavy.
    "det2lab_xds": "meerkat.xds.geometry:det2lab_xds",
    "rotvec2mat": "meerkat.xds.geometry:rotvec2mat",
    "read_xparm": "meerkat.xds.xparm:read_xparm",
    "write_xparm": "meerkat.xds.xparm:write_xparm",
    "vecs2cell": "meerkat.xds.xparm:vecs2cell",
    "cell2vecs": "meerkat.xds.xparm:cell2vecs",
    "read_spot_xds": "meerkat.xds.spot:read_spot_xds",
    "params_from_xds_inp": "meerkat.xds.xds_inp:params_from_xds_inp",
    "INSTRUMENT_KEYS": "meerkat.xds.xparm:INSTRUMENT_KEYS",
    "CRYSTAL_KEYS": "meerkat.xds.xparm:CRYSTAL_KEYS",
    # The 0.3.x spelling. Kept indefinitely: it is what improve_orientation,
    # xparm_transform, the Ewald GUI, and every user script call.
    "read_XPARM": "meerkat.xds.xparm:read_xparm",
    # Pulls in fabio + h5py.
    "reconstruct_data": "meerkat.meerkat:reconstruct_data",
    "accumulate_intensity": "meerkat.meerkat:accumulate_intensity",
    "correction_coefficients": "meerkat.meerkat:correction_coefficients",
    "air_absorption_coefficient": "meerkat.meerkat:air_absorption_coefficient",
    "cov2corr": "meerkat.meerkat:cov2corr",
    "create_h5py_with_large_cache": "meerkat.meerkat:create_h5py_with_large_cache",
    "r_get_numbers": "meerkat.meerkat:r_get_numbers",
}

# `from .meerkat import *` used to leak the module's imports as package attributes.
# Nobody should be reaching for meerkat.np, but it worked in 0.3.8 and this is a
# published package, so keep it resolving -- loudly -- rather than break a script we
# cannot see. Remove in 0.5.
_DEPRECATED_REEXPORTS = {
    "np": "numpy",
    "numpy": "numpy",
    "re": "re",
    "os": "os",
    "h5py": "h5py",
    "fabio": "fabio",
}

__all__ = sorted(_ORIGIN)


def __getattr__(name):
    target = _ORIGIN.get(name)
    if target is not None:
        module_name, _, attribute = target.partition(":")
        value = getattr(importlib.import_module(module_name), attribute)
        globals()[name] = value  # resolve once, then it is a normal global
        return value

    module_name = _DEPRECATED_REEXPORTS.get(name)
    if module_name is not None:
        import warnings

        warnings.warn(
            f"meerkat.{name} is an accidental re-export from meerkat 0.3.x and will "
            f"be removed in 0.5. Import {module_name!r} directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        value = importlib.import_module(module_name)
        globals()[name] = value
        return value

    if name == "norm":
        import warnings

        warnings.warn(
            "meerkat.norm is an accidental re-export from meerkat 0.3.x and will be "
            "removed in 0.5. Use numpy.linalg.norm instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from numpy.linalg import norm

        globals()["norm"] = norm
        return norm

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | set(globals()))
