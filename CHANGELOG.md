# Changelog

## 0.4.0

The first release with a command line, a config file format, and provenance.
Reconstructions made with 0.3.8 are reproduced bit for bit unless noted below.

### Added

- **A command line.** `meerkat reconstruct`, `info`, `dump-config`,
  `improve-orientation`, `transform-xparm`. Every parameter can come from a `.mrk`
  config file, from flags, or both — flags win. `--dry-run` prints the resolved
  configuration without computing anything.
- **The `.mrk` config file**, following Meerkat2's keyword format. Comments with `#`
  or `!`. Unknown keywords report the line and suggest a correction; duplicate
  keywords are an error rather than silently last-wins.
- **Provenance.** Every reconstruction records the config you wrote, a resolved
  re-runnable copy, the command line, the XPARM contents and hash, package versions,
  host, and which frames were used — in a `meerkat_provenance` group, invisible to
  Yell. `meerkat dump-config out.h5` recovers a config that reproduces the file bit
  for bit. A `.mrk` sidecar is written alongside the output.
- **`meerkat improve-orientation`**, absorbing `improve_orientation_v0.82.py`, with
  three new features:
  - `--dr-schedule 0.25,0.15,0.08,0.04` — sequential refinement, tightening as it
    goes, re-selecting spots from the full list each pass. Lets refinement start from
    a geometry too poor for a single tight cut: with the beam centre 6 px out, a
    single `--dr 0.04` pass finds zero spots, while the schedule recovers the centre
    to about a pixel.
  - `--instrument-xparm` — import detector geometry refined on a standard sample.
    Crystal and scan parameters are never imported; a wavelength mismatch is refused.
  - `--cell-restraints a,a,c,90,90,120` — restrain the cell by symmetry.
    `--restraint-weight` is normalized against the spot count, so 1 means "trust the
    symmetry about as much as the data" regardless of how many spots there are.
  - Reports parameters the data cannot constrain, and refuses a runaway cell.
- **`meerkat transform-xparm`**, absorbing `xparm_transform_0.21.py`. Refuses singular
  transforms, and volume-changing ones without `--force`.
- **`RECONSTRUCT_EVERY_NTH_FRAME`** and **`MICROSTEP_FRAMES`** as first-class
  keywords.
- **`meerkat.xds`** — a numpy-only subpackage (XPARM/XDS.INP/SPOT.XDS I/O and
  diffraction geometry) that imports without fabio or h5py, so GUI tooling can depend
  on it without inheriting an image stack.
- `meerkat.__version__`.
- A test suite (there was none) and CI across python 3.9–3.13 on linux and macOS.

### Fixed

- **`maxind` was float32 and is float64 again.** It was `np.float_` — which *is*
  float64 — until commit 7c19787 swapped it while removing aliases numpy 2 dropped.
  Zero data voxels change (`step_size_inv` was already float64 by promotion, and
  float32-representable limits like 7, 5, 1.5 were unaffected either way), but
  `lower_limits` is written to the output and Yell reads it to place the grid: a
  `maxind` of 7.3 was stored as −7.3000001907. Now exact.
- **Frame decimation never worked.** `microsteps=[1,1,0.1]` ("every tenth frame")
  raised `IndexError`, because `1/0.1` is a float and the frame numbers became floats.
- **`write_xparm` hardcoded a Pilatus 6M detector** in the copy used by
  `xparm_transform`, which is why that script printed "THIS SCRIPT IS BROKEN / PIXEL
  SIZE IS WRONG" on every run. There is now one canonical implementation.
- **`write_xparm` passed a 1-element array to `%f`** for `x_center`, which
  `numpy >= 1.25` deprecates and will make an error. It only worked when `xycenter`
  happened to be refined first.
- **`improve-orientation` selected spots using uncorrected coordinates**, applying the
  XDS convention offset and the X/Y-CORRECTIONS tables only afterwards — so the cut
  was biased by a pixel and half an oscillation relative to the objective it fed.
- `create_h5py_with_large_cache` now accepts `pathlib.Path`.

### Changed

- `setup.py` → `pyproject.toml`. The old `setup.cfg` used `description-file`, which
  was never a real setuptools key, so PyPI showed no description.
- `importing meerkat` no longer eagerly imports fabio and h5py. All the 0.3.x names
  still resolve.
- `meerkat.det2lab_xds` moved to `meerkat.xds.geometry`. The old path works and warns;
  it goes away in 0.5.
- The accidental 0.3.x re-exports (`meerkat.np`, `meerkat.h5py`, …) warn on access.
  They go away in 0.5.
- README corrected: it described the pre-0.3.7 two-dataset output as the default and
  claimed `maxind` and `number_of_pixels` were stored, which they never were.

### Known limitations

- x/y (sub-pixel) microstepping is not implemented. It never has been — an assert has
  always blocked it. `meerkat/meerkat.py` marks where it would go and what it would
  cost.
- Asymmetric grids parse and resolve, but reconstruction still requires symmetric
  limits.
- Reconstruction is single-threaded.
