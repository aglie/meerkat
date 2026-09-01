# meerkat

A python library and command-line tool for reciprocal space reconstruction from
single crystal x-ray measurements.

## Installation

```
pip install meerkat
```

`meerkat` itself installs numpy, h5py and fabio and nothing else, so it goes onto
a headless compute node without dragging a GUI stack behind it. The heavier
pieces are extras:

```
pip install meerkat[refine]     # scipy, for improve-orientation
pip install meerkat[viewer]     # PyQt5 and PyOpenGL, for the Ewald sphere viewer
pip install meerkat[cbf]        # cbf, to read XDS's X/Y-CORRECTIONS tables
pip install meerkat[all]        # all of the above
```

A subcommand whose extra is missing says which one to install; nothing else is
affected.

On Windows we recommend a virtual environment such as
[anaconda](https://www.anaconda.com/), which simplifies the python installation.

#### Note for anaconda users

Some anaconda distributions fail on `pip install meerkat` while trying to compile
`h5py`. Anaconda ships `h5py` already, so this works instead:

```
pip install meerkat --no-deps
pip install fabio
```

## The command line

Everything is driven by one `meerkat` command with subcommands:

| Command | What it does |
| --- | --- |
| [`meerkat reconstruct`](#reconstruction) | reconstruct reciprocal space from a set of frames |
| [`meerkat improve-orientation`](#refining-the-geometry) | refine experimental geometry against indexed spots |
| [`meerkat view`](#looking-at-the-ewald-sphere) | show the measured spots in reciprocal space |
| [`meerkat transform-xparm`](#re-indexing-a-cell) | re-index a crystal to a different cell setting |
| [`meerkat info`](#looking-at-files) | summarize an XPARM or a reconstruction |
| [`meerkat dump-config`](#looking-at-files) | recover the config that made a reconstruction |
| [`meerkat dials-to-xds`](#starting-from-dials) | convert a DIALS experiment to XDS.INP/XPARM.XDS/SPOT.XDS |

`meerkat COMMAND --help` documents each one in full. `meerkat --version` prints the
version.

Reconstruction is based on the orientation matrix determined by
[XDS](https://xds.mpimf-heidelberg.mpg.de), so in addition to the diffraction frames
`meerkat` needs an `XPARM.XDS` or `GXPARM.XDS`. If your data was processed with DIALS
instead, start at [Starting from DIALS](#starting-from-dials).

## Reconstruction

There are three ways to drive `meerkat reconstruct`, and they all run the same code.

### A parameter file

The recommended way. The file stays next to your data and records what you did.

```
# reconstruction.mrk
DATA_FILE_TEMPLATE  ../frames/PdCPTN01002_%05i.cbf
XPARM_FILE          /home/arkadiy/work/data/PdCPTN01002/xds
FIRST_FRAME         1
LAST_FRAME          3600

NUMBER_OF_PIXELS    801 801 801
LOWER_LIMITS        -4 -5 -16
SYMMETRIC_LIMITS            # so h = -4..4, k = -5..5, l = -16..16

POLARIZATION_FACTOR 0.5     # 0.5 for a laboratory source, 1 for a synchrotron
OUTPUT_FILENAME     reconstruction.h5
```

```
meerkat reconstruct reconstruction.mrk
```

Comments start with `#` or `!`. An unknown keyword reports the line and suggests a
correction; a repeated keyword is an error rather than silently last-wins.

### Command-line flags

Every keyword is also a flag, and flags override the file. Useful for scripting and
for trying one thing without editing anything:

```
meerkat reconstruct reconstruction.mrk --polarization-factor 1 --output-filename test.h5
```

A config file is not required — a run can be driven entirely by flags:

```
meerkat reconstruct --data-file-template '../frames/img_%05i.cbf' \
                    --xparm-file xds/ --first-frame 1 --last-frame 3600 \
                    --number-of-pixels 501 501 501 \
                    --lower-limits -4 -5 -16 --symmetric-limits \
                    --output-filename reconstruction.h5
```

Check what a run *would* do, without doing it:

```
meerkat reconstruct reconstruction.mrk --dry-run
```

`--dump-config FILE` writes the fully resolved configuration out (`-` for stdout), so
an exploratory command line can be turned back into a parameter file.

### Every parameter

| Keyword | Flag | Meaning |
| --- | --- | --- |
| `DATA_FILE_TEMPLATE` | `--data-file-template` | printf-style template for the frames, e.g. `frames/img_%05i.cbf` |
| `XPARM_FILE` | `--xparm-file` | `XPARM.XDS`/`GXPARM.XDS`, or a directory holding one (default `.`) |
| `FIRST_FRAME` | `--first-frame` | first frame number |
| `LAST_FRAME` | `--last-frame` | last frame number |
| `MASK` | `--mask` | image whose negative pixels mark untrusted detector areas (default: derive from the first frame) |
| `SCALES` | `--scales` | text file with one scale factor per frame |
| `NUMBER_OF_PIXELS` | `--number-of-pixels` | output grid dimensions |
| `LOWER_LIMITS` | `--lower-limits` | hkl of voxel `[0,0,0]` |
| `UPPER_LIMITS` | `--upper-limits` | hkl of the last voxel |
| `STEP_SIZES` | `--step-sizes` | grid step in r.l.u. |
| `SYMMETRIC_LIMITS` | `--symmetric-limits` | mirror the given limits about the origin |
| `POLARIZATION_FACTOR` | `--polarization-factor` | 1 for a synchrotron, 0.5 for an unpolarized laboratory source |
| `POLARIZATION_PLANE_NORMAL` | `--polarization-plane-normal` | normal of the polarization plane (default `0 1 0`) |
| `MEDIUM` | `--medium` | medium between crystal and detector: `Air` or `Helium` |
| `UNIT_CELL_TRANSFORM` | `--unit-cell-transform` | 3×3 matrix (row-major) applied to the cell vectors before reconstruction |
| `RECONSTRUCT_IN_ORTHONORMAL_BASIS` | `--reconstruct-in-orthonormal-basis` | see [Reconstruction coordinates](#reconstruction-coordinates) |
| `OUTPUT_FILENAME` | `--output-filename` | output HDF5 file (default `reconstruction.h5`) |
| `OUTPUT_FORMAT` | `--output-format` | `YELL_1.0` or `YELL_0.9`, see [Output](#output) |
| `OVERWRITE` | `--overwrite` | overwrite the output file if it exists |
| `ALL_IN_MEMORY` | `--all-in-memory` | hold the whole grid in RAM, see [Memory usage](#memory-usage) |
| `SIZE_OF_CACHE` | `--size-of-cache` | HDF5 chunk cache size in MB (default 100) |
| `MICROSTEP_FRAMES` | `--microstep-frames` | see [Sampling](#sampling) |
| `RECONSTRUCT_EVERY_NTH_FRAME` | `--reconstruct-every-nth-frame` | see [Sampling](#sampling) |

Only the grid is subtle: give any two of `NUMBER_OF_PIXELS`, `LOWER_LIMITS`/
`UPPER_LIMITS` and `STEP_SIZES` and the third is derived. `SYMMETRIC_LIMITS` lets you
give one set of limits and have the other mirrored.

Three more flags control what gets recorded: `--no-provenance`, `--no-sidecar` and
`--checksum-frames` — see [Provenance](#provenance).

## Refining the geometry

`meerkat improve-orientation` refines the experimental geometry against the spots XDS
already indexed, writing a better `XPARM`. Diffuse scattering is unforgiving about
orientation, and an XDS solution good enough for Bragg integration is often not good
enough for a reconstruction.

```
meerkat improve-orientation --xds-folder xds/ -o xds/MXPARM.XDS
```

It reads `SPOT.XDS` and `XPARM.XDS` from `--xds-folder` (override either with
`--spot-file` and `--input-xparm`), and applies XDS's `X-CORRECTIONS.cbf` /
`Y-CORRECTIONS.cbf` if they are there (`--no-xds-corrections` to ignore them).

**Which spots to use.** `--imin`/`--imax` cut on intensity, `--frame-min`/
`--frame-max` on frame number, `--filter-fcc` keeps only reflections allowed by an
F-centred lattice, and `--dr` keeps spots within that distance in hkl of an integer
reflection.

**Refining in stages.** A single tight `--dr` needs a starting geometry good enough to
find spots at all. `--dr-schedule` runs sequential passes, tightening as it goes and
re-selecting from the full spot list each time, so a better fit brings more spots in:

```
meerkat improve-orientation --xds-folder xds/ --dr-schedule 0.25,0.15,0.08,0.04
```

With the beam centre 6 px out, a single `--dr 0.04` pass finds zero spots while the
schedule recovers the centre to about a pixel.

**What to refine.** `--refine` takes a space-separated subset of `distance xycenter
beam cell axis`. `distance` is excluded by default — it is nearly degenerate with the
overall cell scale. Parameters the data cannot constrain are reported, and a runaway
cell is refused.

**Importing instrument geometry.** `--instrument-xparm` takes detector position, beam
centre, pixel size and rotation axis from an XPARM refined on a standard sample.
Crystal and scan parameters are never imported, and a wavelength mismatch is refused
unless you pass `--allow-wavelength-mismatch`.

**Restraining the cell by symmetry.** `--cell-restraints` takes six comma-separated
slots for `a,b,c,alpha,beta,gamma`, where a number pins a value, a repeated label ties
parameters together, and `*` leaves one free — hexagonal is `a,a,c,90,90,120`, cubic
is `a,a,a,90,90,90`. `--restraint-weight` is normalized against the spot count, so 1
trusts the symmetry about as much as the data regardless of how many spots there are;
10 mostly imposes it, 0.1 is a gentle nudge.

`--print-hkl` prints the indexed hkl of the selected spots.

This command needs scipy (`pip install meerkat[refine]`); without it the subcommand is
not offered.

## Looking at the Ewald sphere

`meerkat view` opens the spots XDS found as a point cloud in reciprocal space,
with every instrument parameter editable while you watch:

```
meerkat view xds/                   # XPARM.XDS and SPOT.XDS from a folder
meerkat view xds/XPARM.XDS          # or name the file
meerkat view                        # or start empty and use File > Open
```

Drag to rotate, wheel to zoom. Switch to selection mode to drag a rectangle over
part of the cloud, then select or unselect what is inside it; *File > Save
selected* writes those spots out in SPOT.XDS format, which is how you cut a
second lattice or a bad run of frames out of the list before refining against it.
The *Bragg peak filter* narrows the cloud by intensity and by frame number.

The point of the parameter panel is that a wrong beam centre, distance or
rotation axis has a recognisable signature: the shells go lopsided, or the same
reflection measured on opposite sides of the rotation fails to land in the same
place. Nudge the parameter until it looks right, and take that as the starting
point for `meerkat improve-orientation`.

A folder is opened through `XPARM.XDS` in preference to `XDS.INP`. XDS.INP works,
but it records neither the phi origin nor the orientation matrix, so the whole
cloud sits half an oscillation away from where the data actually is; the status
bar says so when that is what you have opened.

This is the one command that needs a GUI: `pip install meerkat[viewer]` (or,
under conda, `conda install pyqt pyopengl`). It is PyQt5, and it needs a real
OpenGL context -- so a local display, or ssh with X forwarding.

## Re-indexing a cell

`meerkat transform-xparm` applies a 3×3 transformation to the cell vectors in an
XPARM — for moving to a different setting, or a supercell.

```
meerkat transform-xparm -i XPARM.XDS -o XPARM_new.XDS -t 0 1 0  1 0 0  0 0 -1
```

The matrix (`-t`/`--transform`) is row-major and the new cell vectors are
`T · (a, b, c)`, so the example above swaps **a** and **b** and flips **c**. A singular
transform is refused, and one that changes the cell volume is refused without
`--force`.

## Looking at files

`meerkat info` summarizes an `XPARM.XDS`/`GXPARM.XDS`, a directory holding one, or a
reconstruction `.h5`:

```
$ meerkat info xds/XPARM.XDS
XPARM: xds/XPARM.XDS
  cell            : 8.2287 8.2299 11.0122  90.013 90.025 59.974
  space group     : 1
  wavelength      : 0.774899 A
  detector        : 2463 x 2527 px @ 0.172 x 0.172 mm
  distance        : 200.058 mm
  beam centre     : 1214.77, 1261.04 px
  oscillation     : 0.1 deg/frame, starting at 0.0 deg on frame 2
  rotation axis   : 0.999999 -0.000949 0.000863
  cell vectors (rows are a, b, c):
    a :  -2.497232  -3.449634  -7.040996
    b :  -0.802756   4.595787  -6.779770
    c :  10.471534  -2.116248  -2.671483
```

`meerkat dump-config` recovers the parameter file that made a reconstruction, read
back from the provenance stored inside it:

```
meerkat dump-config reconstruction.h5              # a config that reproduces it
meerkat dump-config reconstruction.h5 --original   # the file you originally wrote
meerkat dump-config reconstruction.h5 --all        # command line, XPARM, versions, host, inputs
```

## Starting from DIALS

If your Bragg data was processed with [DIALS](https://dials.github.io/) rather than
XDS directly, `meerkat dials-to-xds` converts the result into the
XDS.INP/XPARM.XDS/SPOT.XDS that everything else here (reconstruction,
`improve-orientation`) needs:

```
meerkat dials-to-xds experiments.expt reflections.refl -o xds_from_dials/
```

`-o`/`--output-dir` is where `XDS.INP`, `XPARM.XDS` and `SPOT.XDS` are written. For a
multi-experiment `.expt`/`.refl`, pick one with `--experiment-index`.

Needs `dials` and `dxtbx` importable (not required for any other `meerkat`
command). This is a from-scratch converter, not a wrapper around `dials.export
format=xds` — that command has two real geometry bugs, found and fixed while
processing real beamline data:

* `INCIDENT_BEAM_DIRECTION` written as the wavevector (magnitude `1/wavelength`)
  instead of a unit vector.
* `POLARIZATION_PLANE_NORMAL` written straight from DIALS' native frame, without
  the rotation applied to every other lab-frame direction (detector axes,
  rotation axis, unit cell) that aligns them to XDS's canonical frame — wrong for
  any detector that isn't already axis-aligned in dxtbx's own frame.

A fix for both has been submitted upstream to `dxtbx`; until it lands (and for
any dxtbx version already released), use `meerkat dials-to-xds`. See
`meerkat.dials.to_xds` for the full details and the geometry conventions used.

## Output

The result is an [hdf5](https://www.hdfgroup.org/solutions/hdf5/) file in Yell 1.0
format. The scattering intensity is in the `data` dataset. Voxels the Ewald sphere
never swept are `NaN` — that is how "no data here" is signalled, so mind it when
plotting.

Alongside `data` the file carries the grid (`lower_limits`, `step_sizes`) and
information from XDS (`unit_cell`, `space_group_nr`, `metric_tensor`).

Setting `OUTPUT_FORMAT YELL_0.9` writes the older layout instead: two datasets,
`rebinned_data` (the corrected sum of pixel intensities) and
`number_of_pixels_rebinned` (how many pixels contributed), whose ratio is the
intensity.

### Provenance

Every reconstruction records how it was made, in a `meerkat_provenance` group: the
config file you wrote, a resolved and re-runnable copy of it, the command line, the
contents and sha256 of the XPARM, package versions, and which frames were used. A
`.mrk` sidecar is also written next to the output.

So a stray `.h5` can always explain itself:

```
meerkat dump-config reconstruction.h5           # a config that reproduces it
meerkat dump-config reconstruction.h5 --all     # everything recorded
```

Re-running the recovered config reproduces the data bit for bit. The group is
invisible to Yell, which opens `data` by name.

`--no-provenance` turns the recording off and `--no-sidecar` suppresses the `.mrk`
next to the output. `--checksum-frames` additionally hashes every input frame; it is
off by default, since hashing 100 GB of frames to write a 1 GB output is rarely worth
it.

## Reconstruction coordinates

By default the reconstruction is in crystallographic coordinates. These are easy to
symmetry-average, and numerical analysis of diffuse scattering is more
straightforward in them — [Yell](https://github.com/YellProgram/Yell/) uses these
coordinates.

The downside is that crystallographic coordinates are generally not orthogonal, which
makes plotting slightly more involved. For an orthonormal reconstruction:

```
RECONSTRUCT_IN_ORTHONORMAL_BASIS
```

The new basis a\*', b\*', c\*' is derived from the crystal's a\*, b\*, c\*: a'\* is
parallel to a\*, b'\* lies in the a\*–b\* plane, and c'\* is orthogonal to both.

## Sampling

`MICROSTEP_FRAMES n` subdivides each frame's rotation into `n` sub-steps, spread
symmetrically about the frame centre. This fills the gaps between frames when the
oscillation is coarse, at `n` times the projection cost.

`RECONSTRUCT_EVERY_NTH_FRAME n` uses only every `n`th frame — handy for a quick look
at a large scan.

## Memory usage

Reconstructed arrays are large. An 801³ grid needs about 4.1 GB while it is being
built (2.06 GB of float32 data plus 2.06 GB of counts), so it fits in RAM on a 16 GB
machine but not with much to spare.

Set `ALL_IN_MEMORY` to hold it all in RAM. Without it, the array lives on disk via
hdf5 and only a cache is held in memory:

```
SIZE_OF_CACHE 500     # MB
```

Out-of-core is roughly three times slower, and is what you want for grids beyond
about 900³.

## The old way: reconstructing from python

The 0.3.x API still works unchanged, and remains the way to drive a reconstruction
from inside a script or a notebook. It runs the same engine as `meerkat reconstruct`,
but it writes no provenance and does not read `.mrk` files.

```python
from meerkat import reconstruct_data

reconstruct_data(filename_template='../frames/PdCPTN01002_%05i.cbf',
                 first_image=1,
                 last_image=3600,
                 maxind=[4, 5, 16],          # h = -4..4, k = -5..5, l = -16..16
                 number_of_pixels=[801, 801, 801],
                 polarization_factor=0.5,
                 path_to_XPARM='/home/arkadiy/work/data/PdCPTN01002/xds',
                 output_filename='reconstruction.h5',
                 all_in_memory=False,
                 scale=None)  # per-frame scale factors, e.g. for uneven illumination
```

The other 0.3.x names — `read_XPARM`, `write_xparm`, `det2lab_xds`, `cell2vecs`,
`vecs2cell`, `create_h5py_with_large_cache` and the rest — still resolve from the
`meerkat` namespace too.

XDS file formats and diffraction geometry now also live in `meerkat.xds`, which
imports with numpy alone (no fabio, no h5py) so that a GUI can depend on it without
inheriting an image stack:

```python
from meerkat.xds import read_xparm, write_xparm, read_spot_xds, params_from_xds_inp
```

`meerkat.det2lab_xds` has moved to `meerkat.xds.geometry`. The old path still works
and warns; it goes away in 0.5, along with the accidental 0.3.x re-exports
(`meerkat.np`, `meerkat.h5py`, …).
