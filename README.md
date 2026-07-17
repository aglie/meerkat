# meerkat

A python library and command-line tool for reciprocal space reconstruction from
single crystal x-ray measurements.

## Installation

```
pip install meerkat
```

Refining orientation matrices additionally needs scipy:

```
pip install meerkat[refine]
```

On Windows we recommend a virtual environment such as
[anaconda](https://www.anaconda.com/), which simplifies the python installation.

#### Note for anaconda users

Some anaconda distributions fail on `pip install meerkat` while trying to compile
`h5py`. Anaconda ships `h5py` already, so this works instead:

```
pip install meerkat --no-deps
pip install fabio
```

## Usage

Reconstruction is based on the orientation matrix determined by
[XDS](https://xds.mpimf-heidelberg.mpg.de), so in addition to the diffraction frames
`meerkat` needs an `XPARM.XDS` or `GXPARM.XDS`.

There are three ways to drive it, and they all run the same code.

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

Comments start with `#` or `!`. `meerkat reconstruct --help` lists every keyword.

### Command-line flags

Every keyword is also a flag, and flags override the file. Useful for scripting and
for trying one thing without editing anything:

```
meerkat reconstruct reconstruction.mrk --polarization-factor 1 --output-filename test.h5
```

Check what a run *would* do, without doing it:

```
meerkat reconstruct reconstruction.mrk --dry-run
```

### From python

The 0.3.x API still works unchanged:

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

## Other commands

```
meerkat info FILE                  # summarize an XPARM or a reconstruction
meerkat dump-config OUTPUT.h5      # recover the config that made a reconstruction
meerkat improve-orientation ...    # refine geometry against indexed spots
meerkat transform-xparm ...        # re-index a crystal to a different cell setting
```

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
