# meerkat
A python library for performing reciprocal space reconstruction from single crystal x-ray measurements. 

## Installation
The package can be installed using pip:

    pip install meerkat

## Usage

The reciprocal space reconstruction is based on the orientation matrix determined by [XDS](“xds.mpimf-heidelberg.mpg.de”). Thus, in order to run the reconstruction, in addition to the diffraction frames, `meerkat` requires either `XPARM.XDS` or `GXPARM.XDS`.

The reconstruction can be run using the following python script:

```
from meerkat import reconstruct_data
import fabio

#load the pixels which should be used in reconstruction
measured_pixels=fabio.open('BKGINIT.cbf').data>=0

#reconstruct dataset
reconstruct_data(filename_template='../frames/PdCPTN01002_%05i.cbf,
        first_image=1,
        last_image=3600,
        reconstruct_in_orthonormal_basis=False,
        maxind=[4,5,16], #the reconstruction will be made for h=-4...4, k=-5...5, l=-16...16
        number_of_pixels=[801, 801, 801], #The resulting size of the array. Controls the step size
        polarization_factor=0.5,
        path_to_XPARM='/home/arkadiy/work/data/PdCPTN01002/xds',
        measured_pixels=measured_pixels,
        output_filename='reconstruction.h5',
        all_in_memory=False,
        size_of_cache=100,
        override=True)
```

## Output
The result is saved as an [hdf5](“http://www.hdfgroup.org/HDF5/”) file. The reconstruction is held in two datasets: `rebinned_data` and `number_of_pixels_rebinned`, the former is a corrected sum of intensities of reconstructed pixels, while the latter counts how many pixels were reconstructed. The scattering intensity can be obtained by dividing the two: `rebinned_data[i,j,k]/number_of_pixels_rebinned[i,j,k]`.

In addition to the two datasets, the reconstruction file contains parameters of the reconstruction `maxind`,  `number_of_pixels`, calculated `step_size`, and information from XDS files: `unit_cell`, `space_group_nr` and `metric_tensor`.

## Reconstruction coordinates

By default the reconstruction is performed in crystallographic coordinates. Such reconstructions can be easily symmetry-averaged. Also the numerical analysis of diffuse scattering is more straightforwardly performed in crystallographic coordinates (for example the program [Yell](“https://github.com/YellProgram/Yell/”) uses such coordinates).

The downside of the crystallographic coordinates is that they are in general not orthorombic, which makes the reconstructions in such coordinates slightly more complicated to plot. If the reconstructions are required in orthonormal coordinates, this can be achieved by setting: 

     reconstruct_in_orthonormal_basis=True

If the reconstruction is performed in orthonormal basis, the new basis $a_o^*,b_o^*,c_o^*$ is calculated from the crystal $a^*,b^*,c^*$ vectors. In the new basis $a_o^*=a_o^*/|a_o^*|$, the $b_o^*$ is in the plane spawned by $a^*$ and $b^*$, and $c_o^*$ is orthogonal to  $a_o^*$ and $b_o^*$.

## Memory usage
The three dimensional arrays containing the reconstructed reciprocal space are typically large (~10Gb). We appreciate that not all computers might have enough operating memory to hold this datasets. Thanks to the `hdf5`, it is possible to use large arrays hosted on hard drive. In such case, only a small portion of the array will be cached in the operating memory. In order to turn on caching set parameter `all_in_memory` to `False` and define the size of the memory for cache.

    all_in_memory = False

Such scheme is approximately three times slower, than holding all datasets in memory.
