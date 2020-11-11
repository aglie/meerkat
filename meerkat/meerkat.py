import numpy as np
import fabio
import re
import os
from .det2lab_xds import det2lab_xds, rotvec2mat
import h5py
from numpy.linalg import norm


def r_get_numbers(matchgroup, num):
    """A helper function which can be used similarly to fscanf(fid,'%f',num) to extract num arguments from the regex iterator"""
    res = []
    for i in range(num):
        res.append(float(next(matchgroup).group()))
    return np.array(res)


def read_XPARM(path_to_XPARM='.'):
    """Loads the instrumental geometry information from the XPARM.XDS or GXPARM.XDS files at the proposed location"""

    if not os.path.exists(path_to_XPARM):
        raise Exception("path " + path_to_XPARM + "does not exist")

    if os.path.isdir(path_to_XPARM):
        candidate = os.path.join(path_to_XPARM, 'GXPARM.XDS')
        if os.path.isfile(candidate):
            path_to_XPARM = candidate
        else:
            candidate = os.path.join(path_to_XPARM, 'XPARM.XDS')
            if os.path.isfile(candidate):
                path_to_XPARM = candidate
            else:
                raise Exception("files GXPARM.XDS and XPARM.XDS are not found in the folder " + path_to_XPARM)

    with open(path_to_XPARM) as f:
        f.readline()  # skip header
        text = f.read()

    # parse the rest to numbers
    f = re.compile('-?\d+\.?\d*').finditer(text)

    try:
        result = dict(starting_frame=r_get_numbers(f, 1),
                      starting_angle=r_get_numbers(f, 1),
                      oscillation_angle=r_get_numbers(f, 1),
                      rotation_axis=r_get_numbers(f, 3),

                      wavelength=r_get_numbers(f, 1),
                      wavevector=r_get_numbers(f, 3),

                      space_group_nr=r_get_numbers(f, 1),
                      cell=r_get_numbers(f, 6),
                      unit_cell_vectors=np.reshape(r_get_numbers(f, 9), (3, 3)),

                      number_of_detector_segments=r_get_numbers(f, 1),
                      NX=r_get_numbers(f, 1),
                      NY=r_get_numbers(f, 1),
                      pixelsize_x=r_get_numbers(f, 1),
                      pixelsize_y=r_get_numbers(f, 1),

                      x_center=r_get_numbers(f, 1),
                      y_center=r_get_numbers(f, 1),
                      distance_to_detector=r_get_numbers(f, 1),

                      detector_x=r_get_numbers(f, 3),
                      detector_y=r_get_numbers(f, 3),
                      detector_normal=r_get_numbers(f, 3),
                      detector_segment_crossection=r_get_numbers(f, 5),
                      detector_segment_geometry=r_get_numbers(f, 9))

    except StopIteration:
        raise Exception('Wrong format of the XPARM.XDS file')

    # check there is nothing left
    try:
        next(f)
    except StopIteration:
        pass
    else:
        raise Exception('Wrong format of the XPARM.XDS file')

    return result


def cov2corr(inp):
    sigma = np.sqrt(np.diag(inp))
    return sigma, inp / np.outer(sigma, sigma)


def air_absorption_coefficient(medium, wavelength):
    """ 
     The function returns linear absorbtion coefficient of selected medium at
     given x-ray wavelength [mm^-1]

     Mass attenuation coefficients are taken from NIST "Tables of X-Ray Mass
     Attenuation Coefficients and Mass Energy-Absorption Coefficients from 1
     keV to 20 MeV for Elements Z = 1 to 92 and 48 Additional Substances of Dosimetric Interest
     J. H. Hubbell and S. M. Seltzer

    http://www.nist.gov/pml/data/xraycoef/index.cfm
    """
    if medium == 'Helium':
        density = 1.663e-04
        # the table contains photon energy [Mev] and mass attenuation coefficient
        # mu/sigma [cm^2/g]
        mass_attenuation_coefficient = np.array([[1.00000e-03, 6.084e+01],
                                                 [1.50000e-03, 1.676e+01],
                                                 [2.00000e-03, 6.863e+00],
                                                 [3.00000e-03, 2.007e+00],
                                                 [4.00000e-03, 9.329e-01],
                                                 [5.00000e-03, 5.766e-01],
                                                 [6.00000e-03, 4.195e-01],
                                                 [8.00000e-03, 2.933e-01],
                                                 [1.00000e-02, 2.476e-01],
                                                 [1.50000e-02, 2.092e-01],
                                                 [2.00000e-02, 1.960e-01],
                                                 [3.00000e-02, 1.838e-01],
                                                 [4.00000e-02, 1.763e-01],
                                                 [5.00000e-02, 1.703e-01],
                                                 [6.00000e-02, 1.651e-01],
                                                 [8.00000e-02, 1.562e-01],
                                                 [1.00000e-01, 1.486e-01],
                                                 [1.50000e-01, 1.336e-01],
                                                 [2.00000e-01, 1.224e-01],
                                                 [3.00000e-01, 1.064e-01],
                                                 [4.00000e-01, 9.535e-02],
                                                 [5.00000e-01, 8.707e-02],
                                                 [6.00000e-01, 8.054e-02],
                                                 [8.00000e-01, 7.076e-02],
                                                 [1.00000e+00, 6.362e-02],
                                                 [1.25000e+00, 5.688e-02],
                                                 [1.50000e+00, 5.173e-02],
                                                 [2.00000e+00, 4.422e-02],
                                                 [3.00000e+00, 3.503e-02],
                                                 [4.00000e+00, 2.949e-02],
                                                 [5.00000e+00, 2.577e-02],
                                                 [6.00000e+00, 2.307e-02],
                                                 [8.00000e+00, 1.940e-02],
                                                 [1.00000e+01, 1.703e-02],
                                                 [1.50000e+01, 1.363e-02],
                                                 [2.00000e+01, 1.183e-02]])

    elif medium == 'Air':
        density = 1.205e-03
        mass_attenuation_coefficient = np.array([[1.00000e-03, 3.606e+03],
                                                 [1.50000e-03, 1.191e+03],
                                                 [2.00000e-03, 5.279e+02],
                                                 [3.00000e-03, 1.625e+02],
                                                 [3.20290e-03, 1.340e+02],
                                                 [3.202900000001e-03, 1.485e+02],
                                                 [4.00000e-03, 7.788e+01],
                                                 [5.00000e-03, 4.027e+01],
                                                 [6.00000e-03, 2.341e+01],
                                                 [8.00000e-03, 9.921e+00],
                                                 [1.00000e-02, 5.120e+00],
                                                 [1.50000e-02, 1.614e+00],
                                                 [2.00000e-02, 7.779e-01],
                                                 [3.00000e-02, 3.538e-01],
                                                 [4.00000e-02, 2.485e-01],
                                                 [5.00000e-02, 2.080e-01],
                                                 [6.00000e-02, 1.875e-01],
                                                 [8.00000e-02, 1.662e-01],
                                                 [1.00000e-01, 1.541e-01],
                                                 [1.50000e-01, 1.356e-01],
                                                 [2.00000e-01, 1.233e-01],
                                                 [3.00000e-01, 1.067e-01],
                                                 [4.00000e-01, 9.549e-02],
                                                 [5.00000e-01, 8.712e-02],
                                                 [6.00000e-01, 8.055e-02],
                                                 [8.00000e-01, 7.074e-02],
                                                 [1.00000e+00, 6.358e-02],
                                                 [1.25000e+00, 5.687e-02],
                                                 [1.50000e+00, 5.175e-02],
                                                 [2.00000e+00, 4.447e-02],
                                                 [3.00000e+00, 3.581e-02],
                                                 [4.00000e+00, 3.079e-02],
                                                 [5.00000e+00, 2.751e-02],
                                                 [6.00000e+00, 2.522e-02],
                                                 [8.00000e+00, 2.225e-02],
                                                 [1.00000e+01, 2.045e-02],
                                                 [1.50000e+01, 1.810e-02],
                                                 [2.00000e+01, 1.705e-02]])
    else:
        raise Exception('Unknown medium ' + medium)

    etw = 1.23985e-2  # [Mev*Angstroem]
    photon_energy = etw / wavelength

    if photon_energy < min(mass_attenuation_coefficient[:, 0]):
        raise Exception('Wavelength is too large, using nearest value')

    if photon_energy > max(mass_attenuation_coefficient[:, 0]):
        raise Exception('Wavelength is too small, using nearest value')

    # 0.1 here converts from cm^-1 to mm^-1
    mu = 0.1 * density * np.interp(photon_energy,
                                   mass_attenuation_coefficient[:, 0],
                                   mass_attenuation_coefficient[:, 1])

    return mu


def create_h5py_with_large_cache(filename, cache_size_mb):
    """
Allows to open the hdf5 file with specified cache size
    """
    # h5py does not allow to control the cache size from the high level
    # we employ the workaround
    # sources:
    #http://stackoverflow.com/questions/14653259/how-to-set-cache-settings-while-using-h5py-high-level-interface
    #https://groups.google.com/forum/#!msg/h5py/RVx1ZB6LpE4/KH57vq5yw2AJ
    propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    settings = list(propfaid.get_cache())
    settings[2] = 1024 * 1024 * cache_size_mb
    propfaid.set_cache(*settings)
    fid = h5py.h5f.create(filename, flags=h5py.h5f.ACC_EXCL, fapl=propfaid)
    fin = h5py.File(fid)
    return fin


def accumulate_intensity(intensity,
                         indices,
                         rebinned_data,
                         number_of_pixels_rebinned,
                         number_of_pixels,
                         all_in_memory):
    # remove elements which indices are outside the dataset
    pixels_in_range = np.logical_and(np.all(indices >= 0, axis=0),
                                     np.all(indices < np.reshape(number_of_pixels, (3, 1)), axis=0))
    intensity = intensity[pixels_in_range]
    indices = indices[:, pixels_in_range]

    # accumulate similar indices in a temporary in-memory array
    indices_str = np.ravel_multi_index((indices[0, :], indices[1, :], indices[2, :]), number_of_pixels)
    unique_ind_flatten, unique_ind_ind, n = np.unique(indices_str, return_index=True, return_inverse=True)

    accumulated_intensity = np.bincount(n, weights=intensity)
    no_accumulated_pixels = np.bincount(n)

    # now comes the fun part
    if all_in_memory:
        rebinned_data[unique_ind_flatten] = rebinned_data[unique_ind_flatten] + accumulated_intensity
        number_of_pixels_rebinned[unique_ind_flatten] = number_of_pixels_rebinned[
                                                            unique_ind_flatten] + no_accumulated_pixels
    else:
        #since h5py does not allow to add the data through high level interface, the low level interface is used instead
        unique_ind = indices[:, unique_ind_ind]

        old_intensity = np.zeros((unique_ind.shape[1],), dtype=rebinned_data.dtype)
        old_count = np.zeros((unique_ind.shape[1],), dtype=number_of_pixels_rebinned.dtype)

        fspace = h5py.h5s.create_simple(tuple(number_of_pixels))
        fspace.select_elements(unique_ind.T)

        mspace = h5py.h5s.create_simple(old_count.shape)

        rebinned_data._id.read(mspace, fspace, old_intensity)
        rebinned_data._id.write(mspace, fspace, old_intensity + accumulated_intensity)

        number_of_pixels_rebinned._id.read(mspace, fspace, old_count)
        number_of_pixels_rebinned._id.write(mspace, fspace, old_count + no_accumulated_pixels)


def correction_coefficients(h, instrument_parameters, medium, polarization_factor, polarization_plane_normal,
                            wavelength, wavevector, detector_normal):

    [_, scattering_vector_mm, unit_scattering_vector] = det2lab_xds(h, 0, **instrument_parameters)

    mu = air_absorption_coefficient(medium, wavelength)
    air_absorption = np.exp(
        -mu * np.sqrt(np.sum(scattering_vector_mm ** 2, axis=0)))  
    
    #% Polarisation
    polarization_plane_normal = np.array(polarization_plane_normal)
    polarization_plane_normal = polarization_plane_normal / np.linalg.norm(polarization_plane_normal)  # just in case
    polarization_plane_other_comp = np.cross(polarization_plane_normal, wavevector.T)
    polarization_plane_other_comp = polarization_plane_other_comp / np.linalg.norm(polarization_plane_other_comp)
    polarization_correction = (1 - polarization_factor) * (
        1 - np.dot(polarization_plane_normal, unit_scattering_vector) ** 2) + \
                              polarization_factor * (
                                  1 - np.dot(polarization_plane_other_comp, unit_scattering_vector) ** 2)
    #% solid angle correction
    detector_normal = detector_normal/norm(detector_normal)
    solid_angle_correction = abs(np.dot(detector_normal, unit_scattering_vector) ** 3)
    
    #corrections = solid_angle_correction.*polarization_correction.*air_absorption;
    corrections = solid_angle_correction * polarization_correction * air_absorption
    
    #if(exist('detector_efficiency_correction.mat','file'))
    #    load detector_efficiency_correction;
    #    corrections = corrections./detector_efficiency_correction(:)';
    #    clear detector_efficiency_correction;
    #end
    #
    #corrections = corrections(measured_pixels)';
    #
    # TODO: implement detector efficiency for Pilatus
    return corrections


# function [rebinned_data,number_of_pixels_rebinned,Tp,metric_tensor]=...
# reconstruct_data(filename_template,...
# last_image,...
# reconstruct_in_orthonormal_basis,...
# maxind,...
# number_of_pixels,...
# measured_pixels,...
# microsteps,...
#                     unit_cell_transform_matrix)
def reconstruct_data(filename_template,
                     first_image,
                     last_image,
                     maxind,
                     number_of_pixels,
                     reconstruct_in_orthonormal_basis=False,
                     measured_pixels=None,
                     microsteps=[1, 1, 1],
                     #on the angle also allows fractional values. for example 1 1 0.1 will only take every tenth frame
                     unit_cell_transform_matrix=np.eye(3),
                     polarization_plane_normal=[0, 1, 0],  #default for synchrotron
                     polarization_factor=1,  #0.5 for laboratory
                     medium='Air',  #'Air' or 'Helium
                     path_to_XPARM=".",
                     output_filename='reconstruction.h5',
                     size_of_cache=100,
                     all_in_memory=False,
                     override=False,
                     scale=None,
                     keep_number_of_pixels=False):

    def image_name(num):
        return filename_template % num  #test above


    def get_image(fname):
        return fabio.open(fname).data

    #TODO: check mar2000 and 2300 is done properly with respect to oversaturated reflections. Check what happens in other cases too

    if measured_pixels is None:
        measured_pixels = get_image(image_name(1)) >= 0

    #TODO: maybe add scale 'median' where scale is defined as a median of a frame divided by a median of a first frame?
    if scale is None:
        scale = np.ones(last_image-first_image+1)
    else:
        assert(len(scale)==last_image-first_image+1)

    if microsteps is None:
        microsteps = (1, 1, 1)

    assert 3 == len(microsteps), 'Microsteps should have three values: along x, y and phi.'

    incr_xy = np.array(microsteps)[0:2]
    assert np.all(np.mod(incr_xy, 1) == 0), 'microsteps in x and y direction should be integer'

    #TODO: microstepping is omitted in this version
    assert np.all(
        incr_xy == np.array([1, 1])), 'microsteps are not implemented atm'  #see next section and also down there
    if not np.all(incr_xy == np.array([1, 1])):
        def get_image(fname):
            np.kron(fabio.open(fname).data,
                 np.ones(incr_xy))  # TODO: remove the copypaste from the previous definition of get_image
        measured_pixels = 1 == np.kron(measured_pixels, np.ones(incr_xy))

    microsteps = microsteps[2]
    if microsteps < 1:
        image_increment = 1 / microsteps
        assert (np.mod(image_increment, 1) == 0)
        microsteps = 1
    else:
        image_increment = 1

    assert (3, 3) == np.shape(unit_cell_transform_matrix)

    # prepare hkl indices
    h = np.mgrid[1:np.size(measured_pixels, 1) + 1, 1:np.size(measured_pixels, 0) + 1].T
    h = h.reshape((int(np.size(h) / 2), 2))
    h = h[np.reshape(measured_pixels, (-1)), :]

    number_of_pixels = np.array(number_of_pixels)
    assert len(number_of_pixels) == 3

    maxind = np.array(maxind, dtype=np.float_)
    assert len(maxind) == 3

    step_size_inv = 1.0 * (number_of_pixels - 1) / maxind / 2
    step_size = 1.0/step_size_inv

    to_index = lambda c: np.around(step_size_inv[:,np.newaxis]*(c+maxind[:,np.newaxis])).astype(np.int64)

    if output_filename is not None:
        if os.path.exists(output_filename):
            if override:
                os.remove(output_filename)
            else:
                raise Exception('file ' + output_filename + ' already exists')
        output_file = create_h5py_with_large_cache(output_filename, size_of_cache)

    if all_in_memory:
        rebinned_data = np.zeros(np.prod(number_of_pixels),dtype=np.float_)
        number_of_pixels_rebinned = np.zeros(np.prod(number_of_pixels),dtype=np.int_)
    else:
        if output_filename is None:
            raise Exception("output filename shoud be provided")

        rebinned_data = output_file.create_dataset('rebinned_data', shape=number_of_pixels, dtype='float32',
                                                   chunks=True)
        number_of_pixels_rebinned = output_file.create_dataset('number_of_pixels_rebinned', shape=number_of_pixels,
                                                               dtype='int', chunks=True)

    #read_xparm
    instrument_parameters = read_XPARM(path_to_XPARM)

    unit_cell_vectors = instrument_parameters['unit_cell_vectors']
    starting_frame = instrument_parameters['starting_frame']
    starting_angle = instrument_parameters['starting_angle']
    rotation_axis = instrument_parameters['rotation_axis']
    wavevector = instrument_parameters['wavevector']
    wavelength = instrument_parameters['wavelength']
    oscillation_angle = instrument_parameters['oscillation_angle']
    detector_normal = instrument_parameters['detector_normal']

    #TODO: implement microstepping
    #%in case of microstepping
    #if exist('incr_xy','var')
    #    NX=NX*incr_xy(1);
    #    NY=NY*incr_xy(2);
    #    pixelsize_x=pixelsize_x/incr_xy(1);
    #    pixelsize_y=pixelsize_y/incr_xy(2);
    #    x_center=x_center*incr_xy(1);
    #    y_center=y_center*incr_xy(2);
    #end

    unit_cell_vectors = np.dot(unit_cell_transform_matrix, unit_cell_vectors)

    if reconstruct_in_orthonormal_basis:
        [Q, _] = np.linalg.qr(unit_cell_vectors.T)
        unit_cell_vectors = Q.T


    metric_tensor = np.dot(unit_cell_vectors, unit_cell_vectors.T)
    [_, normalized_metric_tensor] = cov2corr(metric_tensor)
    transfrom_matrix = np.linalg.cholesky(np.linalg.inv(normalized_metric_tensor))

    corrections = correction_coefficients(h, instrument_parameters, medium, polarization_factor,
                                          polarization_plane_normal, wavelength, wavevector, detector_normal)


    micro_oscillation_angle = oscillation_angle / microsteps

    #Calculate h for frame number 0
    h_starting = det2lab_xds(h, 0, **instrument_parameters)[0]

    for frame_number in np.arange(first_image, last_image+1, image_increment):
        print ("reconstructing frame number %i" % frame_number)

        image = get_image(image_name(frame_number))
        image = image[measured_pixels]
        image = image / corrections * scale[frame_number-first_image]

        for m in np.arange(0, microsteps):
            #Phi is with respect to phi at frame number 0
            phi_minus_phi0=( (frame_number - 0.5) * microsteps + m + 0.5) * micro_oscillation_angle
            h_frame = np.dot(rotvec2mat(rotation_axis, -np.deg2rad(phi_minus_phi0)), h_starting)

            fractional = np.dot(unit_cell_vectors, h_frame)
            del h_frame
            indices = to_index(fractional)
            del fractional

            accumulate_intensity(image, indices, rebinned_data, number_of_pixels_rebinned, number_of_pixels,
                                 all_in_memory)
    
            
    if all_in_memory:
        if output_filename is None:
            result = {}
        else:
            result = output_file

        if keep_number_of_pixels:
            result["rebinned_data"] = np.reshape(rebinned_data, number_of_pixels)
            result["number_of_pixels_rebinned"] = np.reshape(number_of_pixels_rebinned, number_of_pixels)
        else:
            rebinned_data/=number_of_pixels_rebinned
            result["data"] = np.reshape(rebinned_data, number_of_pixels)
    else:
        result = output_file
        if not keep_number_of_pixels:
            data = output_file.create_dataset('data', shape=number_of_pixels, dtype='float32', 
                                              chunks=True)
            for i in range(number_of_pixels[0]):
                data[i,:,:]=result["rebinned_data"][i,:,:]/result["number_of_pixels_rebinned"][i,:,:]
            del result['rebinned_data']
            del result['number_of_pixels_rebinned']

    if keep_number_of_pixels:
        result['format']="Yell 0.9"
    else:
        result['format']="Yell 1.0"
            
    result['space_group_nr'] = instrument_parameters['space_group_nr']
    result['unit_cell'] = instrument_parameters['cell']
    result['metric_tensor'] = metric_tensor
    result["step_sizes"] = step_size
    result["lower_limits"] = -maxind
    result['is_direct'] = False

    
    if output_filename is None:
        return result
    else:
        result.close()

        
        
    
        
#todo: add lower limits, they are needed here
#todo: add string for file version
#todo: think of making the output nexus compatible
