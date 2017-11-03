import numpy as np


def rotvec2mat(u, phi):
    """Convert rotation from axis and angle to matrix representation"""

    phi = np.squeeze(phi)
    norm_u = np.linalg.norm(u)

    if norm_u < 1e-12:
        raise Exception("the rotation vector is equal to zero")

    u = u / norm_u
    # http://en.wikipedia.org/wiki/Rotation_matrix
    s = np.sin(phi)
    c = np.cos(phi)
    t = 1 - c

    ux = u[0]
    uy = u[1]
    uz = u[2]
    res = np.array([[t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
                    [t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
                    [t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c]])

    return res


def det2lab_xds(
        pixels_coord, frame_number,
        starting_frame, starting_angle, oscillation_angle,
        rotation_axis,
        wavelength, wavevector,
        NX, NY, pixelsize_x, pixelsize_y,
        distance_to_detector, x_center, y_center,
        detector_x, detector_y, detector_normal, **kwargs):
    """Converts pixels coordinates from the frame into q-vector"""

    array_shape = (1, 3)

    if detector_x.shape == array_shape:
        detector_x = detector_x.T
        detector_y = detector_y.T
        detector_normal = detector_normal.T
    if wavevector.shape == array_shape:
        wavevector = wavevector.T
    if rotation_axis.shape == array_shape:
        rotation_axis = rotation_axis.T
    xmm = (pixels_coord[:, [0]] - x_center) * pixelsize_x
    ymm = (pixels_coord[:, [1]] - y_center) * pixelsize_y
    # find scattering vector of each pixel
    scattering_vector_mm = np.outer(xmm, detector_x) + \
                           np.outer(ymm, detector_y) + \
                           distance_to_detector * np.outer(np.ones(shape=xmm.shape),
                                                           detector_normal)
    scattering_vector_mm = scattering_vector_mm.T
    phi = (frame_number - starting_frame) * oscillation_angle + \
          starting_angle
    # calculating norm for each column
    norms = np.sum(scattering_vector_mm ** 2., axis=0) ** (1. / 2)
    #deviding scattering vector by its own norm
    unit_scattering_vector = scattering_vector_mm / norms
    #subtracting incident beam vector

    h = unit_scattering_vector / wavelength - \
        np.tile(wavevector, (unit_scattering_vector.shape[1], 1)).T
    #rotating
    if phi.size == 1:
        h = np.dot(rotvec2mat(rotation_axis.T, -2 * np.pi * phi / 360), h)
    else:
        for i in range(phi.size):
            h[:, [i]] = np.dot(
                rotvec2mat(rotation_axis.T, -2 * np.pi * phi[i] / 360), h[:, [i]])

    return h, scattering_vector_mm, unit_scattering_vector


def test_rotvec2mat():
    print (rotvec2mat(np.array([1, 2, 3]), -2 * np.pi * 90 / 360))


def test_det2lab_xds():
    instrument_parameters = {'NX': np.array([2463.]),
                             'NY': np.array([2527.]),
                             'cell': np.array([8.2287, 8.2299, 11.0122, 90.013, 90.025, 59.974]),
                             'detector_normal': np.array([0., 0., 1.]),
                             'detector_segment_crossection': np.array([1.00000000e+00, 1.00000000e+00, 2.46300000e+03,
                                                                       1.00000000e+00, 2.52700000e+03]),
                             'detector_segment_geometry': np.array([0., 0., 0., 1., 0., 0., 0., 1., 0.]),
                             'detector_x': np.array([1., 0., 0.]),
                             'detector_y': np.array([0., 1., 0.]),
                             'distance_to_detector': np.array([200.057556]),
                             'number_of_detector_segments': np.array([1.]),
                             'oscillation_angle': np.array([0.1]),
                             'pixelsize_x': np.array([0.172]),
                             'pixelsize_y': np.array([0.172]),
                             'rotation_axis': np.array([9.99999000e-01, -9.49000000e-04, 8.63000000e-04]),
                             'space_group_nr': np.array([1.]),
                             'starting_angle': np.array([0.]),
                             'starting_frame': np.array([2.]),
                             'unit_cell_vectors': np.array([[-2.497232, -0.802756, 10.471534],
                                                            [-3.449634, 4.595787, -2.116248],
                                                            [-7.040996, -6.77977, -2.671483]]),
                             'wavelength': np.array([0.774899]),
                             'wavevector': np.array([-0.001803, -0.001785, 1.290488]),
                             'x_center': np.array([1214.768921]),
                             'y_center': np.array([1261.04126])}

    bragg_peaks = np.array( \
        [0, 0, -1, 6.733E+03, 3.791E+03, 1134.6, 1235.3, 1313.9, 0.02007, 91, 20, 48.81,
         0, 0, 1, 2.534E+04, 1.417E+04, 1294.6, 1286.5, 1163.8, 0.01869, 97, 6, -65.66,
         0, 0, -2, 3.477E+04, 1.947E+04, 1055.3, 1210.2, 1399.5, 0.04181, 100, 19, 39.25,
         0, 0, 2, 6.908E+03, 4.025E+03, 1375.5, 1311.4, 1104.7, 0.03721, 84, 9, -71.30,
         0, 0, -3, 5.960E+03, 3.548E+03, 974.0, 1186.5, 1471.4, 0.06282, 90, 15, 31.72,
         0, 0, -4, 2.530E+03, 1.847E+03, 888.9, 1165.2, 1544.2, 0.08304, 100, 19, 24.08,
         0, 0, 4, 3.991E+02, 1.226E+03, 1542.7, 1349.7, 981.6, 0.06863, 100, 20, -78.05,
         0, 0, -5, 3.830E+03, 2.740E+03, 797.5, 1148.9, 1636.3, 0.09818, 79, 4, 15.73,
         0, 0, 5, 2.646E+03, 2.017E+03, 1629.7, 1363.2, 917.6, 0.08613, 100, 15, -82.24,
         0, 0, -6, 1.266E+06, 7.082E+05, 700.0, 1135.1, 1716.1, 0.11208, 100, 18, 9.18,
         0, -1, 5, 3.873E+05, 2.162E+05, 1621.8, 1067.1, 114.1, 0.17007, 100, 19, -162.95,
         0, -1, 5, 3.484E+05, 1.945E+05, 1622.3, 1433.2, 1398.7, 0.14894, 100, 21, -32.57,
         0, -1, 4, 4.500E+06, 2.510E+06, 1531.2, 1435.0, 1555.9, 0.15460, 100, 40, -13.64,
         0, -1, 4, 4.867E+06, 2.715E+06, 1531.0, 1082.6, 102.2, 0.15329, 100, 41, -163.49,
         0, -1, 3, 6.455E+06, 3.599E+06, 1445.4, 1098.6, 111.5, 0.13758, 100, 51, -161.71,
         0, -1, 3, 6.950E+06, 3.875E+06, 1447.3, 1422.1, 1649.0, 0.13493, 99, 28, -4.70,
         0, -1, 2, 7.910E+05, 4.411E+05, 1364.6, 1404.3, 1783.7, 0.11945, 100, 21, 9.62,
         0, -1, 2, 3.262E+06, 1.819E+06, 1363.3, 1113.2, 143.4, 0.12417, 86, 39, -157.83,
         0, 1, -2, 8.408E+05, 4.688E+05, 1061.8, 1407.3, 27.9, 0.12166, 84, 28, 167.63,
         0, -1, 1, 5.900E+05, 3.289E+05, 1283.2, 1124.9, 196.9, 0.11354, 100, 20, -154.11,
         0, 1, -1, 5.294E+05, 2.952E+05, 1142.3, 1394.0, 122.7, 0.11345, 100, 20, 158.42,
         0, -1, 0, 2.460E+06, 1.371E+06, 1204.6, 1132.2, 303.8, 0.10831, 100, 41, -160.60])

    bragg_peaks = np.reshape(bragg_peaks, (-1, 12))
    hkl = bragg_peaks[:, 0:3]
    pixels_coord = bragg_peaks[:, 5:7]
    frame_number = bragg_peaks[:, 7]
    unit_cell_vectors = np.array([[-2.497232, -0.802756, 10.471534],
                                  [-3.449634, 4.595787, -2.116248],
                                  [-7.040996, -6.77977, -2.671483]])

    laboratory_bragg_coordinates = det2lab_xds(pixels_coord, frame_number, **instrument_parameters)[
        0]  # the operator **should unpack the instrumental parameters i believe
    fractional_coordinates = np.dot(unit_cell_vectors.T, laboratory_bragg_coordinates)
    assert np.all(np.abs(fractional_coordinates.T - hkl) < 0.15), "Something seems to be wrong"
    print ("Test passed!")


if __name__ == "__main__":
    test_det2lab_xds()
    # test_rotvec2mat()
