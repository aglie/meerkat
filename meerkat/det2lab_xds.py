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
