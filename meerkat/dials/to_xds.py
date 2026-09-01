"""Convert a DIALS-processed experiment (.expt/.refl) directly to XDS format.

Replaces `dials.export format=xds`, which has real, verified bugs for at least
this beamline's geometry:
  * INCIDENT_BEAM_DIRECTION is written as a non-unit vector.
  * POLARIZATION_PLANE_NORMAL is written straight from DIALS' native frame
    with no rotation applied, even though the detector axes it writes
    alongside it (DIRECTION_OF_DETECTOR_X/Y-AXIS) ARE rotated to canonical
    (1,0,0)/(0,1,0) -- so the two are no longer mutually consistent.

Convention here: the incident beam direction is fixed at (0, 0, 1) in the
output frame. Every other lab-frame direction (detector axes, rotation axis,
polarization normal, crystal orientation) is rotated by the SAME rotation
that achieves that, so everything stays self-consistent by construction --
there is no second field to independently get wrong.

Quantities that are frame-invariant (pixel size, image size, beam centre in
pixels, crystal-to-detector distance, trusted pixel range, wavelength) are
taken directly from dxtbx without any rotation.

Detector-hardware specifics that dials.export gets right but that are not
part of the DIALS geometry model at all (module-gap masking) come from
DETECTOR_TABLE, keyed by detector identifier substring. Add an entry there
for a detector model not yet covered rather than guessing.
"""

from __future__ import annotations

import os

import numpy as np

from ..xds import vecs2cell, write_xparm

__all__ = [
    "DETECTOR_TABLE",
    "convert",
    "read_dials_geometry",
    "rotation_aligning",
    "write_spot_xds",
    "write_xds_inp",
]


# Module-gap masking and related static facts, per detector model. Keyed by a
# substring of dxtbx Panel.get_identifier(). Add an entry when a new detector
# model is encountered -- these are fixed hardware properties, not something
# derivable from a single dataset's geometry.
DETECTOR_TABLE = {
    "PILATUS 2M": {
        "detector": "PILATUS",
        "untrusted_rectangles": [
            (487, 495, 0, 1680),
            (981, 989, 0, 1680),
            (0, 1476, 195, 213),
            (0, 1476, 407, 425),
            (0, 1476, 619, 637),
            (0, 1476, 831, 849),
            (0, 1476, 1043, 1061),
            (0, 1476, 1255, 1273),
            (0, 1476, 1467, 1485),
        ],
        "trusted_region": (0.0, 1.41),
    },
}


def rotation_aligning(a, b):
    """Rotation matrix R such that R @ a is parallel to unit vector b."""
    a = np.asarray(a, dtype=float)
    a = a / np.linalg.norm(a)
    b = np.asarray(b, dtype=float)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        return _rotation_180(a)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _rotation_180(axis):
    """Rotation matrix for a 180-degree turn about an axis perpendicular to it."""
    other = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, other)) > 0.9:
        other = np.array([0.0, 1.0, 0.0])
    perp = np.cross(axis, other)
    perp /= np.linalg.norm(perp)
    return 2 * np.outer(perp, perp) - np.eye(3)


def read_dials_geometry(expt_path, experiment_index=0):
    """Extract instrument + crystal geometry from a DIALS .expt file.

    Returns a dict with:
      * everything meerkat.xds.write_xparm needs (see meerkat.xds.xparm),
      * a few extras for the XDS.INP writer: image_range, name_template,
        min_valid_pixel_value, overload, thickness, material, identifier.

    Beam direction is fixed at (0, 0, 1); see the module docstring.
    """
    from dxtbx.model.experiment_list import ExperimentListFactory

    experiments = ExperimentListFactory.from_json_file(str(expt_path), check_format=False)
    experiment = experiments[experiment_index]

    beam = experiment.beam
    detector = experiment.detector[0]
    goniometer = experiment.goniometer
    scan = experiment.scan
    crystal = experiment.crystal

    beam_direction_native = -np.array(beam.get_sample_to_source_direction())
    target = np.array([0.0, 0.0, 1.0])
    R = rotation_aligning(beam_direction_native, target)

    def rot(v):
        return R @ np.asarray(v, dtype=float)

    real_space_vectors = np.array([rot(v) for v in crystal.get_real_space_vectors()])
    cell = vecs2cell(real_space_vectors)

    wavelength = beam.get_wavelength()
    nx, ny = detector.get_image_size()
    px_x, px_y = detector.get_pixel_size()
    # XDS's ORGX,ORGY is the pixel position of the foot of the perpendicular from
    # the sample to the detector plane -- a purely geometric quantity, independent
    # of beam direction. It is NOT the same as "where the beam hits the detector"
    # (dxtbx's get_beam_centre_px), which differs whenever the detector is tilted
    # relative to the beam, as it is here. Verified against dials.export's own
    # (correct) ORGX,ORGY on a known-good dataset: matches to 3 decimals.
    # +0.5 is XDS's pixel-corner-vs-pixel-centre indexing convention.
    origin = np.array(detector.get_origin())
    fast_native = np.array(detector.get_fast_axis())
    slow_native = np.array(detector.get_slow_axis())
    beam_centre_x = -np.dot(origin, fast_native) / px_x + 0.5
    beam_centre_y = -np.dot(origin, slow_native) / px_y + 0.5
    min_valid_pixel_value, overload = detector.get_trusted_range()
    # XDS's own valid range is documented as >= 0; dxtbx's trusted_range commonly
    # starts at -1 (its own "anything is valid" convention), which recent XDS
    # builds reject outright rather than clamp.
    min_valid_pixel_value = max(0.0, min_valid_pixel_value)
    starting_angle, oscillation_angle = scan.get_oscillation()
    first_frame, last_frame = scan.get_image_range()

    params = {
        "starting_frame": float(first_frame),
        "starting_angle": float(starting_angle),
        "oscillation_angle": float(oscillation_angle),
        "rotation_axis": rot(goniometer.get_rotation_axis()),
        "wavelength": float(wavelength),
        "wavevector": target / wavelength,
        "space_group_nr": int(crystal.get_space_group().type().number()),
        "cell": cell,
        "unit_cell_vectors": real_space_vectors,
        "NX": int(nx),
        "NY": int(ny),
        "pixelsize_x": float(px_x),
        "pixelsize_y": float(px_y),
        "x_center": float(beam_centre_x),
        "y_center": float(beam_centre_y),
        "distance_to_detector": float(detector.get_distance()),
        "detector_x": rot(detector.get_fast_axis()),
        "detector_y": rot(detector.get_slow_axis()),
        "detector_normal": rot(detector.get_normal()),
    }

    extras = {
        "image_range": (int(first_frame), int(last_frame)),
        "name_template": experiment.imageset.get_template().replace("#", "?"),
        "min_valid_pixel_value": float(min_valid_pixel_value),
        "overload": float(overload),
        "thickness": float(detector.get_thickness()),
        "material": detector.get_material(),
        "identifier": detector.get_identifier(),
        "polarization_normal": rot(beam.get_polarization_normal()),
        "polarization_fraction": float(beam.get_polarization_fraction()),
    }

    return params, extras


def write_spot_xds(refl_path, out_path, experiment_index=0, min_weight=1.01):
    """Write SPOT.XDS from a DIALS .refl reflection table.

    Only rows flagged 'indexed' (equivalently 'strong' here -- both are the
    initial spot-finding result, before integration re-predicts and adds many
    more, weaker, positions) are written: those extra post-integration rows
    are not what indexing should be judged against, and including them
    measurably hurt IDXREF's refinement in testing.

    Sorted by intensity descending, matching dials' own SPOT.XDS writer
    (dials.util.xds.export_spot_xds) -- not required by XDS itself, but
    worth keeping consistent with the convention DIALS already established.

    x, y, z come straight from xyzobs.px.value -- no coordinate transform
    needed, SPOT.XDS is in detector pixel / frame units regardless of lab
    frame. The 4th column (intensity/weight) is floored strictly above 1:
    XDS's IDXREF ignores every SPOT.XDS row from the first one with
    intensity <= 1 onward, not just that one row, so a single weak-but-real
    spot can silently truncate the rest of the file.
    """
    from dials.array_family import flex

    reflections = flex.reflection_table.from_file(str(refl_path))
    if "id" in reflections and len(set(reflections["id"])) > 1:
        reflections = reflections.select(reflections["id"] == experiment_index)
    if "flags" in reflections:
        reflections = reflections.select(reflections.get_flags(reflections.flags.indexed))

    intensity_column = (
        "intensity.sum.value" if "intensity.sum.value" in reflections
        else "intensity.observed.value" if "intensity.observed.value" in reflections
        else None
    )
    if intensity_column is not None:
        reflections.sort(intensity_column, reverse=True)
        intensity = reflections[intensity_column]
    else:
        intensity = [min_weight] * len(reflections)

    xyz = reflections["xyzobs.px.value"]

    with open(out_path, "w") as f:
        for (x, y, z), i in zip(xyz, intensity):
            value = float(i) if float(i) > 1.0 else min_weight
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {value:.2f}\n")


def write_xds_inp(path, params, extras, detector_table=DETECTOR_TABLE):
    """Write XDS.INP from a (params, extras) pair from read_dials_geometry."""
    identifier = extras.get("identifier", "")
    hw = next((v for k, v in detector_table.items() if k in identifier), None)
    if hw is None:
        raise ValueError(
            f"no DETECTOR_TABLE entry matches detector identifier {identifier!r}. "
            "Add one -- module-gap masking is hardware-specific and cannot be "
            "guessed from a single dataset's geometry."
        )

    rot_axis = params["rotation_axis"]
    detector_x = params["detector_x"]
    detector_y = params["detector_y"]
    wavevector = params["wavevector"]
    beam_direction = wavevector / np.linalg.norm(wavevector)
    polarization_normal = extras["polarization_normal"]
    trusted_region = hw.get("trusted_region", (0.0, 1.41))

    lines = [
        f"DETECTOR={hw['detector']} "
        f"MINIMUM_VALID_PIXEL_VALUE={int(extras['min_valid_pixel_value'])} "
        f"OVERLOAD={int(extras['overload'])}",
        f"SENSOR_THICKNESS= {extras['thickness']:.3f}",
        f"DIRECTION_OF_DETECTOR_X-AXIS= {detector_x[0]:.6f} {detector_x[1]:.6f} {detector_x[2]:.6f}",
        f"DIRECTION_OF_DETECTOR_Y-AXIS= {detector_y[0]:.6f} {detector_y[1]:.6f} {detector_y[2]:.6f}",
        f"NX={params['NX']} NY={params['NY']} QX={params['pixelsize_x']:.4f} QY={params['pixelsize_y']:.4f}",
        f"DETECTOR_DISTANCE= {params['distance_to_detector']:.6f}",
        f"ORGX= {params['x_center']:.2f} ORGY= {params['y_center']:.2f}",
        f"ROTATION_AXIS= {rot_axis[0]:.5f} {rot_axis[1]:.5f} {rot_axis[2]:.5f}",
        f"STARTING_ANGLE= {params['starting_angle']:.3f}",
        f"OSCILLATION_RANGE= {params['oscillation_angle']:.3f}",
        f"X-RAY_WAVELENGTH= {params['wavelength']:.5f}",
        f"INCIDENT_BEAM_DIRECTION= {beam_direction[0]:.6f} {beam_direction[1]:.6f} {beam_direction[2]:.6f}",
        f"FRACTION_OF_POLARIZATION= {extras['polarization_fraction']:.3f}",
        f"POLARIZATION_PLANE_NORMAL= {polarization_normal[0]:.6f} "
        f"{polarization_normal[1]:.6f} {polarization_normal[2]:.6f}",
        f"NAME_TEMPLATE_OF_DATA_FRAMES= {extras['name_template']}",
        f"TRUSTED_REGION= {trusted_region[0]:.1f} {trusted_region[1]:.2f}",
    ]
    lines += [f"UNTRUSTED_RECTANGLE= {a} {b} {c} {d}" for a, b, c, d in hw["untrusted_rectangles"]]
    lines += [
        f"DATA_RANGE= {extras['image_range'][0]} {extras['image_range'][1]}",
        f"SPACE_GROUP_NUMBER= {params['space_group_nr']}",
        f"UNIT_CELL_CONSTANTS= {' '.join(f'{v:.4f}' for v in params['cell'])}",
        "JOB= IDXREF",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def convert(expt_path, refl_path, out_dir, experiment_index=0):
    """Write XDS.INP, XPARM.XDS and SPOT.XDS into out_dir from a DIALS experiment.

    The bug-free replacement for `dials.export *.expt *.refl format=xds`.
    Returns the (params, extras) geometry dict, in case the caller wants it.
    """
    os.makedirs(out_dir, exist_ok=True)
    params, extras = read_dials_geometry(expt_path, experiment_index=experiment_index)

    write_xds_inp(os.path.join(out_dir, "XDS.INP"), params, extras)
    write_xparm(os.path.join(out_dir, "XPARM.XDS"), params)
    write_spot_xds(refl_path, os.path.join(out_dir, "SPOT.XDS"), experiment_index=experiment_index)

    return params, extras
