"""Tightly-coupled pseudorange and Doppler factors.

Receiver clock bias/drift states use metres and metres/second.  Pose translation,
receiver velocity, satellite position, and satellite velocity are all ECEF.
"""

from __future__ import annotations

import gtsam
import numpy as np

from sensors.gnss_models import CorrectedGnssMeasurement
from sensors.gnss_satellite import SPEED_OF_LIGHT


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def create_raw_pseudorange_factor(
    pose_key: int,
    clock_bias_key: int,
    measurement: CorrectedGnssMeasurement,
    noise_model: gtsam.noiseModel.Base,
    lever_arm_body: np.ndarray | None = None,
    reference_ecef: np.ndarray | None = None,
    ecef_R_local: np.ndarray | None = None,
    inter_system_bias_key: int | None = None,
    wet_delay_key: int | None = None,
) -> gtsam.CustomFactor:
    """Connect one corrected code observation to pose and receiver clock bias."""
    if measurement.pseudorange_m is None:
        raise ValueError("measurement has no pseudorange")
    lever = np.zeros(3) if lever_arm_body is None else np.asarray(lever_arm_body, dtype=float).reshape(3)
    origin = np.zeros(3) if reference_ecef is None else np.asarray(reference_ecef, dtype=float).reshape(3)
    frame_rotation = np.eye(3) if ecef_R_local is None else np.asarray(ecef_R_local, dtype=float).reshape(3, 3)
    satellite = measurement.satellite_state

    def error(_factor, values: gtsam.Values, jacobians) -> np.ndarray:
        pose = values.atPose3(pose_key)
        rotation = pose.rotation().matrix()
        antenna_local = np.asarray(pose.translation()) + rotation @ lever
        receiver = origin + frame_rotation @ antenna_local
        delta = satellite.position_ecef - receiver
        distance = np.linalg.norm(delta)
        line_of_sight = delta / distance
        inter_system_bias = (
            0.0 if inter_system_bias_key is None
            else values.atDouble(inter_system_bias_key)
        )
        wet_mapping = (
            0.0 if wet_delay_key is None else
            1.0 / max(np.sin(float(measurement.elevation_rad)), np.sin(np.deg2rad(3.0)))
        )
        wet_delay = (
            0.0 if wet_delay_key is None
            else wet_mapping * values.atDouble(wet_delay_key)
        )
        predicted = (
            distance
            + values.atDouble(clock_bias_key)
            + inter_system_bias
            - SPEED_OF_LIGHT * satellite.clock_bias_s
            + measurement.atmosphere.total_m
            + wet_delay
        )
        if jacobians is not None:
            antenna_pose_jacobian = np.hstack((-rotation @ _skew(lever), rotation))
            jacobians[0] = -line_of_sight.reshape(1, 3) @ frame_rotation @ antenna_pose_jacobian
            jacobians[1] = np.ones((1, 1))
            if inter_system_bias_key is not None:
                jacobians[2] = np.ones((1, 1))
            if wet_delay_key is not None:
                wet_index = 3 if inter_system_bias_key is not None else 2
                jacobians[wet_index] = np.array([[wet_mapping]])
        return np.array([predicted - measurement.pseudorange_m])

    keys = [pose_key, clock_bias_key]
    if inter_system_bias_key is not None:
        keys.append(inter_system_bias_key)
    if wet_delay_key is not None:
        if measurement.elevation_rad is None:
            raise ValueError("wet-delay factor requires measurement elevation")
        keys.append(wet_delay_key)
    return gtsam.CustomFactor(noise_model, keys, error)


def create_doppler_factor(
    pose_key: int,
    velocity_key: int,
    clock_drift_key: int,
    measurement: CorrectedGnssMeasurement,
    noise_model: gtsam.noiseModel.Base,
    lever_arm_body: np.ndarray | None = None,
    reference_ecef: np.ndarray | None = None,
    ecef_R_local: np.ndarray | None = None,
) -> gtsam.CustomFactor:
    """Connect Doppler range-rate to pose, ECEF velocity, and clock drift.

    Antenna rotational velocity is omitted.  Add ``omega_ecef × lever_ecef``
    to receiver velocity when precise body angular-rate compensation is needed.
    """
    if measurement.range_rate_mps is None:
        raise ValueError("measurement has no Doppler range rate")
    lever = np.zeros(3) if lever_arm_body is None else np.asarray(lever_arm_body, dtype=float).reshape(3)
    origin = np.zeros(3) if reference_ecef is None else np.asarray(reference_ecef, dtype=float).reshape(3)
    frame_rotation = np.eye(3) if ecef_R_local is None else np.asarray(ecef_R_local, dtype=float).reshape(3, 3)
    satellite = measurement.satellite_state

    def error(_factor, values: gtsam.Values, jacobians) -> np.ndarray:
        pose = values.atPose3(pose_key)
        rotation = pose.rotation().matrix()
        antenna_local = np.asarray(pose.translation()) + rotation @ lever
        receiver_position = origin + frame_rotation @ antenna_local
        receiver_velocity = frame_rotation @ np.asarray(values.atVector(velocity_key)).reshape(3)
        delta = satellite.position_ecef - receiver_position
        distance = np.linalg.norm(delta)
        line_of_sight = delta / distance
        relative_velocity = satellite.velocity_ecef - receiver_velocity
        predicted = (
            line_of_sight @ relative_velocity
            + values.atDouble(clock_drift_key)
            - SPEED_OF_LIGHT * satellite.clock_drift_sps
        )
        if jacobians is not None:
            projector = np.eye(3) - np.outer(line_of_sight, line_of_sight)
            derivative_position = -(relative_velocity @ projector) / distance
            antenna_pose_jacobian = np.hstack((-rotation @ _skew(lever), rotation))
            jacobians[0] = derivative_position.reshape(1, 3) @ frame_rotation @ antenna_pose_jacobian
            jacobians[1] = -line_of_sight.reshape(1, 3) @ frame_rotation
            jacobians[2] = np.ones((1, 1))
        return np.array([predicted - measurement.range_rate_mps])

    return gtsam.CustomFactor(
        noise_model, [pose_key, velocity_key, clock_drift_key], error
    )
