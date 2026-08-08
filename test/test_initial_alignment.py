import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensors.imu import ImuMeasurement
from tightly_coupled_fgo import (
    _body_P_sensor,
    _consistent_course_yaw,
    _stationary_initialization,
)


def test_rfu_extrinsic_maps_sensor_axes_to_flu_body():
    pose = _body_P_sensor({
        "translation_m": [0, 0, 0],
        "quaternion_xyzw": [0, 0, -np.sqrt(0.5), np.sqrt(0.5)],
    })
    rotation = pose.rotation().matrix()

    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0, -1, 0])
    assert rotation @ np.array([0.0, 1.0, 0.0]) == pytest.approx([1, 0, 0])
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0, 0, 1])


def test_stationary_initialization_estimates_tilt_and_body_gyro_bias():
    roll, pitch = np.deg2rad([2.0, -3.0])
    body_specific_force = 9.81 * np.array([
        -np.sin(pitch),
        np.sin(roll) * np.cos(pitch),
        np.cos(roll) * np.cos(pitch),
    ])
    gyro_bias = np.array([1e-3, -2e-3, 3e-3])
    samples = [
        ImuMeasurement(k * 0.01, body_specific_force, gyro_bias)
        for k in range(101)
    ]

    estimated_roll, estimated_pitch, estimated_bias = _stationary_initialization(
        samples, 0, np.eye(3), 1.0, 9.81,
    )

    assert estimated_roll == pytest.approx(roll)
    assert estimated_pitch == pytest.approx(pitch)
    assert estimated_bias == pytest.approx(gyro_bias)


def test_course_alignment_handles_angle_wrap_and_rejects_slow_samples():
    directions = np.deg2rad([179.0, -179.0, 178.0])
    velocities = [np.array([0.1, 0.0, 0.0])] + [
        3.0 * np.array([np.cos(angle), np.sin(angle), 0.0])
        for angle in directions
    ]

    alignment = _consistent_course_yaw(velocities, 2.0, 5.0, 3)

    assert alignment is not None
    yaw, index = alignment
    assert index == 3
    assert abs(abs(np.rad2deg(yaw)) - 180.0) < 1.0
