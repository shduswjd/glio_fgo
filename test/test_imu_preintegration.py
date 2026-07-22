import sys
from pathlib import Path

import gtsam
import numpy as np
import pytest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
try:
    from factors.imu_preintegration import (
        ImuNoise,
        create_bias_factor,
        create_bias_random_walk,
        create_imu_factor,
        create_pim,
        create_preintegration_params,
        integrate_imu_measurements,
        predict_next_state,
    )
    from sensors.imu import ImuMeasurement, read_imu_txt
    from states import NavigationState, StateKeys
except ImportError:
    src_root = root / "src"
    if src_root.exists():
        sys.path.insert(0, str(src_root))
    from factors.imu_preintegration import (
        ImuNoise,
        create_bias_factor,
        create_bias_random_walk,
        create_imu_factor,
        create_pim,
        create_preintegration_params,
        integrate_imu_measurements,
        predict_next_state,
    )
    from sensors.imu import ImuMeasurement, read_imu_txt
    from states import NavigationState, StateKeys


def test_imu_measurement_validates_vector_shape():
    with pytest.raises(ValueError, match="shape"):
        ImuMeasurement(
            timestamp=0.0,
            acceleration=[0.0, 0.0],
            angular_velocity=[0.0, 0.0, 0.0],
        )

    with pytest.raises(ValueError, match="orientation"):
        ImuMeasurement(
            timestamp=0.0,
            acceleration=np.zeros(3),
            angular_velocity=np.zeros(3),
            orientation=[0.0, 0.0, 1.0],
        )


def test_read_imu_txt(tmp_path):
    imu_file = tmp_path / "imu.txt"
    np.savetxt(
        imu_file,
        [
            [1_000_000_000, 0.0, 0.0, 9.81, 0.1, 0.2, 0.3],
            [1_010_000_000, 0.0, 0.0, 9.81, 0.1, 0.2, 0.3],
        ],
    )

    measurements = list(read_imu_txt(imu_file, timestamp_scale=1e-9))

    assert len(measurements) == 2
    assert measurements[0].timestamp == pytest.approx(1.0)
    assert np.allclose(measurements[0].acceleration, [0.0, 0.0, 9.81])
    assert np.allclose(measurements[0].angular_velocity, [0.1, 0.2, 0.3])


def test_preintegration_prediction_and_factors():
    noise = ImuNoise(0.1, 0.01, 1e-4, 1e-3, 1e-4)
    params = create_preintegration_params(9.81, noise)
    pim = create_pim(params, NavigationState.identity().bias)
    measurements = [
        ImuMeasurement(0.00, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]),
        ImuMeasurement(0.01, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]),
        ImuMeasurement(0.02, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]),
    ]

    integrate_imu_measurements(measurements, pim)
    predicted = predict_next_state(NavigationState.identity(), pim)
    zero_change, bias_noise = create_bias_random_walk(noise, pim)
    imu_factor = create_imu_factor(0, pim)
    bias_factor = create_bias_factor(0, zero_change, bias_noise)

    assert pim.deltaTij() == pytest.approx(0.02)
    assert np.allclose(predicted.velocity, np.zeros(3), atol=1e-10)
    assert list(imu_factor.keys()) == [
        StateKeys.at(0).pose,
        StateKeys.at(0).velocity,
        StateKeys.at(1).pose,
        StateKeys.at(1).velocity,
        StateKeys.at(0).bias,
    ]
    assert list(bias_factor.keys()) == [
        StateKeys.at(0).bias,
        StateKeys.at(1).bias,
    ]


def test_preintegration_params_accept_body_sensor_extrinsic():
    noise = ImuNoise(0.1, 0.01, 1e-4, 1e-3, 1e-4)
    body_P_sensor = gtsam.Pose3(
        gtsam.Rot3.Rz(np.pi),
        np.array([0.1, 0.0, 0.3]),
    )

    params = create_preintegration_params(9.81, noise, body_P_sensor)

    assert isinstance(params, gtsam.PreintegrationParams)


def test_imu_noise_accepts_scientific_notation_loaded_as_string():
    noise = ImuNoise("1e-4", "1e-5", "1e-8", "1e-6", "1e-7")

    assert noise.integration_sigma == pytest.approx(1e-8)
