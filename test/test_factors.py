import sys
from pathlib import Path

import gtsam
import numpy as np
import pytest

src_root = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src_root))

from factors.prior import create_prior_factors, create_prior_noise
from factors.gnss_position import (
    create_factor_from_measurement,
    create_gnss_noise,
    create_gnss_position_factor,
    llh_to_enu,
)
from sensors.gnss import GnssMeasurement
from sensors.imu import ImuMeasurement
from states import NavigationState, StateKeys
import optimizer as optimizer_module
from optimizer import (
    build_graph,
    load_config,
    load_imu_segment,
    save_position_angular_velocity_csv,
    trim_imu_to_gnss_start,
)


def test_create_prior_factors_for_initial_state():
    state = NavigationState.identity(timestamp=0.0)
    noise_models = create_prior_noise(
        rotation_sigma=np.array([0.1, 0.1, 0.5]),
        position_sigma=np.array([1.0, 1.0, 2.0]),
        velocity_sigma=0.5,
        accel_bias_sigma=np.full(3, 0.1),
        gyro_bias_sigma=np.full(3, 0.01),
    )

    factors = create_prior_factors(state, *noise_models, index=0)
    keys = StateKeys.at(0)

    assert [list(factor.keys()) for factor in factors] == [
        [keys.pose],
        [keys.velocity],
        [keys.bias],
    ]

    # Prior의 기준 state를 initial values에 넣으면 초기 error는 0이어야 한다.
    values = gtsam.Values()
    state.insert_into(values, index=0)
    assert sum(factor.error(values) for factor in factors) == pytest.approx(0.0)


def test_prior_noise_rejects_non_positive_sigma():
    with pytest.raises(ValueError, match="positive"):
        create_prior_noise(
            rotation_sigma=np.array([0.1, 0.1, 0.0]),
            position_sigma=np.ones(3),
            velocity_sigma=0.5,
            accel_bias_sigma=np.ones(3),
            gyro_bias_sigma=np.ones(3),
        )


def test_llh_to_enu_returns_zero_at_reference():
    reference = np.array([52.0, 13.0, 40.0])
    assert llh_to_enu(*reference, *reference) == pytest.approx(np.zeros(3))


def test_gnss_noise_clamps_invalid_eigenvalues():
    model = create_gnss_noise(np.diag([-1.0, 0.0, 4.0]), minimum_sigma=0.2)
    assert model.covariance() == pytest.approx(np.diag([0.04, 0.04, 4.0]))


def test_gnss_factor_residual_includes_body_lever_arm():
    key = gtsam.symbol("x", 0)
    pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, -0.2, 0.3), [4.0, 5.0, 6.0])
    lever = np.array([1.0, -0.2, 0.4])
    antenna_position = np.asarray(pose.transformFrom(lever))
    factor = create_gnss_position_factor(
        key,
        antenna_position,
        gtsam.noiseModel.Isotropic.Sigma(3, 1.0),
        lever,
    )
    values = gtsam.Values()
    values.insert(key, pose)
    assert factor.unwhitenedError(values) == pytest.approx(np.zeros(3))


def test_create_factor_from_measurement_uses_measurement_covariance():
    reference = np.array([52.0, 13.0, 40.0])
    measurement = GnssMeasurement(
        timestamp=1.0,
        latitude=reference[0],
        longitude=reference[1],
        altitude=reference[2],
        position_covariance=np.diag([1.0, 2.0, 3.0]),
        covariance_type=2,
        status=0,
        service=1,
    )
    key = gtsam.symbol("x", 0)
    factor = create_factor_from_measurement(key, measurement, reference)
    values = gtsam.Values()
    values.insert(key, gtsam.Pose3())
    assert factor.unwhitenedError(values) == pytest.approx(np.zeros(3))


def test_build_imu_graph_adds_gnss_factor_to_each_keyframe():
    imu_measurements = [
        ImuMeasurement(0.0, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]),
        ImuMeasurement(1.0, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]),
    ]
    gnss_measurements = [
        GnssMeasurement(
            timestamp=timestamp,
            latitude=52.0,
            longitude=13.0,
            altitude=40.0,
            position_covariance=np.eye(3),
            covariance_type=2,
            status=0,
            service=1,
        )
        for timestamp in (0.0, 1.0)
    ]

    graph, initial_values, last_index = build_graph(
        imu_measurements,
        keyframe_interval=1.0,
        gnss_measurements=gnss_measurements,
    )

    # prior 3 + GNSS 2 + IMU 1 + bias random walk 1
    assert graph.size() == 7
    assert last_index == 1
    assert initial_values.atPose3(StateKeys.at(1).pose).translation() == pytest.approx(
        np.zeros(3)
    )


def test_build_graph_does_not_reuse_one_gnss_measurement():
    imu_measurements = [
        ImuMeasurement(timestamp, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
        for timestamp in (0.0, 1.0, 2.0)
    ]
    gnss_measurements = [
        GnssMeasurement(
            timestamp=1.0,
            latitude=52.0,
            longitude=13.0,
            altitude=40.0,
            position_covariance=np.eye(3),
            covariance_type=2,
            status=0,
            service=1,
        )
    ]

    graph, _, last_index = build_graph(
        imu_measurements,
        keyframe_interval=1.0,
        gnss_measurements=gnss_measurements,
    )

    # prior 3 + GNSS 1 + (IMU 2 + bias random walk 2)
    assert graph.size() == 8
    assert last_index == 2


def test_load_optimizer_yaml_config():
    config_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    config = load_config(config_path)

    assert config["topics"]["imu"] == "/microstrain/imu/data"
    assert config["gnss"]["lever_arm_body"] == [-0.18, 0.0, 0.5317]


def test_load_imu_segment_accepts_none_duration(monkeypatch):
    imu_measurements = [
        ImuMeasurement(0.0, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]),
        ImuMeasurement(1.0, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]),
        ImuMeasurement(2.0, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0]),
    ]
    monkeypatch.setattr(
        optimizer_module,
        "read_imu_ros1_bag",
        lambda _path, _topic: iter(imu_measurements),
    )

    loaded = load_imu_segment(Path("unused.bag"), "/imu", duration=None)

    assert loaded == imu_measurements


def test_trim_imu_to_gnss_start_uses_first_imu_at_or_after_gnss():
    imu_measurements = [
        ImuMeasurement(timestamp, [0.0, 0.0, 9.81], [0.0, 0.0, 0.0])
        for timestamp in (0.0, 0.1, 0.2, 0.3)
    ]
    gnss_measurements = [
        GnssMeasurement(
            timestamp=0.15,
            latitude=52.0,
            longitude=13.0,
            altitude=40.0,
            position_covariance=np.eye(3),
            covariance_type=2,
            status=0,
            service=1,
        )
    ]

    trimmed, skipped = trim_imu_to_gnss_start(
        imu_measurements,
        gnss_measurements,
    )

    assert [item.timestamp for item in trimmed] == [0.2, 0.3]
    assert skipped == 2


def test_save_position_angular_velocity_csv(tmp_path):
    imu_measurements = [
        ImuMeasurement(0.0, [0.0, 0.0, 9.81], [0.1, 0.2, 0.3]),
        ImuMeasurement(1.0, [0.0, 0.0, 9.81], [0.4, 0.5, 0.6]),
    ]
    graph, initial_values, last_index = build_graph(
        imu_measurements,
        keyframe_interval=1.0,
    )
    output_path = tmp_path / "trajectory.csv"

    save_position_angular_velocity_csv(
        output_path,
        initial_values,
        imu_measurements,
        keyframe_interval=1.0,
        last_index=last_index,
        body_P_sensor=None,
    )

    rows = output_path.read_text().splitlines()
    assert len(rows) == 3
    assert rows[0].startswith("timestamp_s,elapsed_s,position_east_m")
    first_values = rows[1].split(",")
    assert [float(value) for value in first_values[5:8]] == pytest.approx(
        [0.1, 0.2, 0.3]
    )
