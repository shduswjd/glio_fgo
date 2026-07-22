"""Minimal IMU + GNSS factor graph runner.

This is a pipeline smoke test, not yet the final multi-sensor optimizer.  It
reads a short part of a ROS1 bag, creates one state per keyframe, and optimizes
prior + IMU + bias + GNSS position factors.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gtsam
import numpy as np
import yaml

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
from factors.gnss_position import create_factor_from_measurement, llh_to_enu
from factors.prior import create_prior_factors, create_prior_noise
from sensors.gnss import DEFAULT_GNSS_TOPIC, GnssMeasurement, read_gnss_ros1_bag
from sensors.imu import ImuMeasurement, read_imu_ros1_bag
from states import NavigationState, StateKeys, create_initial_state


DEFAULT_BAG = Path("01_13B_Jackal/base_2023-07-18-14-26-48_0.bag")
DEFAULT_IMU_TOPIC = "/microstrain/imu/data"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_config(config_path: Path) -> dict:
    """YAML 설정 파일을 읽고 최상위 항목을 확인한다."""
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")

    required_sections = (
        "bag",
        "topics",
        "optimizer",
        "imu_noise",
        "imu_extrinsic",
        "gnss",
        "prior_noise",
    )
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"config is missing sections: {', '.join(missing)}")
    return config


def create_body_P_sensor(imu_extrinsic: dict) -> gtsam.Pose3:
    """YAML의 [qx, qy, qz, qw] IMU extrinsic을 GTSAM Pose3로 변환한다."""
    translation = np.asarray(imu_extrinsic["translation"], dtype=float)
    quaternion = np.asarray(imu_extrinsic["quaternion_xyzw"], dtype=float)

    if translation.shape != (3,):
        raise ValueError("imu_extrinsic.translation must have 3 values")
    if quaternion.shape != (4,):
        raise ValueError("imu_extrinsic.quaternion_xyzw must have 4 values")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
        raise ValueError("imu_extrinsic values must be finite")

    quaternion_norm = np.linalg.norm(quaternion)
    if quaternion_norm == 0.0:
        raise ValueError("imu_extrinsic quaternion must be non-zero")
    qx, qy, qz, qw = quaternion / quaternion_norm

    rotation = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
    return gtsam.Pose3(rotation, translation)


def load_imu_segment(
    bag_path: Path,
    topic: str,
    duration: float | None,
) -> list[ImuMeasurement]:
    """bag 시작부터 ``duration``초를 읽는다. None이면 전체를 읽는다."""
    measurements: list[ImuMeasurement] = []
    start_time: float | None = None

    for measurement in read_imu_ros1_bag(bag_path, topic):
        if start_time is None:
            start_time = measurement.timestamp
        if duration is not None and measurement.timestamp - start_time > duration:
            break
        measurements.append(measurement)

    if len(measurements) < 2:
        raise ValueError("the selected bag segment contains fewer than two IMU samples")
    return measurements


def load_gnss_segment(
    bag_path: Path,
    topic: str,
    start_time: float,
    end_time: float,
) -> list[GnssMeasurement]:
    """IMU 구간과 겹치는 GNSS 측정만 읽는다."""
    measurements = []
    for measurement in read_gnss_ros1_bag(bag_path, topic):
        if measurement.timestamp < start_time:
            continue
        if measurement.timestamp > end_time:
            break
        measurements.append(measurement)
    return measurements


def trim_imu_to_gnss_start(
    imu_measurements: list[ImuMeasurement],
    gnss_measurements: list[GnssMeasurement],
) -> tuple[list[ImuMeasurement], int]:
    """Discard IMU samples before the first valid GNSS measurement.

    Starting the graph before an absolute position is available leaves the
    initial pose constrained only by its prior and IMU propagation.  The first
    retained IMU sample becomes state zero and is at or after the first GNSS
    timestamp, so the normal nearest-measurement logic can anchor it.
    """
    if not gnss_measurements:
        raise ValueError("cannot align graph start without GNSS measurements")

    first_gnss_time = gnss_measurements[0].timestamp
    trimmed = [
        measurement
        for measurement in imu_measurements
        if measurement.timestamp >= first_gnss_time
    ]
    if len(trimmed) < 2:
        raise ValueError("fewer than two IMU samples remain after GNSS start")
    return trimmed, len(imu_measurements) - len(trimmed)


def nearest_gnss_measurement(
    measurements: list[GnssMeasurement],
    timestamp: float,
    maximum_time_difference: float,
) -> GnssMeasurement | None:
    """keyframe 시각에 충분히 가까운 GNSS 측정 하나를 고른다."""
    if not measurements:
        return None

    measurement = min(
        measurements,
        key=lambda item: abs(item.timestamp - timestamp),
    )
    if abs(measurement.timestamp - timestamp) > maximum_time_difference:
        return None
    return measurement


def nearest_unused_gnss_measurement(
    measurements: list[GnssMeasurement],
    timestamp: float,
    maximum_time_difference: float,
    used_indices: set[int],
) -> GnssMeasurement | None:
    """아직 사용하지 않은 GNSS 중 keyframe에 가장 가까운 측정을 고른다."""
    candidates = [
        (index, measurement)
        for index, measurement in enumerate(measurements)
        if index not in used_indices
    ]
    if not candidates:
        return None

    index, measurement = min(
        candidates,
        key=lambda item: abs(item[1].timestamp - timestamp),
    )
    if abs(measurement.timestamp - timestamp) > maximum_time_difference:
        return None

    used_indices.add(index)
    return measurement


def body_rotation_from_imu(
    measurement: ImuMeasurement,
    body_P_sensor: gtsam.Pose3 | None,
) -> gtsam.Rot3 | None:
    """IMU orientation을 T_world_body 회전으로 변환한다."""
    if measurement.orientation is None:
        return None

    qx, qy, qz, qw = measurement.orientation
    world_R_sensor = gtsam.Rot3.Quaternion(qw, qx, qy, qz)
    if body_P_sensor is None:
        return world_R_sensor

    # body_P_sensor.rotation()은 sensor 좌표를 body 좌표로 회전한다.
    sensor_R_body = body_P_sensor.rotation().inverse()
    return world_R_sensor.compose(sensor_R_body)


def gnss_antenna_position(
    measurement: GnssMeasurement,
    reference_llh: np.ndarray,
) -> np.ndarray:
    """GNSS 측정을 기준점에 대한 ENU antenna 위치로 변환한다."""
    return np.asarray(llh_to_enu(
        measurement.latitude,
        measurement.longitude,
        measurement.altitude,
        *reference_llh,
    ))


def estimate_gnss_velocity(
    measurements: list[GnssMeasurement],
    timestamp: float,
    reference_llh: np.ndarray,
    window: float,
) -> np.ndarray | None:
    """keyframe 주변 GNSS 위치를 직선으로 적합해 ENU 속도를 구한다."""
    nearby = [
        measurement for measurement in measurements
        if abs(measurement.timestamp - timestamp) <= window
    ]
    if len(nearby) < 2:
        return None

    times = np.array([item.timestamp - timestamp for item in nearby])
    if np.ptp(times) <= 0.0:
        return None
    positions = np.array([
        gnss_antenna_position(item, reference_llh) for item in nearby
    ])
    design = np.column_stack((times, np.ones(len(times))))
    coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
    return coefficients[0]


def create_measurement_initial_state(
    timestamp: float,
    imu_measurement: ImuMeasurement,
    gnss_measurement: GnssMeasurement | None,
    gnss_measurements: list[GnssMeasurement],
    reference_llh: np.ndarray,
    lever_arm_body: np.ndarray,
    keyframe_interval: float,
    body_P_sensor: gtsam.Pose3 | None,
    fallback_state: NavigationState,
) -> NavigationState:
    """IMU 자세와 GNSS 위치/속도로 batch FGO의 initial state를 만든다."""
    rotation = body_rotation_from_imu(imu_measurement, body_P_sensor)
    if rotation is None:
        rotation = fallback_state.pose.rotation()

    position = np.asarray(fallback_state.pose.translation())
    if gnss_measurement is not None:
        antenna_position = gnss_antenna_position(gnss_measurement, reference_llh)
        position = antenna_position - rotation.matrix() @ lever_arm_body

    velocity = estimate_gnss_velocity(
        gnss_measurements,
        timestamp,
        reference_llh,
        window=keyframe_interval,
    )
    if velocity is None:
        velocity = fallback_state.velocity

    return NavigationState(
        timestamp=timestamp,
        pose=gtsam.Pose3(rotation, position),
        velocity=velocity,
        accel_bias=fallback_state.accel_bias,
        gyro_bias=fallback_state.gyro_bias,
    )


def build_graph(
    measurements: list[ImuMeasurement],
    keyframe_interval: float,
    gnss_measurements: list[GnssMeasurement] | None = None,
    reference_llh: np.ndarray | None = None,
    lever_arm_body: np.ndarray | None = None,
    imu_noise: ImuNoise | None = None,
    prior_noise: dict | None = None,
    gravity: float = 9.81,
    gnss_minimum_sigma: float = 0.1,
    body_P_sensor: gtsam.Pose3 | None = None,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values, int]:
    """짧은 구간으로 prior + IMU + bias + GNSS graph를 만든다."""
    if keyframe_interval <= 0.0:
        raise ValueError("keyframe_interval must be positive")

    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    used_gnss_indices: set[int] = set()

    if lever_arm_body is None:
        lever_arm_body = np.zeros(3)
    else:
        lever_arm_body = np.asarray(lever_arm_body, dtype=float)

    if gnss_measurements and reference_llh is None:
        first_gnss = gnss_measurements[0]
        reference_llh = np.array([
            first_gnss.latitude,
            first_gnss.longitude,
            first_gnss.altitude,
        ])

    current_state = create_initial_state(timestamp=measurements[0].timestamp)
    initial_gnss = None
    if gnss_measurements:
        initial_gnss = nearest_unused_gnss_measurement(
            gnss_measurements,
            measurements[0].timestamp,
            keyframe_interval,
            used_gnss_indices,
        )
        current_state = create_measurement_initial_state(
            measurements[0].timestamp,
            measurements[0],
            initial_gnss,
            gnss_measurements,
            reference_llh,
            lever_arm_body,
            keyframe_interval,
            body_P_sensor,
            current_state,
        )
    current_state.insert_into(initial_values, index=0)

    if prior_noise is None:
        prior_noise = {
            "rotation_sigma": [0.1, 0.1, 0.5],
            "position_sigma": [1.0, 1.0, 2.0],
            "velocity_sigma": 0.5,
            "accel_bias_sigma": [0.1, 0.1, 0.1],
            "gyro_bias_sigma": [0.01, 0.01, 0.01],
        }
    prior_noises = create_prior_noise(
        rotation_sigma=np.asarray(prior_noise["rotation_sigma"]),
        position_sigma=np.asarray(prior_noise["position_sigma"]),
        velocity_sigma=prior_noise["velocity_sigma"],
        accel_bias_sigma=np.asarray(prior_noise["accel_bias_sigma"]),
        gyro_bias_sigma=np.asarray(prior_noise["gyro_bias_sigma"]),
    )
    for factor in create_prior_factors(current_state, *prior_noises, index=0):
        graph.add(factor)

    if gnss_measurements:
        if initial_gnss is not None:
            graph.add(create_factor_from_measurement(
                StateKeys.at(0).pose,
                initial_gnss,
                reference_llh,
                lever_arm_body,
                gnss_minimum_sigma,
            ))

    # 아래 값은 실행 확인용 예시다. 실제 센서 datasheet/Allan variance로 교체한다.
    if imu_noise is None:
        imu_noise = ImuNoise(
            accelerometer_sigma=0.1,
            gyroscope_sigma=0.01,
            integration_sigma=1e-4,
            accelerometer_bias_sigma=1e-3,
            gyroscope_bias_sigma=1e-4,
        )
    params = create_preintegration_params(
        gravity=gravity,
        noise=imu_noise,
        body_P_sensor=body_P_sensor,
    )

    state_index = 0
    interval_measurements = [measurements[0]]
    interval_start = measurements[0].timestamp

    for measurement in measurements[1:]:
        interval_measurements.append(measurement)
        if measurement.timestamp - interval_start < keyframe_interval:
            continue

        # 현재 keyframe 구간의 모든 고주파 IMU 측정을 하나의 PIM으로 압축한다.
        pim = create_pim(params, current_state.bias)
        integrate_imu_measurements(interval_measurements, pim)

        graph.add(create_imu_factor(state_index, pim))
        zero_change, bias_noise = create_bias_random_walk(imu_noise, pim)
        graph.add(create_bias_factor(state_index, zero_change, bias_noise))

        # PIM 예측은 GNSS/IMU 측정 초기값이 없을 때의 fallback으로 사용한다.
        predicted_state = predict_next_state(current_state, pim)
        next_state = predicted_state
        keyframe_gnss = None
        if gnss_measurements:
            keyframe_gnss = nearest_unused_gnss_measurement(
                gnss_measurements,
                measurement.timestamp,
                keyframe_interval,
                used_gnss_indices,
            )
            next_state = create_measurement_initial_state(
                measurement.timestamp,
                measurement,
                keyframe_gnss,
                gnss_measurements,
                reference_llh,
                lever_arm_body,
                keyframe_interval,
                body_P_sensor,
                predicted_state,
            )
        next_state.insert_into(initial_values, index=state_index + 1)

        state_index += 1
        current_state = next_state

        if gnss_measurements:
            if keyframe_gnss is not None:
                graph.add(create_factor_from_measurement(
                    StateKeys.at(state_index).pose,
                    keyframe_gnss,
                    reference_llh,
                    lever_arm_body,
                    gnss_minimum_sigma,
                ))

        interval_measurements = [measurement]
        interval_start = measurement.timestamp

    return graph, initial_values, state_index


def optimize(
    graph: gtsam.NonlinearFactorGraph,
    initial_values: gtsam.Values,
) -> gtsam.Values:
    """Levenberg-Marquardt로 현재 batch graph를 최적화한다."""
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    return optimizer.optimize()


def keyframe_imu_measurements(
    measurements: list[ImuMeasurement],
    keyframe_interval: float,
) -> list[ImuMeasurement]:
    """graph 생성과 같은 기준으로 각 state의 IMU 측정을 고른다."""
    keyframes = [measurements[0]]
    interval_start = measurements[0].timestamp
    for measurement in measurements[1:]:
        if measurement.timestamp - interval_start < keyframe_interval:
            continue
        keyframes.append(measurement)
        interval_start = measurement.timestamp
    return keyframes


def save_position_angular_velocity_csv(
    output_path: Path,
    result: gtsam.Values,
    measurements: list[ImuMeasurement],
    keyframe_interval: float,
    last_index: int,
    body_P_sensor: gtsam.Pose3 | None,
) -> None:
    """optimized base_link 위치와 bias 보정 각속도를 CSV로 저장한다."""
    keyframes = keyframe_imu_measurements(measurements, keyframe_interval)
    if len(keyframes) != last_index + 1:
        raise ValueError("keyframe count does not match optimized state count")

    sensor_to_body = (
        np.eye(3)
        if body_P_sensor is None
        else body_P_sensor.rotation().matrix()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp_s",
            "elapsed_s",
            "position_east_m",
            "position_north_m",
            "position_up_m",
            "angular_velocity_body_x_rad_s",
            "angular_velocity_body_y_rad_s",
            "angular_velocity_body_z_rad_s",
            "roll",
            "pitch",
            "yaw"
        ])

        start_time = keyframes[0].timestamp
        for index, measurement in enumerate(keyframes):
            keys = StateKeys.at(index)
            position = np.asarray(result.atPose3(keys.pose).translation())
            rpy = np.asarray(result.atPose3(keys.pose).rotation().rpy())
            gyro_bias = result.atConstantBias(keys.bias).gyroscope()
            angular_velocity_body = sensor_to_body @ (
                measurement.angular_velocity - gyro_bias
            )
            writer.writerow([
                measurement.timestamp,
                measurement.timestamp - start_time,
                *position,
                *angular_velocity_body,
                *rpy
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    bag_path = Path(config["bag"]["path"])
    duration = config["bag"]["duration"]
    imu_topic = config["topics"]["imu"]
    gnss_topic = config["topics"]["gnss"]
    keyframe_interval = config["optimizer"]["keyframe_interval"]
    gravity = config["optimizer"]["gravity"]
    imu_noise = ImuNoise(**config["imu_noise"])
    body_P_sensor = create_body_P_sensor(config["imu_extrinsic"])
    lever_arm_body = np.asarray(config["gnss"]["lever_arm_body"], dtype=float)

    measurements = load_imu_segment(bag_path, imu_topic, duration)
    gnss_measurements = load_gnss_segment(
        bag_path,
        gnss_topic,
        measurements[0].timestamp,
        measurements[-1].timestamp,
    )
    if not gnss_measurements:
        raise ValueError("the selected bag segment contains no GNSS measurements")
    measurements, skipped_imu_count = trim_imu_to_gnss_start(
        measurements,
        gnss_measurements,
    )

    graph, initial_values, last_index = build_graph(
        measurements,
        keyframe_interval,
        gnss_measurements=gnss_measurements,
        lever_arm_body=lever_arm_body,
        imu_noise=imu_noise,
        prior_noise=config["prior_noise"],
        gravity=gravity,
        gnss_minimum_sigma=config["gnss"]["minimum_sigma"],
        body_P_sensor=body_P_sensor,
    )
    result = optimize(graph, initial_values)

    output_path = Path(
        config.get("output", {}).get(
            "trajectory_csv",
            "agr_fgo/output/trajectory.csv",
        )
    )
    save_position_angular_velocity_csv(
        output_path,
        result,
        measurements,
        keyframe_interval,
        last_index,
        body_P_sensor,
    )

    print(f"IMU measurements: {len(measurements)}")
    print(f"IMU samples skipped before first GNSS: {skipped_imu_count}")
    print(f"Graph start timestamp: {measurements[0].timestamp:.6f}")
    print(f"GNSS measurements: {len(gnss_measurements)}")
    print(f"States: {last_index + 1}")
    print(f"Factors: {graph.size()}")
    print(f"Initial error: {graph.error(initial_values):.6f}")
    print(f"Final error: {graph.error(result):.6f}")
    # print("Final pose:\n", result.atPose3(gtsam.symbol("x", last_index)).matrix())
    print("Final pose:\n", result.atPose3(gtsam.symbol("x", last_index)).translation())
    print("Final RPY (rad): ", result.atPose3(gtsam.symbol("x", last_index)).rotation().rpy())
    print("Final RPY (deg): ", np.rad2deg(result.atPose3(gtsam.symbol("x", last_index)).rotation().rpy()))
    print(f"Trajectory CSV: {output_path}")


if __name__ == "__main__":
    main()
