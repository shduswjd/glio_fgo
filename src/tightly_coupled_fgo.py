"""Batch tightly-coupled IMU + GPS pseudorange + Doppler FGO for tc.yaml."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gtsam
import numpy as np
import yaml

from factors.clock import (
    beidou_isb_key, clock_bias_key, clock_drift_key,
    create_beidou_isb_random_walk_factor, create_clock_dynamics_factor,
    create_isb_random_walk_factor, galileo_isb_key,
)
from factors.gnss_raw_factors import create_doppler_factor, create_raw_pseudorange_factor
from factors.imu_preintegration import (
    ImuNoise, create_bias_factor, create_bias_random_walk, create_imu_factor,
    create_pim, create_preintegration_params, integrate_imu_measurements,
    predict_next_state,
)
from factors.prior import create_prior_factors, create_prior_noise
from factors.troposphere import (
    create_wet_delay_random_walk_factor, wet_delay_key,
)
from run_tc_gnss import configured_signals, estimate_receiver_position
from sensors.gnss_corrections import ecef_R_enu
from sensors.gnss_preprocessor import prepare_epoch
from sensors.gnss_raw import load_tc_config, read_rinex_observations, read_tc_navigation
from sensors.gnss_satellite import EphemerisStore, SPEED_OF_LIGHT
from sensors.imu import ImuMeasurement, read_imu_txt
from states import NavigationState, StateKeys


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "tightly_coupled" not in config:
        raise ValueError("config requires a tightly_coupled section")
    return config


def _load_imu(path: Path, options: dict) -> list[ImuMeasurement]:
    return list(read_imu_txt(
        path,
        timestamp_column=int(options["timestamp_column"]),
        acceleration_columns=tuple(options["accel_columns"]),
        angular_velocity_columns=tuple(options["gyro_columns"]),
        angular_velocity_scale=float(options.get("angular_velocity_scale", 1.0)),
        skip_rows=int(options.get("skip_rows", 0)),
    ))


def _overlapping_epochs(config_path: Path, imu: list[ImuMeasurement], duration: float | None):
    dataset = load_tc_config(config_path)
    selected = []
    start = None
    for epoch in read_rinex_observations(dataset.observation_path):
        epoch_time = epoch.gps_seconds_of_week
        if epoch_time < imu[0].timestamp:
            continue
        if epoch_time > imu[-1].timestamp:
            break
        if start is None:
            start = epoch_time
        if duration is not None and epoch_time - start > duration:
            break
        selected.append(epoch)
    return selected


def _nearest_imu_index(imu: list[ImuMeasurement], timestamp: float, start: int = 0) -> int:
    times = np.fromiter((item.timestamp for item in imu[start:]), dtype=float)
    return start + int(np.argmin(np.abs(times - timestamp)))


def _body_P_sensor(options: dict) -> gtsam.Pose3:
    """Return the IMU sensor pose expressed in the estimator body frame."""
    quaternion = np.asarray(options.get("quaternion_xyzw", [0, 0, 0, 1]), dtype=float)
    translation = np.asarray(options.get("translation_m", [0, 0, 0]), dtype=float)
    if quaternion.shape != (4,) or translation.shape != (3,):
        raise ValueError("imu_extrinsic requires quaternion_xyzw[4] and translation_m[3]")
    norm = np.linalg.norm(quaternion)
    if not np.all(np.isfinite(quaternion)) or not np.all(np.isfinite(translation)) or norm == 0:
        raise ValueError("imu_extrinsic values must be finite and quaternion non-zero")
    qx, qy, qz, qw = quaternion / norm
    return gtsam.Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz), translation)


def _stationary_initialization(
    imu: list[ImuMeasurement], start_index: int, body_R_sensor: np.ndarray,
    duration_s: float, gravity: float,
) -> tuple[float, float, np.ndarray]:
    """Estimate level attitude and gyro bias from the initial quiet interval."""
    start_time = imu[start_index].timestamp
    samples = [
        item for item in imu[start_index:]
        if item.timestamp - start_time <= duration_s
    ]
    if len(samples) < 2:
        return 0.0, 0.0, np.zeros(3)
    acceleration = np.array([body_R_sensor @ item.acceleration for item in samples])
    angular_rate = np.array([body_R_sensor @ item.angular_velocity for item in samples])
    # Reject moving samples before taking robust component-wise medians.
    quiet = (
        np.abs(np.linalg.norm(acceleration, axis=1) - gravity) < 0.35
    ) & (np.linalg.norm(angular_rate, axis=1) < np.deg2rad(1.0))
    if np.count_nonzero(quiet) < max(10, len(samples) // 2):
        return 0.0, 0.0, np.zeros(3)
    mean_acceleration = np.median(acceleration[quiet], axis=0)
    # Gyro output is quantized on this dataset; the median collapses the small
    # z-bias to exactly zero. The quiet-sample mean preserves the sub-LSB bias.
    gyro_bias = np.mean(angular_rate[quiet], axis=0)
    roll = float(np.arctan2(mean_acceleration[1], mean_acceleration[2]))
    pitch = float(np.arctan2(
        -mean_acceleration[0],
        np.hypot(mean_acceleration[1], mean_acceleration[2]),
    ))
    return roll, pitch, gyro_bias


def _consistent_course_yaw(
    velocities_local: list[np.ndarray], minimum_speed: float, tolerance_deg: float,
    required_samples: int,
) -> tuple[float, int] | None:
    """Find the first stable GNSS course, expressed as ENU yaw from East."""
    headings: list[float] = []
    tolerance = np.deg2rad(tolerance_deg)
    for velocity_index, velocity in enumerate(velocities_local):
        if np.linalg.norm(velocity[:2]) < minimum_speed:
            headings.clear()
            continue
        heading = float(np.arctan2(velocity[1], velocity[0]))
        headings.append(heading)
        headings = headings[-required_samples:]
        if len(headings) == required_samples:
            mean = float(np.arctan2(
                np.mean(np.sin(headings)), np.mean(np.cos(headings))
            ))
            errors = np.arctan2(np.sin(np.asarray(headings) - mean),
                                np.cos(np.asarray(headings) - mean))
            if np.max(np.abs(errors)) <= tolerance:
                return mean, velocity_index
    return None


def estimate_velocity_and_clock_drift(measurements, receiver_ecef: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve ECEF receiver velocity and clock drift from Doppler observations."""
    rows, values = [], []
    for measurement in measurements:
        if measurement.range_rate_mps is None:
            continue
        satellite = measurement.satellite_state
        delta = satellite.position_ecef - receiver_ecef
        line_of_sight = delta / np.linalg.norm(delta)
        rows.append(np.r_[-line_of_sight, 1.0])
        values.append(
            measurement.range_rate_mps
            - line_of_sight @ satellite.velocity_ecef
            + SPEED_OF_LIGHT * satellite.clock_drift_sps
        )
    if len(rows) < 4:
        return np.zeros(3), 0.0
    solution, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(values), rcond=None)
    return solution[:3], float(solution[3])


def _measurement_sigma(raw: dict, measurement) -> float:
    configured = raw["pseudorange_sigma_m"]
    if isinstance(configured, dict):
        key = (
            f"C_{measurement.orbit_type}"
            if measurement.satellite.startswith("C") else measurement.satellite[0]
        )
        base = float(configured[key])
    else:
        base = float(configured)
    weighting = raw.get("weighting", {})
    elevation_floor = float(weighting.get("elevation_sin_floor", 1.0))
    elevation_scale = 1.0
    if measurement.elevation_rad is not None:
        elevation_scale = 1.0 / max(
            float(np.sin(measurement.elevation_rad)), elevation_floor
        )
    cn0_scale = 1.0
    if measurement.cn0_dbhz is not None:
        reference = float(weighting.get("cn0_reference_dbhz", measurement.cn0_dbhz))
        cn0_scale = min(
            10.0 ** ((reference - measurement.cn0_dbhz) / 20.0),
            float(weighting.get("cn0_max_scale", 1.0)),
        )
        cn0_scale = max(cn0_scale, 1.0)
    return base * elevation_scale * cn0_scale


def build_tightly_coupled_graph(
    config_path: Path, beidou_orbits_override: set[str] | None = None,
):
    config = _load_yaml(config_path)
    dataset = load_tc_config(config_path)
    tc = config["tightly_coupled"]
    raw = config.get("gnss_raw", {})
    imu = _load_imu(dataset.imu_path, tc["imu"])
    epochs = _overlapping_epochs(config_path, imu, dataset.duration)
    if len(epochs) < 2:
        raise ValueError("fewer than two GNSS epochs overlap the IMU time range")

    ephemerides = EphemerisStore.from_rinex(read_tc_navigation(config_path))
    signals = configured_signals(raw)
    signal, secondary_signal = signals.get("G", ("1C", None))
    use_galileo = "E" in signals
    use_beidou = "C" in signals
    wet_delay_options = tc.get("wet_delay", {})
    use_wet_delay = bool(wet_delay_options.get("enabled", False))
    beidou_orbits = (
        beidou_orbits_override
        if beidou_orbits_override is not None
        else {str(item) for item in raw.get(
            "beidou_orbits", ["MEO", "IGSO", "GEO"]
        )}
    )
    weighting = raw.get("weighting", {})
    minimum_cn0 = weighting.get("minimum_cn0_dbhz")
    minimum_cn0 = None if minimum_cn0 is None else float(minimum_cn0)
    residual_gate = weighting.get("residual_gate_m")
    residual_gate = None if residual_gate is None else float(residual_gate)
    reference_ecef, initial_clock = estimate_receiver_position(
        epochs[0], ephemerides, signal, secondary_signal
    )
    frame_rotation = ecef_R_enu(reference_ecef)
    lever = np.asarray(tc["lever_arm_body_m"], dtype=float)
    imu_noise = ImuNoise(**tc["imu_noise"])
    gravity = float(tc["gravity_mps2"])
    body_P_sensor = _body_P_sensor(tc.get("imu_extrinsic", {}))
    body_R_sensor = body_P_sensor.rotation().matrix()
    params = create_preintegration_params(
        gravity, imu_noise, body_P_sensor=body_P_sensor,
    )
    prior = tc["prior_noise"]

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    first_prepared = prepare_epoch(
        epochs[0], reference_ecef, ephemerides, signal=signal,
        secondary_signal=secondary_signal,
        minimum_elevation_deg=float(raw["minimum_elevation_deg"]),
        signals=signals,
        beidou_orbits=beidou_orbits,
        minimum_cn0_dbhz=minimum_cn0, residual_gate_m=residual_gate,
    )
    velocity_ecef, initial_drift = estimate_velocity_and_clock_drift(first_prepared, reference_ecef)
    initial_imu_cursor = _nearest_imu_index(imu, epochs[0].gps_seconds_of_week)
    alignment = tc.get("initial_alignment", {})
    initial_roll, initial_pitch, initial_gyro_bias = _stationary_initialization(
        imu, initial_imu_cursor, body_R_sensor,
        float(alignment.get("stationary_duration_s", 10.0)), gravity,
    )
    course_velocities = [frame_rotation.T @ velocity_ecef]
    search_duration = float(alignment.get("course_search_duration_s", 90.0))
    for alignment_epoch in epochs[1:]:
        if alignment_epoch.gps_seconds_of_week - epochs[0].gps_seconds_of_week > search_duration:
            break
        try:
            alignment_ecef, _ = estimate_receiver_position(
                alignment_epoch, ephemerides, signal, secondary_signal
            )
            alignment_prepared = prepare_epoch(
                alignment_epoch, alignment_ecef, ephemerides, signal=signal,
                secondary_signal=secondary_signal,
                minimum_elevation_deg=float(raw["minimum_elevation_deg"]),
                signals=signals, beidou_orbits=beidou_orbits,
                minimum_cn0_dbhz=minimum_cn0, residual_gate_m=residual_gate,
            )
            alignment_velocity_ecef, _ = estimate_velocity_and_clock_drift(
                alignment_prepared, alignment_ecef
            )
            course_velocities.append(frame_rotation.T @ alignment_velocity_ecef)
        except ValueError:
            course_velocities.append(np.zeros(3))
    course_alignment = _consistent_course_yaw(
        course_velocities,
        float(alignment.get("minimum_course_speed_mps", 2.0)),
        float(alignment.get("course_tolerance_deg", 12.0)),
        int(alignment.get("required_course_samples", 3)),
    )
    if course_alignment is None:
        initial_yaw = float(alignment.get("fallback_yaw_deg", 0.0)) * np.pi / 180.0
    else:
        course_yaw, course_epoch_index = course_alignment
        course_time = epochs[course_epoch_index].gps_seconds_of_week
        end_imu = _nearest_imu_index(imu, course_time, initial_imu_cursor)
        integrated_body_yaw = 0.0
        for previous, following in zip(
            imu[initial_imu_cursor:end_imu],
            imu[initial_imu_cursor + 1:end_imu + 1],
        ):
            body_rate = body_R_sensor @ previous.angular_velocity - initial_gyro_bias
            integrated_body_yaw += body_rate[2] * (following.timestamp - previous.timestamp)
        initial_yaw = float(np.arctan2(
            np.sin(course_yaw - integrated_body_yaw),
            np.cos(course_yaw - integrated_body_yaw),
        ))
    print(
        "initial alignment [deg]: "
        f"roll={np.rad2deg(initial_roll):.3f} "
        f"pitch={np.rad2deg(initial_pitch):.3f} "
        f"yaw={np.rad2deg(initial_yaw):.3f}; "
        f"gyro bias [deg/s]={np.rad2deg(initial_gyro_bias)}",
        flush=True,
    )
    initial_rotation = gtsam.Rot3.Ypr(initial_yaw, initial_pitch, initial_roll)
    initial_body_position = -initial_rotation.matrix() @ lever
    current = NavigationState(
        timestamp=epochs[0].gps_seconds_of_week,
        pose=gtsam.Pose3(initial_rotation, initial_body_position),
        velocity=frame_rotation.T @ velocity_ecef,
        gyro_bias=initial_gyro_bias,
    )
    current_clock_bias = initial_clock
    current_clock_drift = initial_drift
    current.insert_into(values, 0)
    values.insert(clock_bias_key(0), initial_clock)
    values.insert(clock_drift_key(0), initial_drift)
    if use_galileo:
        values.insert(galileo_isb_key(0), 0.0)
    if use_beidou:
        values.insert(beidou_isb_key(0), 0.0)
    if use_wet_delay:
        values.insert(wet_delay_key(0), 0.0)
    navigation_priors = create_prior_noise(
        np.asarray(prior["rotation_sigma_rad"]),
        np.asarray(prior["position_sigma_m"]),
        float(prior["velocity_sigma_mps"]),
        np.asarray(prior["accel_bias_sigma_mps2"]),
        np.asarray(prior["gyro_bias_sigma_radps"]),
    )
    for factor in create_prior_factors(current, *navigation_priors, index=0):
        graph.add(factor)
    graph.add(gtsam.PriorFactorDouble(
        clock_bias_key(0), initial_clock,
        gtsam.noiseModel.Isotropic.Sigma(1, float(prior["clock_bias_sigma_m"])),
    ))
    if use_galileo:
        graph.add(gtsam.PriorFactorDouble(
            galileo_isb_key(0), 0.0,
            gtsam.noiseModel.Isotropic.Sigma(
                1, float(tc["galileo_isb"]["initial_sigma_m"])
            ),
        ))
    if use_beidou:
        graph.add(gtsam.PriorFactorDouble(
            beidou_isb_key(0), 0.0,
            gtsam.noiseModel.Isotropic.Sigma(
                1, float(tc["beidou_isb"]["initial_sigma_m"])
            ),
        ))
    if use_wet_delay:
        graph.add(gtsam.PriorFactorDouble(
            wet_delay_key(0), 0.0,
            gtsam.noiseModel.Isotropic.Sigma(
                1, float(wet_delay_options["initial_sigma_m"])
            ),
        ))
    graph.add(gtsam.PriorFactorDouble(
        clock_drift_key(0), initial_drift,
        gtsam.noiseModel.Isotropic.Sigma(1, float(prior["clock_drift_sigma_mps"])),
    ))

    doppler_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(1, float(raw["doppler_sigma_mps"])),
    )

    def add_gnss(index: int, prepared) -> None:
        keys = StateKeys.at(index)
        for measurement in prepared:
            pseudorange_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                gtsam.noiseModel.Isotropic.Sigma(
                    1, _measurement_sigma(raw, measurement)
                ),
            )
            graph.add(create_raw_pseudorange_factor(
                keys.pose, clock_bias_key(index), measurement,
                pseudorange_noise, lever, reference_ecef, frame_rotation,
                (galileo_isb_key(index)
                 if measurement.satellite.startswith("E") else
                 beidou_isb_key(index)
                 if measurement.satellite.startswith("C") else None),
                (wet_delay_key(index) if use_wet_delay else None),
            ))
            if measurement.range_rate_mps is not None:
                graph.add(create_doppler_factor(
                    keys.pose, keys.velocity, clock_drift_key(index), measurement,
                    doppler_noise, lever, reference_ecef, frame_rotation,
                ))

    add_gnss(0, first_prepared)
    imu_cursor = initial_imu_cursor
    prepared_epochs = [first_prepared]
    for index, epoch in enumerate(epochs[1:], start=1):
        end_cursor = _nearest_imu_index(imu, epoch.gps_seconds_of_week, imu_cursor + 1)
        segment = imu[imu_cursor:end_cursor + 1]
        if len(segment) < 2:
            raise ValueError(f"no IMU interval for GNSS epoch {epoch.time.isoformat()}")
        pim = create_pim(params, current.bias)
        integrate_imu_measurements(segment, pim)
        graph.add(create_imu_factor(index - 1, pim))
        zero_bias, bias_noise = create_bias_random_walk(imu_noise, pim)
        graph.add(create_bias_factor(index - 1, zero_bias, bias_noise))
        graph.add(create_clock_dynamics_factor(
            index - 1, pim.deltaTij(),
            float(tc["clock_process"]["bias_sigma_m_sqrt_s"]),
            float(tc["clock_process"]["drift_sigma_mps_sqrt_s"]),
        ))
        if use_galileo:
            graph.add(create_isb_random_walk_factor(
                index - 1, pim.deltaTij(),
                float(tc["galileo_isb"]["random_walk_sigma_m_sqrt_s"]),
            ))
        if use_beidou:
            graph.add(create_beidou_isb_random_walk_factor(
                index - 1, pim.deltaTij(),
                float(tc["beidou_isb"]["random_walk_sigma_m_sqrt_s"]),
            ))
        if use_wet_delay:
            graph.add(create_wet_delay_random_walk_factor(
                index - 1, pim.deltaTij(),
                float(wet_delay_options["random_walk_sigma_m_sqrt_s"]),
            ))

        predicted = predict_next_state(current, pim)
        # SPP is used only to seed each new state. Urban blockage can leave an
        # epoch with fewer than four dual-frequency GPS codes; in that case the
        # IMU/clock prediction is a better seed than aborting the whole graph.
        predicted_antenna_local = (
            np.asarray(predicted.pose.translation())
            + predicted.pose.rotation().matrix() @ lever
        )
        predicted_receiver_ecef = reference_ecef + frame_rotation @ predicted_antenna_local
        predicted_clock_bias = current_clock_bias + current_clock_drift * pim.deltaTij()
        try:
            receiver_ecef, clock_bias = estimate_receiver_position(
                epoch, ephemerides, signal, secondary_signal
            )
        except ValueError as exc:
            if "at least four GPS pseudoranges" not in str(exc):
                raise
            receiver_ecef = predicted_receiver_ecef
            clock_bias = predicted_clock_bias
        prepared = prepare_epoch(
            epoch, receiver_ecef, ephemerides, signal=signal,
            secondary_signal=secondary_signal,
            minimum_elevation_deg=float(raw["minimum_elevation_deg"]),
            signals=signals,
            beidou_orbits=beidou_orbits,
            minimum_cn0_dbhz=minimum_cn0, residual_gate_m=residual_gate,
        )
        velocity_ecef, clock_drift = estimate_velocity_and_clock_drift(prepared, receiver_ecef)
        antenna_local = frame_rotation.T @ (receiver_ecef - reference_ecef)
        body_local = antenna_local - predicted.pose.rotation().matrix() @ lever
        current = NavigationState(
            timestamp=segment[-1].timestamp,
            pose=gtsam.Pose3(predicted.pose.rotation(), body_local),
            velocity=frame_rotation.T @ velocity_ecef,
            accel_bias=predicted.accel_bias, gyro_bias=predicted.gyro_bias,
        )
        current.insert_into(values, index)
        values.insert(clock_bias_key(index), clock_bias)
        values.insert(clock_drift_key(index), clock_drift)
        if use_galileo:
            values.insert(galileo_isb_key(index), 0.0)
        if use_beidou:
            values.insert(beidou_isb_key(index), 0.0)
        if use_wet_delay:
            values.insert(wet_delay_key(index), 0.0)
        current_clock_bias = clock_bias
        current_clock_drift = clock_drift
        stabilization = tc.get("numerical_stabilization", {})
        pose_sigmas = np.array(
            [float(stabilization.get("rotation_sigma_rad", 3.0))] * 3
            + [float(stabilization.get("position_sigma_m", 10000.0))] * 3
        )
        graph.add(gtsam.PriorFactorPose3(
            StateKeys.at(index).pose, current.pose,
            gtsam.noiseModel.Diagonal.Sigmas(pose_sigmas),
        ))
        graph.add(gtsam.PriorFactorVector(
            StateKeys.at(index).velocity, current.velocity,
            gtsam.noiseModel.Isotropic.Sigma(
                3, float(stabilization.get("velocity_sigma_mps", 100.0))
            ),
        ))
        bias_sigmas = np.array(
            [float(stabilization.get("accel_bias_sigma_mps2", 10.0))] * 3
            + [float(stabilization.get("gyro_bias_sigma_radps", 1.0))] * 3
        )
        graph.add(gtsam.PriorFactorConstantBias(
            StateKeys.at(index).bias,
            gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
            gtsam.noiseModel.Diagonal.Sigmas(bias_sigmas),
        ))
        graph.add(gtsam.PriorFactorDouble(
            clock_drift_key(index), clock_drift,
            gtsam.noiseModel.Isotropic.Sigma(
                1, float(stabilization.get("clock_drift_sigma_mps", 100.0))
            ),
        ))
        add_gnss(index, prepared)
        prepared_epochs.append(prepared)
        imu_cursor = end_cursor

    return graph, values, epochs, reference_ecef, frame_rotation


def _save_result(
    path: Path, result: gtsam.Values, epochs, reference_ecef, frame_rotation,
    lever_arm_body: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "gps_time_s", "east_m", "north_m", "up_m",
            "ecef_x_m", "ecef_y_m", "ecef_z_m",
            "antenna_ecef_x_m", "antenna_ecef_y_m", "antenna_ecef_z_m",
            "roll", "pitch", "yaw", "clock_bias_m", "clock_drift_mps",
            "galileo_isb_m", "beidou_isb_m", "wet_delay_delta_m",
        ])
        for index, epoch in enumerate(epochs):
            pose = result.atPose3(StateKeys.at(index).pose)
            local = np.asarray(pose.translation())
            ecef = reference_ecef + frame_rotation @ local
            antenna_local = local + pose.rotation().matrix() @ lever_arm_body
            antenna_ecef = reference_ecef + frame_rotation @ antenna_local
            writer.writerow([
                epoch.gps_seconds, *local, *ecef, *antenna_ecef,
                *pose.rotation().rpy(),
                result.atDouble(clock_bias_key(index)),
                result.atDouble(clock_drift_key(index)),
                (result.atDouble(galileo_isb_key(index))
                 if result.exists(galileo_isb_key(index)) else float("nan")),
                (result.atDouble(beidou_isb_key(index))
                 if result.exists(beidou_isb_key(index)) else float("nan")),
                (result.atDouble(wet_delay_key(index))
                 if result.exists(wet_delay_key(index)) else float("nan")),
            ])


def optimize_incrementally(
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    state_count: int,
    relinearize_threshold: float = 0.01,
    relinearize_skip: int = 1,
    final_update_iterations: int = 20,
) -> gtsam.Values:
    """Add one epoch at a time so full data is not one difficult LM step."""
    factors_by_epoch: list[list[gtsam.NonlinearFactor]] = [
        [] for _ in range(state_count)
    ]
    for factor_index in range(graph.size()):
        factor = graph.at(factor_index)
        epoch_index = max(gtsam.Symbol(key).index() for key in factor.keys())
        factors_by_epoch[epoch_index].append(factor)

    parameters = gtsam.ISAM2Params()
    parameters.setRelinearizeThreshold(float(relinearize_threshold))
    parameters.relinearizeSkip = int(relinearize_skip)
    # QR is slower than Cholesky but is safer for weakly observable attitude
    # and IMU-bias directions during the beginning of a GNSS/IMU run.
    parameters.setFactorization("QR")
    isam = gtsam.ISAM2(parameters)

    for index, factors in enumerate(factors_by_epoch):
        new_graph = gtsam.NonlinearFactorGraph()
        for factor in factors:
            new_graph.add(factor)
        new_values = gtsam.Values()
        keys = StateKeys.at(index)
        new_values.insert(keys.pose, initial.atPose3(keys.pose))
        new_values.insert(keys.velocity, initial.atVector(keys.velocity))
        new_values.insert(keys.bias, initial.atConstantBias(keys.bias))
        new_values.insert(clock_bias_key(index), initial.atDouble(clock_bias_key(index)))
        new_values.insert(clock_drift_key(index), initial.atDouble(clock_drift_key(index)))
        if initial.exists(galileo_isb_key(index)):
            new_values.insert(
                galileo_isb_key(index), initial.atDouble(galileo_isb_key(index))
            )
        if initial.exists(beidou_isb_key(index)):
            new_values.insert(
                beidou_isb_key(index), initial.atDouble(beidou_isb_key(index))
            )
        if initial.exists(wet_delay_key(index)):
            new_values.insert(
                wet_delay_key(index), initial.atDouble(wet_delay_key(index))
            )
        isam.update(new_graph, new_values)
    # Adding each epoch once is not sufficient for a long, nonlinear
    # GNSS/IMU graph. Empty updates trigger additional relinearization and
    # back-substitution passes over variables whose estimates moved.
    for _ in range(int(final_update_iterations)):
        isam.update()
    return isam.calculateEstimate()


def main() -> None:
    default = Path(__file__).resolve().parents[1] / "config" / "tc.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument(
        "--beidou-orbits", nargs="+", choices=("MEO", "IGSO", "GEO"),
        help="override gnss_raw.beidou_orbits for an A/B run",
    )
    parser.add_argument("--output", type=Path, help="override output CSV path")
    args = parser.parse_args()
    config = _load_yaml(args.config)
    orbit_override = None if args.beidou_orbits is None else set(args.beidou_orbits)
    graph, initial, epochs, reference, rotation = build_tightly_coupled_graph(
        args.config, orbit_override
    )
    initial_error = graph.error(initial)
    print(
        f"states={len(epochs)} factors={graph.size()} "
        f"initial_error={initial_error:.3f}",
        flush=True,
    )
    optimizer_name = str(config["tightly_coupled"].get("optimizer", "isam2"))
    if optimizer_name == "isam2":
        result = optimize_incrementally(
            graph, initial, len(epochs),
            config["tightly_coupled"].get("relinearize_threshold", 0.01),
            config["tightly_coupled"].get("relinearize_skip", 1),
            config["tightly_coupled"].get("final_update_iterations", 20),
        )
    elif optimizer_name == "batch_lm":
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    else:
        raise ValueError("tightly_coupled.optimizer must be 'isam2' or 'batch_lm'")
    final_error = graph.error(result)
    if not np.isfinite(final_error) or final_error >= initial_error:
        raise RuntimeError(
            "optimization did not reduce graph error "
            f"({initial_error:.3f} -> {final_error:.3f}). "
            "Use a finite data.duration (start with 10-30 s). Full/null data "
            "requires a fixed-lag smoother with marginalization, which this "
            "batch prototype does not yet implement."
        )
    output = (
        args.output.resolve()
        if args.output is not None
        else (args.config.resolve().parent / config["tightly_coupled"]["output_csv"]).resolve()
    )
    lever = np.asarray(config["tightly_coupled"]["lever_arm_body_m"], dtype=float)
    _save_result(output, result, epochs, reference, rotation, lever)
    print(f"final_error={final_error:.3f}")
    print(f"trajectory={output}")


if __name__ == "__main__":
    main()
