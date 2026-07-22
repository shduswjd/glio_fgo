"""GTSAM IMU preintegration and factor construction utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gtsam
import numpy as np

from sensors.imu import ImuMeasurement
from states import NavigationState, StateKeys


@dataclass(frozen=True)
class ImuNoise:
    """IMU white-noise and bias random-walk standard deviations."""

    accelerometer_sigma: float
    gyroscope_sigma: float
    integration_sigma: float
    accelerometer_bias_sigma: float
    gyroscope_bias_sigma: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a positive finite number") from exc
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, value)


def create_preintegration_params(
    gravity: float,
    noise: ImuNoise,
    body_P_sensor: gtsam.Pose3 | None = None,
) -> gtsam.PreintegrationParams:
    """Create preintegration parameters for a z-up local world frame."""
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity must be a positive finite number")

    params = gtsam.PreintegrationParams.MakeSharedU(gravity)
    params.setAccelerometerCovariance(
        np.eye(3) * noise.accelerometer_sigma**2
    )
    params.setGyroscopeCovariance(np.eye(3) * noise.gyroscope_sigma**2)
    params.setIntegrationCovariance(np.eye(3) * noise.integration_sigma**2)

    if body_P_sensor is None:
        body_P_sensor = gtsam.Pose3()
    params.setBodyPSensor(body_P_sensor)
    return params


def create_pim(
    params: gtsam.PreintegrationParams,
    bias: gtsam.imuBias.ConstantBias,
) -> gtsam.PreintegratedImuMeasurements:
    """Create an empty PIM linearized at ``bias``."""
    return gtsam.PreintegratedImuMeasurements(params, bias)


def integrate_imu_interval(
    previous_imu: ImuMeasurement,
    current_imu: ImuMeasurement,
    pim: gtsam.PreintegratedImuMeasurements,
) -> None:
    """Integrate one interval ``[previous_imu, current_imu]`` into a PIM."""
    dt = current_imu.timestamp - previous_imu.timestamp
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("IMU timestamps must increase")

    # Zero-order hold: use the previous sample over this time interval.
    pim.integrateMeasurement(
        previous_imu.acceleration,
        previous_imu.angular_velocity,
        dt,
    )


def integrate_imu_measurements(
    imu_measurements: Sequence[ImuMeasurement],
    pim: gtsam.PreintegratedImuMeasurements,
) -> None:
    """Integrate all adjacent IMU intervals between two keyframes."""
    if len(imu_measurements) < 2:
        raise ValueError("at least two IMU measurements are required")

    for previous, current in zip(imu_measurements[:-1], imu_measurements[1:]):
        integrate_imu_interval(previous, current, pim)


def predict_next_state(
    state: NavigationState,
    pim: gtsam.PreintegratedImuMeasurements,
) -> NavigationState:
    """Create the next optimizer initial value using the integrated IMU data."""
    if pim.deltaTij() <= 0.0:
        raise ValueError("PIM contains no integrated IMU measurements")

    nav_state = gtsam.NavState(state.pose, state.velocity)
    predicted = pim.predict(nav_state, state.bias)

    return NavigationState(
        timestamp=state.timestamp + pim.deltaTij(),
        pose=predicted.pose(),
        velocity=predicted.velocity(),
        accel_bias=state.accel_bias,
        gyro_bias=state.gyro_bias,
    )


def create_imu_factor(
    index: int,
    pim: gtsam.PreintegratedImuMeasurements,
) -> gtsam.ImuFactor:
    """Create the IMU factor connecting state ``index`` to ``index + 1``."""
    previous = StateKeys.at(index)
    current = StateKeys.at(index + 1)
    return gtsam.ImuFactor(
        previous.pose,
        previous.velocity,
        current.pose,
        current.velocity,
        previous.bias,
        pim,
    )


def create_bias_random_walk(
    noise: ImuNoise,
    pim: gtsam.PreintegratedImuMeasurements,
) -> tuple[gtsam.imuBias.ConstantBias, gtsam.noiseModel.Base]:
    """Create the zero change and noise model for a bias random walk."""
    total_dt = pim.deltaTij()
    if total_dt <= 0.0:
        raise ValueError("PIM contains no integrated IMU measurements")

    bias_sigmas = np.array(
        [noise.accelerometer_bias_sigma] * 3
        + [noise.gyroscope_bias_sigma] * 3
    ) * np.sqrt(total_dt)

    zero_bias_change = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    bias_noise = gtsam.noiseModel.Diagonal.Sigmas(bias_sigmas)
    return zero_bias_change, bias_noise


def create_bias_factor(
    index: int,
    zero_bias_change: gtsam.imuBias.ConstantBias,
    bias_noise: gtsam.noiseModel.Base,
) -> gtsam.BetweenFactorConstantBias:
    """Create the bias factor connecting ``B(index)`` to ``B(index + 1)``."""
    previous = StateKeys.at(index)
    current = StateKeys.at(index + 1)
    return gtsam.BetweenFactorConstantBias(
        previous.bias,
        current.bias,
        zero_bias_change,
        bias_noise,
    )
