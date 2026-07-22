"""Prior noise models and prior factors for the first navigation state."""

from __future__ import annotations

import gtsam
import numpy as np

from states import NavigationState, StateKeys


Vector3 = np.ndarray
NoiseModel = gtsam.noiseModel.Base


def _validate_vector3_sigma(value: Vector3, name: str) -> Vector3:
    """표준편차 입력을 양수인 3차원 벡터로 검사한다."""
    sigma = np.asarray(value, dtype=float)
    if sigma.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {sigma.shape}")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    return sigma


def create_prior_noise(
    rotation_sigma: Vector3,
    position_sigma: Vector3,
    velocity_sigma: float,
    accel_bias_sigma: Vector3,
    gyro_bias_sigma: Vector3,
) -> tuple[NoiseModel, NoiseModel, NoiseModel]:
    """Pose, velocity, IMU bias의 prior noise model을 생성한다.

    Args:
        rotation_sigma: 회전 x/y/z 표준편차 [rad]. 작은 각도에서는
            roll/pitch/yaw 표준편차처럼 설정할 수 있다.
        position_sigma: 위치 x/y/z 표준편차 [m].
        velocity_sigma: 세 축에 동일하게 적용할 속도 표준편차 [m/s].
        accel_bias_sigma: 가속도계 bias 표준편차 [m/s^2].
        gyro_bias_sigma: 자이로 bias 표준편차 [rad/s].
    """
    rotation_sigma = _validate_vector3_sigma(rotation_sigma, "rotation_sigma")
    position_sigma = _validate_vector3_sigma(position_sigma, "position_sigma")
    accel_bias_sigma = _validate_vector3_sigma(
        accel_bias_sigma, "accel_bias_sigma"
    )
    gyro_bias_sigma = _validate_vector3_sigma(gyro_bias_sigma, "gyro_bias_sigma")

    if not np.isfinite(velocity_sigma) or velocity_sigma <= 0.0:
        raise ValueError("velocity_sigma must be a positive finite number")

    # Pose3의 local 좌표 순서는 [회전 x,y,z, 위치 x,y,z]이다.
    pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.concatenate((rotation_sigma, position_sigma))
    )
    velocity_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, velocity_sigma)

    # ConstantBias의 순서는 [가속도계 bias 3축, 자이로 bias 3축]이다.
    bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.concatenate((accel_bias_sigma, gyro_bias_sigma))
    )
    return pose_prior_noise, velocity_prior_noise, bias_prior_noise


def create_prior_factors(
    state: NavigationState,
    pose_prior_noise: NoiseModel,
    velocity_prior_noise: NoiseModel,
    bias_prior_noise: NoiseModel,
    index: int = 0,
) -> tuple[
    gtsam.PriorFactorPose3,
    gtsam.PriorFactorVector,
    gtsam.PriorFactorConstantBias,
]:
    """하나의 navigation state를 고정하는 세 prior factor를 생성한다.

    보통 ``index=0``으로 사용하지만, sliding-window 재초기화 등을 위해
    특정 state index에도 prior를 걸 수 있도록 index를 인자로 받는다.
    이 함수는 factor만 반환하며 graph와 initial values는 변경하지 않는다.
    """
    keys = StateKeys.at(index)

    pose_prior_factor = gtsam.PriorFactorPose3(
        keys.pose, state.pose, pose_prior_noise
    )
    # Python GTSAM 이름은 대문자 F를 사용한 PriorFactorVector이다.
    velocity_prior_factor = gtsam.PriorFactorVector(
        keys.velocity, state.velocity, velocity_prior_noise
    )
    bias_prior_factor = gtsam.PriorFactorConstantBias(
        keys.bias, state.bias, bias_prior_noise
    )
    return pose_prior_factor, velocity_prior_factor, bias_prior_factor
