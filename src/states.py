"""State definitions used by the agricultural sensor-fusion graph.

The initial navigation state follows the common IMU FGO layout::

    x_k = {T_world_body, v_world, b_acc, b_gyro}

Sensor measurements and factors should live in their own modules.  This file is
only responsible for defining state variables and mapping them to GTSAM keys.
주로 state는 IMU preintegration 값을 넣는다
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, L, V, X


Vector3 = np.ndarray


def _vector3(value: Vector3, name: str) -> Vector3:
    """Return ``value`` as an independent three-dimensional float vector."""
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    return array.copy()


@dataclass(frozen=True)
class StateKeys:
    """GTSAM variable keys belonging to one fusion epoch. 인덱스를 다루기 위한거임"""

    pose: int
    velocity: int
    bias: int

    @classmethod
    def at(cls, index: int) -> "StateKeys":
        if index < 0:
            raise ValueError("state index must be non-negative")
        return cls(pose=X(index), velocity=V(index), bias=B(index))


def landmark_key(index: int) -> int:
    """Return a key reserved for an optional LiDAR landmark state."""
    if index < 0:
        raise ValueError("landmark index must be non-negative")
    return L(index)


@dataclass
class NavigationState:
    """Platform state at one timestamp.
    해당 인덱스에 있는 상태값
    Conventions:
        pose: ``T_world_body`` (body coordinates to the local world frame).
        velocity: Linear velocity expressed in the local world frame [m/s].
        accel_bias: Accelerometer bias [m/s^2].
        gyro_bias: Gyroscope bias [rad/s].
    """

    timestamp: float
    pose: gtsam.Pose3 = field(default_factory=gtsam.Pose3)
    velocity: Vector3 = field(default_factory=lambda: np.zeros(3))
    accel_bias: Vector3 = field(default_factory=lambda: np.zeros(3))
    gyro_bias: Vector3 = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not isinstance(self.pose, gtsam.Pose3):
            raise TypeError("pose must be a gtsam.Pose3")

        self.velocity = _vector3(self.velocity, "velocity")
        self.accel_bias = _vector3(self.accel_bias, "accel_bias")
        self.gyro_bias = _vector3(self.gyro_bias, "gyro_bias")

    @property
    def bias(self) -> gtsam.imuBias.ConstantBias:
        return gtsam.imuBias.ConstantBias(self.accel_bias, self.gyro_bias)

    @classmethod
    def identity(cls, timestamp: float = 0.0) -> "NavigationState":
        """Create a zero-motion state, useful as an initial estimate skeleton."""
        return cls(timestamp=timestamp)

    def insert_into(self, values: gtsam.Values, index: int) -> None:
        """Insert this state into ``values`` using the keys for ``index``."""
        keys = StateKeys.at(index)
        values.insert(keys.pose, self.pose)
        values.insert(keys.velocity, self.velocity)
        values.insert(keys.bias, self.bias)

    @classmethod
    def from_values(
        cls, values: gtsam.Values, index: int, timestamp: float
    ) -> "NavigationState":
        """Reconstruct a state from optimized GTSAM values."""
        keys = StateKeys.at(index)
        bias = values.atConstantBias(keys.bias)

        # print states 
        optimized_pose = values.atPose3(keys.pose)
        optimized_vel = values.atVector(keys.velocity)
        optimized_bias = bias
        
        print("optimized_pose: ", optimized_pose)
        print("optimized vel: ", optimized_vel)
        print("optimized acc bias: ", optimized_bias.accelerometer())
        print("optimized gyro bias: ", optimized_bias.gyroscope())

        return cls(
            timestamp=timestamp,
            pose=values.atPose3(keys.pose),
            velocity=values.atVector(keys.velocity),
            accel_bias=bias.accelerometer(),
            gyro_bias=bias.gyroscope(),
        )
    
def create_initial_state(
    timestamp: float,
    position: np.ndarray | None = None,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    velocity: np.ndarray | None = None,
    accel_bias: np.ndarray | None = None,
    gyro_bias: np.ndarray | None = None,
) -> NavigationState:
    if position is None:
        position = np.zeros(3)
    if velocity is None:
        velocity = np.zeros(3)
    if accel_bias is None:
        accel_bias = np.zeros(3)
    if gyro_bias is None:
        gyro_bias = np.zeros(3)

    rotation = gtsam.Rot3.Ypr(yaw, pitch, roll)  # radians
    pose0 = gtsam.Pose3(rotation, position)

    return NavigationState(
        timestamp=timestamp,
        pose=pose0,
        velocity=velocity,
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )
