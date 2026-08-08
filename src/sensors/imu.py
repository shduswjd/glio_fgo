"""IMU measurement type and file readers.

Both TXT files and ROS bags are converted to :class:`ImuMeasurement`.  The rest
of the estimator therefore does not need to know where a measurement came from.

Internal conventions
--------------------
* timestamp: seconds [s]
* acceleration: metres per second squared [m/s^2]
* angular velocity: radians per second [rad/s]
* orientation: quaternion [x, y, z, w], or None when unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


Vector3 = np.ndarray


def _as_vector3(value: Sequence[float], name: str) -> Vector3:
    """Convert an input to a finite NumPy vector with shape ``(3,)``."""
    vector = np.asarray(value, dtype=float)

    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")

    return vector.copy()


@dataclass(frozen=True)
class ImuMeasurement:
    """One calibrated IMU sample expressed in the IMU sensor frame."""

    timestamp: float
    acceleration: Vector3
    angular_velocity: Vector3
    orientation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")

        object.__setattr__(
            self, "acceleration", _as_vector3(self.acceleration, "acceleration")
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _as_vector3(self.angular_velocity, "angular_velocity"),
        )
        if self.orientation is not None:
            orientation = np.asarray(self.orientation, dtype=float)
            if orientation.shape != (4,):
                raise ValueError(
                    f"orientation must have shape (4,), got {orientation.shape}"
                )
            norm = np.linalg.norm(orientation)
            if not np.all(np.isfinite(orientation)) or norm == 0.0:
                raise ValueError("orientation must be a finite non-zero quaternion")
            object.__setattr__(self, "orientation", orientation / norm)


def read_imu_txt(
    file_path: str | Path,
    *,
    timestamp_column: int = 0,
    acceleration_columns: tuple[int, int, int] = (1, 2, 3),
    angular_velocity_columns: tuple[int, int, int] = (4, 5, 6),
    timestamp_scale: float = 1.0,
    angular_velocity_scale: float = 1.0,
    delimiter: str | None = None,
    skip_rows: int = 0,
) -> Iterator[ImuMeasurement]:
    """Read IMU samples from a numeric TXT or CSV file.

    The default column layout is::

        time  ax  ay  az  gx  gy  gz

    ``timestamp_scale`` converts the timestamp to seconds.  For example, use
    ``1e-9`` when the file stores nanoseconds.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"IMU file not found: {path}")
    if not np.isfinite(timestamp_scale) or timestamp_scale <= 0.0:
        raise ValueError("timestamp_scale must be a positive finite number")
    if not np.isfinite(angular_velocity_scale) or angular_velocity_scale <= 0.0:
        raise ValueError("angular_velocity_scale must be a positive finite number")

    data = np.loadtxt(path, delimiter=delimiter, skiprows=skip_rows, ndmin=2)
    required_columns = (
        timestamp_column,
        *acceleration_columns,
        *angular_velocity_columns,
    )

    if min(required_columns) < 0:
        raise ValueError("column indices must be non-negative")
    if max(required_columns) >= data.shape[1]:
        raise ValueError(
            f"requested column {max(required_columns)}, but {path} has "
            f"only {data.shape[1]} columns"
        )

    previous_timestamp: float | None = None
    for row_number, row in enumerate(data, start=skip_rows + 1):
        timestamp = float(row[timestamp_column]) * timestamp_scale

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                f"timestamps must increase: invalid timestamp at row {row_number}"
            )

        yield ImuMeasurement(
            timestamp=timestamp,
            acceleration=row[list(acceleration_columns)],
            angular_velocity=(
                row[list(angular_velocity_columns)] * angular_velocity_scale
            ),
        )
        previous_timestamp = timestamp


def read_imu_ros1_bag(
    bag_path: str | Path,
    topic: str = "/imu/data",
) -> Iterator[ImuMeasurement]:
    """Read ``sensor_msgs/Imu`` messages from a ROS1 ``.bag`` file.

    ROS is imported inside this function so TXT processing and estimator tests
    can run on computers where ROS is not installed.
    """
    path = Path(bag_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROS bag not found: {path}")

    try:
        import rosbag  # type: ignore[import-not-found]
    except ImportError:
        # 일반 Python 가상환경에는 ROS1의 rosbag 모듈이 없을 수 있다.
        # 그 경우 pure-Python rosbags 패키지로 같은 ImuMeasurement를 만든다.
        yield from _read_imu_with_rosbags(path, topic)
        return

    previous_timestamp: float | None = None
    with rosbag.Bag(str(path), "r") as bag:
        for _, message, recorded_time in bag.read_messages(topics=[topic]):
            # Sensor time is preferred. Bag time can include transport latency.
            stamp = getattr(getattr(message, "header", None), "stamp", None)
            timestamp = (
                stamp.to_sec()
                if stamp is not None and stamp.to_sec() > 0.0
                else recorded_time.to_sec()
            )

            # 일부 실제 bag에는 중복되거나 순서가 뒤집힌 header stamp가 있다.
            # dt <= 0인 측정은 preintegration할 수 없으므로 건너뛴다.
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                continue

            yield ImuMeasurement(
                timestamp=timestamp,
                acceleration=(
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.linear_acceleration.z,
                ),
                angular_velocity=(
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                ),
                orientation=(
                    message.orientation.x,
                    message.orientation.y,
                    message.orientation.z,
                    message.orientation.w,
                ) if message.orientation_covariance[0] >= 0.0 else None,
            )
            previous_timestamp = timestamp


def _read_imu_with_rosbags(
    bag_path: Path,
    topic: str,
) -> Iterator[ImuMeasurement]:
    """ROS 설치 없이 rosbags 패키지로 ROS1 IMU 메시지를 읽는다."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError(
            "Reading this bag requires either ROS1 rosbag or the 'rosbags' "
            "Python package"
        ) from exc

    with AnyReader([bag_path]) as reader:
        connections = [item for item in reader.connections if item.topic == topic]
        if not connections:
            available = ", ".join(sorted({item.topic for item in reader.connections}))
            raise ValueError(
                f"topic {topic!r} is not in {bag_path.name}. "
                f"Available topics: {available}"
            )

        previous_timestamp: float | None = None
        for connection, recorded_ns, raw_data in reader.messages(
            connections=connections
        ):
            message = reader.deserialize(raw_data, connection.msgtype)
            stamp = message.header.stamp
            sensor_timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            timestamp = (
                sensor_timestamp if sensor_timestamp > 0.0 else recorded_ns * 1e-9
            )

            # 일부 실제 bag에는 중복되거나 순서가 뒤집힌 header stamp가 있다.
            # dt <= 0인 측정은 preintegration할 수 없으므로 건너뛴다.
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                continue

            yield ImuMeasurement(
                timestamp=timestamp,
                acceleration=(
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.linear_acceleration.z,
                ),
                angular_velocity=(
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                ),
                orientation=(
                    message.orientation.x,
                    message.orientation.y,
                    message.orientation.z,
                    message.orientation.w,
                ) if message.orientation_covariance[0] >= 0.0 else None,
            )
            previous_timestamp = timestamp
