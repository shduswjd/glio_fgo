"""GNSS measurement type and ROS1 bag reader.

The bag-specific ROS message is converted to :class:`GnssMeasurement` so the
factor-graph code does not need a ROS installation or ROS message objects.

Internal conventions
--------------------
* timestamp: seconds [s]
* latitude/longitude: degrees [deg]
* altitude: metres above the WGS84 ellipsoid [m]
* position covariance: East-North-Up (ENU), square metres [m^2]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


DEFAULT_GNSS_TOPIC = "/piksi/navsatfix_best_fix"


def _as_covariance(value: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a validated 3x3 position covariance matrix."""
    covariance = np.asarray(value, dtype=float)
    if covariance.size != 9:
        raise ValueError(
            f"position_covariance must contain 9 values, got {covariance.size}"
        )
    covariance = covariance.reshape(3, 3)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("position_covariance must contain only finite values")
    return covariance.copy()


@dataclass(frozen=True)
class GnssMeasurement:
    """One ``sensor_msgs/NavSatFix`` sample independent of ROS.

    ``status`` follows ``sensor_msgs/NavSatStatus``: ``-1`` means no fix,
    while values greater than or equal to zero mean a valid fix.
    ``covariance_type`` follows the constants in ``sensor_msgs/NavSatFix``.
    """

    timestamp: float
    latitude: float
    longitude: float
    altitude: float
    position_covariance: np.ndarray
    covariance_type: int
    status: int
    service: int
    frame_id: str = ""

    def __post_init__(self) -> None:
        numeric_values = (self.timestamp, self.latitude, self.longitude, self.altitude)
        if not np.all(np.isfinite(numeric_values)):
            raise ValueError("timestamp and GNSS coordinates must be finite")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90 degrees")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be between -180 and 180 degrees")
        object.__setattr__(
            self,
            "position_covariance",
            _as_covariance(self.position_covariance),
        )


def _timestamp_from_rosbags(message: object, recorded_ns: int) -> float:
    """Prefer the sensor header stamp and fall back to bag record time."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        timestamp = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if timestamp > 0.0:
            return timestamp
    return recorded_ns * 1e-9


def _measurement_from_message(message: object, timestamp: float) -> GnssMeasurement:
    """Convert a ROS1 or rosbags NavSatFix object to our plain data type."""
    status = message.status
    header = getattr(message, "header", None)
    return GnssMeasurement(
        timestamp=timestamp,
        latitude=message.latitude,
        longitude=message.longitude,
        altitude=message.altitude,
        position_covariance=message.position_covariance,
        covariance_type=message.position_covariance_type,
        status=status.status,
        service=status.service,
        frame_id=getattr(header, "frame_id", ""),
    )


def read_gnss_ros1_bag(
    bag_path: str | Path,
    topic: str = DEFAULT_GNSS_TOPIC,
    *,
    require_fix: bool = True,
) -> Iterator[GnssMeasurement]:
    """Read ``sensor_msgs/NavSatFix`` messages from a ROS1 ``.bag`` file.

    The native ROS1 ``rosbag`` module is used when available.  Otherwise this
    function falls back to the pure-Python ``rosbags`` package.  By default,
    no-fix messages, non-finite positions, and non-increasing timestamps are
    skipped because they cannot be used as GNSS position factors.
    """
    path = Path(bag_path)
    if not path.is_file():
        raise FileNotFoundError(f"ROS bag not found: {path}")

    try:
        import rosbag  # type: ignore[import-not-found]
    except ImportError:
        yield from _read_gnss_with_rosbags(path, topic, require_fix=require_fix)
        return

    previous_timestamp: float | None = None
    found_topic = False
    with rosbag.Bag(str(path), "r") as bag:
        available = sorted(bag.get_type_and_topic_info().topics)
        for _, message, recorded_time in bag.read_messages(topics=[topic]):
            found_topic = True
            stamp = getattr(getattr(message, "header", None), "stamp", None)
            sensor_time = stamp.to_sec() if stamp is not None else 0.0
            timestamp = sensor_time if sensor_time > 0.0 else recorded_time.to_sec()

            if previous_timestamp is not None and timestamp <= previous_timestamp:
                continue
            previous_timestamp = timestamp
            if require_fix and message.status.status < 0:
                continue
            try:
                yield _measurement_from_message(message, timestamp)
            except ValueError:
                continue

    if not found_topic:
        raise ValueError(
            f"topic {topic!r} is not in {path.name}. "
            f"Available topics: {', '.join(available)}"
        )


def _read_gnss_with_rosbags(
    bag_path: Path,
    topic: str,
    *,
    require_fix: bool,
) -> Iterator[GnssMeasurement]:
    """Read ROS1 NavSatFix data without requiring a ROS installation."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError(
            "Reading this bag requires either ROS1 rosbag or the 'rosbags' "
            "Python package (install it with: pip install rosbags)"
        ) from exc

    with AnyReader([bag_path]) as reader:
        connections = [item for item in reader.connections if item.topic == topic]
        if not connections:
            available = ", ".join(sorted({item.topic for item in reader.connections}))
            raise ValueError(
                f"topic {topic!r} is not in {bag_path.name}. "
                f"Available topics: {available}"
            )
        invalid_types = sorted(
            {item.msgtype for item in connections if not item.msgtype.endswith("/NavSatFix")}
        )
        if invalid_types:
            raise TypeError(
                f"topic {topic!r} is not sensor_msgs/NavSatFix; "
                f"found: {', '.join(invalid_types)}"
            )

        previous_timestamp: float | None = None
        for connection, recorded_ns, raw_data in reader.messages(
            connections=connections
        ):
            message = reader.deserialize(raw_data, connection.msgtype)
            timestamp = _timestamp_from_rosbags(message, recorded_ns)
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                continue
            previous_timestamp = timestamp
            if require_fix and message.status.status < 0:
                continue
            try:
                yield _measurement_from_message(message, timestamp)
            except ValueError:
                continue


def main() -> None:
    """Small command-line check for users unfamiliar with ROS tooling."""
    parser = argparse.ArgumentParser(description="Read GNSS fixes from a ROS1 bag")
    parser.add_argument("bag", type=Path, help="path to a ROS1 .bag file")
    parser.add_argument("--topic", default=DEFAULT_GNSS_TOPIC)
    parser.add_argument("--limit", type=int, default=5, help="rows to print")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    count = 0
    for measurement in read_gnss_ros1_bag(args.bag, args.topic):
        print(
            f"{measurement.timestamp:.6f}  "
            f"lat={measurement.latitude:.9f}  "
            f"lon={measurement.longitude:.9f}  "
            f"alt={measurement.altitude:.3f} m  "
            f"status={measurement.status}"
        )
        count += 1
        if count >= args.limit:
            break
    if count == 0:
        raise SystemExit("No valid GNSS fixes found")


if __name__ == "__main__":
    main()

# .venv/bin/python agr_fgo/src/sensors/gnss.py \
#   01_13B_Jackal/base_2023-07-18-14-26-48_0.bag \
#   --limit 5

# 결과: timestamp, lat, lon, alt, status
