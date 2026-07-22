"""Small, dependency-free RINEX 3 parser for tightly-coupled GNSS.

The campus data uses RINEX 3.02 observation files and RINEX 3.04 mixed
navigation files.  This module deliberately parses measurements and broadcast
ephemeris records only; satellite orbit propagation and atmospheric
corrections belong in a separate preprocessing module.

Units
-----
* code/pseudorange (``C...``): metres
* carrier phase (``L...``): cycles
* Doppler (``D...``): hertz
* signal strength (``S...``): dB-Hz
* ``range_rate``: metres/second, using ``-wavelength * Doppler``
* epoch time: GPS week and GPS seconds-of-week
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, TextIO


SPEED_OF_LIGHT = 299_792_458.0
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

# Frequencies for the signals used by the campus receiver.  GLONASS is not in
# this table because its FDMA frequency depends on the satellite channel.
_BAND_FREQUENCIES_HZ: Mapping[str, Mapping[str, float]] = {
    "G": {"1": 1575.42e6, "2": 1227.60e6, "5": 1176.45e6},
    "E": {"1": 1575.42e6, "5": 1176.45e6, "6": 1278.75e6, "7": 1207.14e6, "8": 1191.795e6},
    "C": {"1": 1575.42e6, "2": 1561.098e6, "5": 1176.45e6, "6": 1268.52e6, "7": 1207.14e6, "8": 1191.795e6},
    "J": {"1": 1575.42e6, "2": 1227.60e6, "5": 1176.45e6, "6": 1278.75e6},
    "S": {"1": 1575.42e6, "5": 1176.45e6},
    "I": {"5": 1176.45e6, "9": 2492.028e6},
}


@dataclass(frozen=True)
class RinexValue:
    """One 16-character RINEX observation field."""

    value: float
    lli: int | None = None
    signal_strength_flag: int | None = None


@dataclass(frozen=True)
class SatelliteObservation:
    """All observation fields for one satellite at one epoch."""

    satellite: str
    observations: Mapping[str, RinexValue]

    def value(self, observation_type: str) -> float | None:
        item = self.observations.get(observation_type)
        return None if item is None else item.value

    def pseudorange(self, signal: str = "1C") -> float | None:
        return self.value("C" + signal)

    def doppler(self, signal: str = "1C") -> float | None:
        return self.value("D" + signal)

    def cn0(self, signal: str = "1C") -> float | None:
        return self.value("S" + signal)

    def range_rate(self, signal: str = "1C") -> float | None:
        """Return observed pseudorange rate [m/s]."""
        doppler_hz = self.doppler(signal)
        if doppler_hz is None:
            return None
        frequency_hz = carrier_frequency_hz(self.satellite, signal)
        return -SPEED_OF_LIGHT / frequency_hz * doppler_hz


@dataclass(frozen=True)
class ObservationEpoch:
    """One RINEX observation epoch."""

    time: datetime
    gps_week: int
    gps_seconds_of_week: float
    flag: int
    receiver_clock_offset: float | None
    satellites: tuple[SatelliteObservation, ...]

    @property
    def gps_seconds(self) -> float:
        return self.gps_week * 604_800.0 + self.gps_seconds_of_week


@dataclass(frozen=True)
class BroadcastEphemeris:
    """One raw broadcast-navigation record.

    ``clock`` contains af0/af1/af2 for Keplerian constellations. ``values``
    contains the remaining fields in RINEX order.  Keeping these values raw
    avoids silently applying a GPS field interpretation to GLONASS/SBAS.
    """

    satellite: str
    toc: datetime
    clock: tuple[float, float, float]
    values: tuple[float, ...]


@dataclass(frozen=True)
class TcDataset:
    """Input paths and optional duration loaded from ``config/tc.yaml``."""

    observation_path: Path
    navigation_path: Path
    imu_path: Path
    duration: float | None = None


def _float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _optional_digit(text: str) -> int | None:
    return int(text) if text.strip().isdigit() else None


def gps_week_and_sow(time: datetime) -> tuple[int, float]:
    """Convert a calendar value expressed in GPST to GPS week/SOW.

    No leap-second correction is made: RINEX epochs labelled GPS are already
    expressed in GPS system time, even though ``datetime`` uses a UTC tzinfo as
    a convenient non-naive container.
    """
    elapsed = (time - GPS_EPOCH).total_seconds()
    week = math.floor(elapsed / 604_800.0)
    return week, elapsed - week * 604_800.0


def carrier_frequency_hz(satellite: str, signal: str) -> float:
    """Return carrier frequency for e.g. ``satellite='G05', signal='1C'``."""
    if len(satellite) < 2 or len(signal) < 1:
        raise ValueError("invalid satellite or signal identifier")
    system, band = satellite[0], signal[0]
    try:
        return _BAND_FREQUENCIES_HZ[system][band]
    except KeyError as exc:
        if system == "R":
            raise ValueError(
                "GLONASS carrier frequency requires the slot frequency channel"
            ) from exc
        raise ValueError(f"unsupported signal {satellite}/{signal}") from exc


def _read_header(stream: TextIO) -> tuple[float, dict[str, tuple[str, ...]]]:
    version: float | None = None
    observation_types: dict[str, list[str]] = {}
    expected_counts: dict[str, int] = {}

    for line in stream:
        label = line[60:80].strip() if len(line) >= 60 else ""
        if label == "RINEX VERSION / TYPE":
            version = float(line[:9])
        elif label == "SYS / # / OBS TYPES":
            system = line[0]
            if system.strip():
                expected_counts[system] = int(line[3:6])
                observation_types[system] = []
            elif not observation_types:
                raise ValueError("observation-type continuation without a system")
            else:
                system = next(reversed(observation_types))
            observation_types[system].extend(line[7:60].split())
        elif label == "END OF HEADER":
            break
    else:
        raise ValueError("RINEX END OF HEADER not found")

    if version is None or not 3.0 <= version < 4.0:
        raise ValueError(f"only RINEX 3.x is supported, found {version}")
    for system, count in expected_counts.items():
        if len(observation_types[system]) != count:
            raise ValueError(
                f"{system} declares {count} observation types but "
                f"{len(observation_types[system])} were parsed"
            )
    return version, {key: tuple(value) for key, value in observation_types.items()}


def _epoch_time(line: str) -> datetime:
    fields = line[1:43].split()
    if len(fields) < 6:
        raise ValueError(f"invalid epoch line: {line.rstrip()}")
    year, month, day, hour, minute = map(int, fields[:5])
    second = float(fields[5])
    whole_second = int(second)
    microsecond = round((second - whole_second) * 1_000_000)
    if microsecond == 1_000_000:
        whole_second += 1
        microsecond = 0
    return datetime(
        year, month, day, hour, minute, whole_second, microsecond,
        tzinfo=timezone.utc,
    )


def _parse_observation_field(field: str) -> RinexValue | None:
    field = field.ljust(16)
    if not field[:14].strip():
        return None
    return RinexValue(
        value=_float(field[:14]),
        lli=_optional_digit(field[14:15]),
        signal_strength_flag=_optional_digit(field[15:16]),
    )


def read_rinex_observations(path: str | Path) -> Iterator[ObservationEpoch]:
    """Yield epochs from a RINEX 3 observation file without loading it all."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"RINEX observation file not found: {file_path}")

    with file_path.open("r", encoding="ascii", errors="strict") as stream:
        _, observation_types = _read_header(stream)
        for line in stream:
            if not line.startswith(">"):
                continue
            time = _epoch_time(line)
            flag = int(line[31:32])
            satellite_count = int(line[32:35])
            clock_text = line[41:56].strip()
            clock_offset = _float(clock_text) if clock_text else None

            # Event epochs contain special records rather than observations.
            if flag not in (0, 1):
                for _ in range(satellite_count):
                    next(stream)
                continue

            satellites: list[SatelliteObservation] = []
            for _ in range(satellite_count):
                first_line = next(stream)
                satellite = first_line[:3].strip()
                if not satellite or satellite[0] not in observation_types:
                    raise ValueError(f"unknown satellite record: {first_line.rstrip()}")
                types = observation_types[satellite[0]]
                # RINEX 3 satellite observation records are allowed to exceed
                # 80 columns (campus rows reach 323 columns).  A short row
                # means its trailing observations are blank, not that the next
                # satellite row is a continuation.
                payload = first_line[3:].rstrip("\r\n").ljust(16 * len(types))

                values: dict[str, RinexValue] = {}
                for index, observation_type in enumerate(types):
                    item = _parse_observation_field(payload[index * 16:(index + 1) * 16])
                    if item is not None:
                        values[observation_type] = item
                satellites.append(SatelliteObservation(satellite, values))

            week, sow = gps_week_and_sow(time)
            yield ObservationEpoch(
                time=time,
                gps_week=week,
                gps_seconds_of_week=sow,
                flag=flag,
                receiver_clock_offset=clock_offset,
                satellites=tuple(satellites),
            )


def _navigation_record_line_count(system: str) -> int:
    return 4 if system in {"R", "S"} else 8


def read_rinex_navigation(path: str | Path) -> Iterator[BroadcastEphemeris]:
    """Yield raw ephemeris records from a mixed RINEX 3 navigation file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"RINEX navigation file not found: {file_path}")

    with file_path.open("r", encoding="ascii", errors="strict") as stream:
        version, _ = _read_header(stream)
        del version
        for first_line in stream:
            if not first_line.strip():
                continue
            satellite = first_line[:3].strip()
            fields = first_line[3:23].split()
            if len(fields) != 6:
                raise ValueError(f"invalid navigation epoch: {first_line.rstrip()}")
            year, month, day, hour, minute, second = map(int, fields)
            toc = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
            clock = tuple(
                _float(first_line[start:start + 19])
                for start in (23, 42, 61)
            )

            values: list[float] = []
            for _ in range(_navigation_record_line_count(satellite[0]) - 1):
                continuation = next(stream)
                for start in (4, 23, 42, 61):
                    text = continuation[start:start + 19].strip()
                    if text:
                        values.append(_float(text))
            yield BroadcastEphemeris(satellite, toc, clock, tuple(values))


def load_tc_config(path: str | Path) -> TcDataset:
    """Load the campus tightly-coupled dataset paths from a YAML file.

    ``data.path`` is interpreted relative to the current working directory,
    matching ``tc.yaml``'s ``./campus`` value.  If it does not exist there, it
    is also tried relative to the YAML file, which makes copied configs easier
    to use from another working directory.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"TC config file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config parsing requires PyYAML") from exc

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("TC config must contain a 'data' mapping")
    data = config["data"]
    missing = [key for key in ("path", "imu", "nav", "obs") if not data.get(key)]
    if missing:
        raise ValueError(f"TC config data is missing: {', '.join(missing)}")

    base = Path(data["path"]).expanduser()
    if not base.is_absolute() and not base.exists():
        base = config_path.resolve().parent / base
    base = base.resolve()

    duration_value = data.get("duration")
    duration = None if duration_value is None else float(duration_value)
    if duration is not None and (not math.isfinite(duration) or duration <= 0.0):
        raise ValueError("data.duration must be null or a positive number")

    dataset = TcDataset(
        observation_path=base / str(data["obs"]),
        navigation_path=base / str(data["nav"]),
        imu_path=base / str(data["imu"]),
        duration=duration,
    )
    for label, input_path in (
        ("observation", dataset.observation_path),
        ("navigation", dataset.navigation_path),
        ("IMU", dataset.imu_path),
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"TC {label} file not found: {input_path}")
    return dataset


def read_tc_observations(config_path: str | Path) -> Iterator[ObservationEpoch]:
    """Read observations selected by ``tc.yaml``, respecting ``duration``."""
    dataset = load_tc_config(config_path)
    start_gps_seconds: float | None = None
    for epoch in read_rinex_observations(dataset.observation_path):
        if start_gps_seconds is None:
            start_gps_seconds = epoch.gps_seconds
        if (
            dataset.duration is not None
            and epoch.gps_seconds - start_gps_seconds > dataset.duration
        ):
            break
        yield epoch


def read_tc_navigation(config_path: str | Path) -> Iterator[BroadcastEphemeris]:
    """Read the navigation file selected by ``tc.yaml``."""
    dataset = load_tc_config(config_path)
    yield from read_rinex_navigation(dataset.navigation_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect campus RINEX observations")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--obs", type=Path, help="RINEX observation file")
    inputs.add_argument("--config", type=Path, help="tc.yaml dataset config")
    parser.add_argument("--limit", type=int, default=1, help="epochs to print")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    epochs = (
        read_tc_observations(args.config)
        if args.config is not None
        else read_rinex_observations(args.obs)
    )
    if args.config is not None:
        dataset = load_tc_config(args.config)
        print(f"OBS: {dataset.observation_path}")
        print(f"NAV: {dataset.navigation_path}")
        print(f"IMU: {dataset.imu_path}")

    for epoch_index, epoch in enumerate(epochs):
        print(
            f"{epoch.time.isoformat()} GPS week={epoch.gps_week} "
            f"sow={epoch.gps_seconds_of_week:.3f} satellites={len(epoch.satellites)}"
        )
        for observation in epoch.satellites:
            rho = observation.pseudorange("1C")
            doppler = observation.doppler("1C")
            if rho is not None and doppler is not None and observation.satellite[0] != "R":
                print(
                    f"  {observation.satellite} C1C={rho:.3f} m "
                    f"D1C={doppler:.3f} Hz "
                    f"rate={observation.range_rate('1C'):.3f} m/s "
                    f"S1C={observation.cn0('1C')}"
                )
        if epoch_index + 1 >= args.limit:
            break


if __name__ == "__main__":
    main()

# python3 agr_fgo/src/sensors/gnss_raw.py \
#   --config agr_fgo/config/tc.yaml \
#   --limit 1