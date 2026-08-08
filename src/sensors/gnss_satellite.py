"""GPS broadcast-orbit propagation and GNSS measurement corrections."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Iterable

import numpy as np

from sensors.gnss_models import (
    BeidouEphemeris, GalileoEphemeris, GpsEphemeris, SatelliteState,
)
from sensors.gnss_raw import BroadcastEphemeris, gps_week_and_sow


SPEED_OF_LIGHT = 299_792_458.0
EARTH_ROTATION_RATE = 7.2921151467e-5
GPS_GRAVITATIONAL_CONSTANT = 3.986005e14
GALILEO_GRAVITATIONAL_CONSTANT = 3.986004418e14
BEIDOU_GRAVITATIONAL_CONSTANT = 3.986004418e14
BEIDOU_GPS_WEEK_OFFSET = 1356
BEIDOU_TO_GPS_SECONDS = 14.0
BEIDOU_GEO_TILT_RAD = math.radians(-5.0)
BEIDOU_B2I_FREQUENCY_HZ = 1207.14e6
BEIDOU_B3I_FREQUENCY_HZ = 1268.52e6
RELATIVISTIC_CLOCK_CONSTANT = -4.442807633e-10
GPS_WEEK_SECONDS = 604_800.0


def _wrap_week(seconds: float) -> float:
    """Return a GPS time difference in the nearest half-week."""
    return (seconds + GPS_WEEK_SECONDS / 2.0) % GPS_WEEK_SECONDS - GPS_WEEK_SECONDS / 2.0


def gps_ephemeris_from_rinex(record: BroadcastEphemeris) -> GpsEphemeris:
    """Convert a raw RINEX navigation record to named GPS fields."""
    if not record.satellite.startswith("G"):
        raise ValueError("only GPS Keplerian ephemeris is supported")
    if len(record.values) < 24:
        raise ValueError(f"incomplete ephemeris for {record.satellite}")
    v = record.values
    toc_week, toc_sow = gps_week_and_sow(record.toc)
    week = int(v[18])
    # RINEX may contain a truncated/rolled week.  Pick the cycle nearest toc.
    while week - toc_week > 512:
        week -= 1024
    while toc_week - week > 512:
        week += 1024
    return GpsEphemeris(
        satellite=record.satellite,
        toc=toc_week * GPS_WEEK_SECONDS + toc_sow,
        af0=record.clock[0], af1=record.clock[1], af2=record.clock[2],
        iode=v[0], crs=v[1], delta_n=v[2], m0=v[3], cuc=v[4],
        eccentricity=v[5], cus=v[6], sqrt_a=v[7], toe=v[8],
        cic=v[9], omega0=v[10], cis=v[11], i0=v[12], crc=v[13],
        argument_of_perigee=v[14], omega_dot=v[15], idot=v[16],
        gps_week=week, tgd=v[22],
    )


def galileo_ephemeris_from_rinex(record: BroadcastEphemeris) -> GalileoEphemeris:
    """Convert a raw RINEX Galileo navigation record to named fields."""
    if not record.satellite.startswith("E"):
        raise ValueError("Galileo ephemeris requires an E satellite")
    if len(record.values) < 24:
        raise ValueError(f"incomplete ephemeris for {record.satellite}")
    v = record.values
    toc_week, toc_sow = gps_week_and_sow(record.toc)
    week = int(v[18])
    while week - toc_week > 2048:
        week -= 4096
    while toc_week - week > 2048:
        week += 4096
    return GalileoEphemeris(
        satellite=record.satellite, toc=toc_week * GPS_WEEK_SECONDS + toc_sow,
        af0=record.clock[0], af1=record.clock[1], af2=record.clock[2],
        iodnav=v[0], crs=v[1], delta_n=v[2], m0=v[3], cuc=v[4],
        eccentricity=v[5], cus=v[6], sqrt_a=v[7], toe=v[8],
        cic=v[9], omega0=v[10], cis=v[11], i0=v[12], crc=v[13],
        argument_of_perigee=v[14], omega_dot=v[15], idot=v[16],
        # Galileo RINEX order after week is SISA, health, the two BGDs,
        # then transmission time. Unlike GPS, the BGD fields are v[21:23].
        gps_week=week, bgd_e5a_e1=v[21], bgd_e5b_e1=v[22],
    )


def beidou_ephemeris_from_rinex(record: BroadcastEphemeris) -> BeidouEphemeris:
    """Convert BDT-based RINEX fields onto the internal GPST axis."""
    if not record.satellite.startswith("C"):
        raise ValueError("BeiDou ephemeris requires a C satellite")
    if len(record.values) < 24:
        raise ValueError(f"incomplete ephemeris for {record.satellite}")
    v = record.values
    gps_week = int(v[18]) + BEIDOU_GPS_WEEK_OFFSET
    # Keep Toe as BDT seconds-of-week because it also appears in the Earth
    # rotation term. Add the 14 s offset only when forming an absolute GPST.
    toe = v[8]
    toc_week, toc_sow = gps_week_and_sow(record.toc)
    toc = toc_week * GPS_WEEK_SECONDS + toc_sow + BEIDOU_TO_GPS_SECONDS
    return BeidouEphemeris(
        satellite=record.satellite, toc=toc,
        af0=record.clock[0], af1=record.clock[1], af2=record.clock[2],
        aode=v[0], crs=v[1], delta_n=v[2], m0=v[3], cuc=v[4],
        eccentricity=v[5], cus=v[6], sqrt_a=v[7], toe=toe,
        cic=v[9], omega0=v[10], cis=v[11], i0=v[12], crc=v[13],
        argument_of_perigee=v[14], omega_dot=v[15], idot=v[16],
        gps_week=gps_week, tgd_b1_b3=v[22], tgd_b2_b3=v[23],
    )


class EphemerisStore:
    """Dictionary-backed time index; values remain typed ephemeris objects."""

    def __init__(self, ephemerides: Iterable[GpsEphemeris | GalileoEphemeris | BeidouEphemeris]):
        grouped: dict[str, list[GpsEphemeris | GalileoEphemeris | BeidouEphemeris]] = defaultdict(list)
        for ephemeris in ephemerides:
            grouped[ephemeris.satellite].append(ephemeris)
        self._by_satellite = {
            satellite: tuple(sorted(items, key=lambda item: item.toe))
            for satellite, items in grouped.items()
        }

    @classmethod
    def from_rinex(cls, records: Iterable[BroadcastEphemeris]) -> "EphemerisStore":
        converted = []
        for record in records:
            if record.satellite.startswith("G"):
                converted.append(gps_ephemeris_from_rinex(record))
            elif record.satellite.startswith("E"):
                converted.append(galileo_ephemeris_from_rinex(record))
            elif record.satellite.startswith("C"):
                converted.append(beidou_ephemeris_from_rinex(record))
        return cls(converted)

    def nearest(self, satellite: str, gps_time: float, max_age_s: float = 14_400.0) -> GpsEphemeris | GalileoEphemeris | BeidouEphemeris:
        candidates = self._by_satellite.get(satellite, ())
        if not candidates:
            raise KeyError(f"no ephemeris for {satellite}")
        selected = min(
            candidates,
            key=lambda item: abs(_wrap_week(gps_time - _toe_time_gps(item))),
        )
        age = abs(_wrap_week(gps_time - _toe_time_gps(selected)))
        if age > max_age_s:
            raise ValueError(f"ephemeris for {satellite} is {age:.0f} s old")
        return selected


def _toe_time_gps(ephemeris: GpsEphemeris | GalileoEphemeris | BeidouEphemeris) -> float:
    offset = BEIDOU_TO_GPS_SECONDS if isinstance(ephemeris, BeidouEphemeris) else 0.0
    return ephemeris.gps_week * GPS_WEEK_SECONDS + ephemeris.toe + offset


def _position_and_clock(
    ephemeris: GpsEphemeris | GalileoEphemeris | BeidouEphemeris,
    gps_time: float,
    gravitational_constant: float,
    apply_group_delay: bool = True,
) -> tuple[np.ndarray, float]:
    toe_time = _toe_time_gps(ephemeris)
    tk = _wrap_week(gps_time - toe_time)
    semi_major_axis = ephemeris.sqrt_a**2
    mean_motion = math.sqrt(gravitational_constant / semi_major_axis**3) + ephemeris.delta_n
    mean_anomaly = ephemeris.m0 + mean_motion * tk

    eccentric_anomaly = mean_anomaly
    for _ in range(12):
        updated = mean_anomaly + ephemeris.eccentricity * math.sin(eccentric_anomaly)
        if abs(updated - eccentric_anomaly) < 1e-13:
            break
        eccentric_anomaly = updated

    e = ephemeris.eccentricity
    true_anomaly = math.atan2(math.sqrt(1.0 - e * e) * math.sin(eccentric_anomaly), math.cos(eccentric_anomaly) - e)
    phi = true_anomaly + ephemeris.argument_of_perigee
    twice_phi = 2.0 * phi
    argument = phi + ephemeris.cus * math.sin(twice_phi) + ephemeris.cuc * math.cos(twice_phi)
    radius = semi_major_axis * (1.0 - e * math.cos(eccentric_anomaly)) + ephemeris.crs * math.sin(twice_phi) + ephemeris.crc * math.cos(twice_phi)
    inclination = ephemeris.i0 + ephemeris.idot * tk + ephemeris.cis * math.sin(twice_phi) + ephemeris.cic * math.cos(twice_phi)
    orbital_x, orbital_y = radius * math.cos(argument), radius * math.sin(argument)
    longitude = ephemeris.omega0 + (ephemeris.omega_dot - EARTH_ROTATION_RATE) * tk - EARTH_ROTATION_RATE * ephemeris.toe
    position = np.array([
        orbital_x * math.cos(longitude) - orbital_y * math.cos(inclination) * math.sin(longitude),
        orbital_x * math.sin(longitude) + orbital_y * math.cos(inclination) * math.cos(longitude),
        orbital_y * math.sin(inclination),
    ])

    dt = _wrap_week(gps_time - ephemeris.toc)
    relativity = RELATIVISTIC_CLOCK_CONSTANT * e * ephemeris.sqrt_a * math.sin(eccentric_anomaly)
    if isinstance(ephemeris, GpsEphemeris):
        group_delay = ephemeris.tgd
    elif isinstance(ephemeris, GalileoEphemeris):
        group_delay = ephemeris.bgd_e5b_e1
    else:
        # The configured C7I/C6I observable is an ionosphere-free B2I/B3I
        # combination. B3I is the clock reference and TGD2 applies to B2I.
        f2_sq = BEIDOU_B2I_FREQUENCY_HZ**2
        f3_sq = BEIDOU_B3I_FREQUENCY_HZ**2
        group_delay = f2_sq / (f2_sq - f3_sq) * ephemeris.tgd_b2_b3
    if not apply_group_delay:
        group_delay = 0.0
    clock = ephemeris.af0 + ephemeris.af1 * dt + ephemeris.af2 * dt * dt + relativity - group_delay
    return position, clock


def propagate_gps(
    ephemeris: GpsEphemeris, transmit_time: float,
    apply_group_delay: bool = True,
) -> SatelliteState:
    """Propagate broadcast ephemeris; velocity and drift use a stable central difference."""
    step = 0.5
    position, clock = _position_and_clock(ephemeris, transmit_time, GPS_GRAVITATIONAL_CONSTANT, apply_group_delay)
    before_position, before_clock = _position_and_clock(ephemeris, transmit_time - step, GPS_GRAVITATIONAL_CONSTANT, apply_group_delay)
    after_position, after_clock = _position_and_clock(ephemeris, transmit_time + step, GPS_GRAVITATIONAL_CONSTANT, apply_group_delay)
    return SatelliteState(
        satellite=ephemeris.satellite,
        transmit_time=transmit_time,
        position_ecef=position,
        velocity_ecef=(after_position - before_position) / (2.0 * step),
        clock_bias_s=clock,
        clock_drift_sps=(after_clock - before_clock) / (2.0 * step),
    )


def propagate_galileo(
    ephemeris: GalileoEphemeris, transmit_time: float,
    apply_group_delay: bool = True,
) -> SatelliteState:
    """Propagate a Galileo broadcast ephemeris and satellite clock."""
    step = 0.5
    position, clock = _position_and_clock(ephemeris, transmit_time, GALILEO_GRAVITATIONAL_CONSTANT, apply_group_delay)
    before_position, before_clock = _position_and_clock(ephemeris, transmit_time - step, GALILEO_GRAVITATIONAL_CONSTANT, apply_group_delay)
    after_position, after_clock = _position_and_clock(ephemeris, transmit_time + step, GALILEO_GRAVITATIONAL_CONSTANT, apply_group_delay)
    return SatelliteState(
        satellite=ephemeris.satellite, transmit_time=transmit_time,
        position_ecef=position,
        velocity_ecef=(after_position - before_position) / (2.0 * step),
        clock_bias_s=clock,
        clock_drift_sps=(after_clock - before_clock) / (2.0 * step),
    )


def classify_beidou_orbit(ephemeris: BeidouEphemeris) -> str:
    """Classify BDS orbit using broadcast geometry rather than fragile PRNs."""
    semi_major_axis = ephemeris.sqrt_a**2
    if semi_major_axis < 35_000_000.0:
        return "MEO"
    if abs(ephemeris.i0) < math.radians(15.0):
        return "GEO"
    return "IGSO"


def _beidou_geo_position_and_clock(
    ephemeris: BeidouEphemeris, gps_time: float,
) -> tuple[np.ndarray, float]:
    """Propagate a BDS GEO using the ICD's extra -5 degree frame rotation."""
    toe_time = _toe_time_gps(ephemeris)
    tk = _wrap_week(gps_time - toe_time)
    a = ephemeris.sqrt_a**2
    mean_motion = math.sqrt(BEIDOU_GRAVITATIONAL_CONSTANT / a**3) + ephemeris.delta_n
    mean_anomaly = ephemeris.m0 + mean_motion * tk
    eccentric_anomaly = mean_anomaly
    for _ in range(12):
        updated = mean_anomaly + ephemeris.eccentricity * math.sin(eccentric_anomaly)
        if abs(updated - eccentric_anomaly) < 1e-13:
            break
        eccentric_anomaly = updated
    e = ephemeris.eccentricity
    true_anomaly = math.atan2(
        math.sqrt(1.0 - e * e) * math.sin(eccentric_anomaly),
        math.cos(eccentric_anomaly) - e,
    )
    phi = true_anomaly + ephemeris.argument_of_perigee
    twice_phi = 2.0 * phi
    argument = phi + ephemeris.cus * math.sin(twice_phi) + ephemeris.cuc * math.cos(twice_phi)
    radius = a * (1.0 - e * math.cos(eccentric_anomaly)) + ephemeris.crs * math.sin(twice_phi) + ephemeris.crc * math.cos(twice_phi)
    inclination = ephemeris.i0 + ephemeris.idot * tk + ephemeris.cis * math.sin(twice_phi) + ephemeris.cic * math.cos(twice_phi)
    orbital_x, orbital_y = radius * math.cos(argument), radius * math.sin(argument)
    longitude = ephemeris.omega0 + ephemeris.omega_dot * tk - EARTH_ROTATION_RATE * ephemeris.toe
    xg = orbital_x * math.cos(longitude) - orbital_y * math.cos(inclination) * math.sin(longitude)
    yg = orbital_x * math.sin(longitude) + orbital_y * math.cos(inclination) * math.cos(longitude)
    zg = orbital_y * math.sin(inclination)
    earth_angle = EARTH_ROTATION_RATE * tk
    ce, se = math.cos(earth_angle), math.sin(earth_angle)
    ct, st = math.cos(BEIDOU_GEO_TILT_RAD), math.sin(BEIDOU_GEO_TILT_RAD)
    position = np.array([
        xg * ce + yg * se * ct + zg * se * st,
        -xg * se + yg * ce * ct + zg * ce * st,
        -yg * st + zg * ct,
    ])
    dt = _wrap_week(gps_time - ephemeris.toc)
    relativity = RELATIVISTIC_CLOCK_CONSTANT * e * ephemeris.sqrt_a * math.sin(eccentric_anomaly)
    f2_sq = BEIDOU_B2I_FREQUENCY_HZ**2
    f3_sq = BEIDOU_B3I_FREQUENCY_HZ**2
    group_delay = f2_sq / (f2_sq - f3_sq) * ephemeris.tgd_b2_b3
    clock = ephemeris.af0 + ephemeris.af1 * dt + ephemeris.af2 * dt * dt + relativity - group_delay
    return position, clock


def propagate_beidou(ephemeris: BeidouEphemeris, transmit_time: float) -> SatelliteState:
    """Propagate BDS MEO/IGSO with Kepler and GEO with its special frame."""
    position_clock = (
        _beidou_geo_position_and_clock
        if classify_beidou_orbit(ephemeris) == "GEO"
        else lambda item, time: _position_and_clock(
            item, time, BEIDOU_GRAVITATIONAL_CONSTANT
        )
    )
    step = 0.5
    position, clock = position_clock(ephemeris, transmit_time)
    before_position, before_clock = position_clock(ephemeris, transmit_time - step)
    after_position, after_clock = position_clock(ephemeris, transmit_time + step)
    return SatelliteState(
        satellite=ephemeris.satellite, transmit_time=transmit_time,
        position_ecef=position,
        velocity_ecef=(after_position - before_position) / (2.0 * step),
        clock_bias_s=clock,
        clock_drift_sps=(after_clock - before_clock) / (2.0 * step),
    )


def propagate(
    ephemeris: GpsEphemeris | GalileoEphemeris | BeidouEphemeris,
    transmit_time: float, apply_group_delay: bool = True,
) -> SatelliteState:
    """Dispatch broadcast propagation by ephemeris type."""
    if isinstance(ephemeris, GpsEphemeris):
        return propagate_gps(ephemeris, transmit_time, apply_group_delay)
    if isinstance(ephemeris, GalileoEphemeris):
        return propagate_galileo(ephemeris, transmit_time, apply_group_delay)
    if isinstance(ephemeris, BeidouEphemeris):
        return propagate_beidou(ephemeris, transmit_time)
    raise TypeError(f"unsupported ephemeris type: {type(ephemeris).__name__}")


def rotate_for_sagnac(vector_ecef: np.ndarray, signal_travel_time_s: float) -> np.ndarray:
    """Rotate a transmit-time ECEF vector into the receive-time ECEF frame."""
    angle = EARTH_ROTATION_RATE * signal_travel_time_s
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, sine, 0.0], [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return rotation @ np.asarray(vector_ecef, dtype=float)
