"""Typed data passed from RINEX parsing to tightly-coupled GNSS factors.

Use dataclasses for records with a fixed physical meaning.  Dictionaries are
reserved for lookup (for example ``ephemerides_by_satellite['G05']``), not for
representing a measurement itself.  All distances are metres and all times
are GPS seconds unless a field says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Vector3 = np.ndarray


def _vector3(value: Vector3, name: str) -> Vector3:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return result.copy()


@dataclass(frozen=True)
class GpsEphemeris:
    """GPS broadcast ephemeris with named ICD fields (SI units)."""

    satellite: str
    toc: float
    af0: float
    af1: float
    af2: float
    iode: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    eccentricity: float
    cus: float
    sqrt_a: float
    toe: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    argument_of_perigee: float
    omega_dot: float
    idot: float
    gps_week: int
    tgd: float


@dataclass(frozen=True)
class GalileoEphemeris:
    """Galileo broadcast ephemeris with named RINEX fields (SI units)."""

    satellite: str
    toc: float
    af0: float
    af1: float
    af2: float
    iodnav: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    eccentricity: float
    cus: float
    sqrt_a: float
    toe: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    argument_of_perigee: float
    omega_dot: float
    idot: float
    gps_week: int
    bgd_e5a_e1: float
    bgd_e5b_e1: float


@dataclass(frozen=True)
class BeidouEphemeris:
    """BeiDou broadcast ephemeris expressed on the GPS absolute time axis."""

    satellite: str
    toc: float
    af0: float
    af1: float
    af2: float
    aode: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    eccentricity: float
    cus: float
    sqrt_a: float
    toe: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    argument_of_perigee: float
    omega_dot: float
    idot: float
    gps_week: int
    tgd_b1_b3: float
    tgd_b2_b3: float


@dataclass(frozen=True)
class SatelliteState:
    """Satellite state at signal transmission time, in ECEF."""

    satellite: str
    transmit_time: float
    position_ecef: Vector3
    velocity_ecef: Vector3
    clock_bias_s: float
    clock_drift_sps: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_ecef", _vector3(self.position_ecef, "position_ecef"))
        object.__setattr__(self, "velocity_ecef", _vector3(self.velocity_ecef, "velocity_ecef"))


@dataclass(frozen=True)
class AtmosphericCorrection:
    ionosphere_m: float = 0.0
    troposphere_m: float = 0.0

    @property
    def total_m(self) -> float:
        return self.ionosphere_m + self.troposphere_m


@dataclass(frozen=True)
class CorrectedGnssMeasurement:
    """One observation after satellite propagation and model corrections."""

    satellite: str
    receive_time: float
    satellite_state: SatelliteState
    pseudorange_m: float | None
    range_rate_mps: float | None
    atmosphere: AtmosphericCorrection = AtmosphericCorrection()
    signal: str = "1C"
    elevation_rad: float | None = None
    cn0_dbhz: float | None = None
    orbit_type: str | None = None
