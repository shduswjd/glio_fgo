"""Small, explicit atmospheric models for single-frequency GNSS."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


SPEED_OF_LIGHT = 299_792_458.0
WGS84_A = 6_378_137.0
WGS84_E2 = 6.69437999014e-3


@dataclass(frozen=True)
class KlobucharParameters:
    alpha: tuple[float, float, float, float]
    beta: tuple[float, float, float, float]


def ecef_to_geodetic(position_ecef: np.ndarray) -> tuple[float, float, float]:
    """Return latitude [rad], longitude [rad], and ellipsoidal height [m]."""
    x, y, z = np.asarray(position_ecef, dtype=float).reshape(3)
    longitude = math.atan2(y, x)
    horizontal = math.hypot(x, y)
    latitude = math.atan2(z, horizontal * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_latitude = math.sin(latitude)
        radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_latitude**2)
        height = horizontal / max(math.cos(latitude), 1e-12) - radius
        latitude = math.atan2(z, horizontal * (1.0 - WGS84_E2 * radius / (radius + height)))
    radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(latitude) ** 2)
    height = horizontal / max(math.cos(latitude), 1e-12) - radius
    return latitude, longitude, height


def azimuth_elevation(receiver_ecef: np.ndarray, satellite_ecef: np.ndarray) -> tuple[float, float]:
    latitude, longitude, _ = ecef_to_geodetic(receiver_ecef)
    delta = np.asarray(satellite_ecef) - np.asarray(receiver_ecef)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
    east = -sin_lon * delta[0] + cos_lon * delta[1]
    north = -sin_lat * cos_lon * delta[0] - sin_lat * sin_lon * delta[1] + cos_lat * delta[2]
    up = cos_lat * cos_lon * delta[0] + cos_lat * sin_lon * delta[1] + sin_lat * delta[2]
    azimuth = math.atan2(east, north) % (2.0 * math.pi)
    elevation = math.atan2(up, math.hypot(east, north))
    return azimuth, elevation


def ecef_R_enu(reference_ecef: np.ndarray) -> np.ndarray:
    """Rotation mapping local ENU vectors into ECEF at ``reference_ecef``."""
    latitude, longitude, _ = ecef_to_geodetic(reference_ecef)
    sin_lat, cos_lat = math.sin(latitude), math.cos(latitude)
    sin_lon, cos_lon = math.sin(longitude), math.cos(longitude)
    return np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [0.0, cos_lat, sin_lat],
    ])


def klobuchar_delay(
    gps_seconds_of_week: float,
    receiver_latitude_rad: float,
    receiver_longitude_rad: float,
    azimuth_rad: float,
    elevation_rad: float,
    parameters: KlobucharParameters,
) -> float:
    """GPS ICD Klobuchar L1 group delay in metres."""
    elevation_sc = max(elevation_rad, 0.0) / math.pi
    latitude_sc = receiver_latitude_rad / math.pi
    longitude_sc = receiver_longitude_rad / math.pi
    azimuth = azimuth_rad
    psi = 0.0137 / (elevation_sc + 0.11) - 0.022
    sub_lat = min(0.416, max(-0.416, latitude_sc + psi * math.cos(azimuth)))
    sub_lon = longitude_sc + psi * math.sin(azimuth) / math.cos(sub_lat * math.pi)
    geomagnetic_lat = sub_lat + 0.064 * math.cos((sub_lon - 1.617) * math.pi)
    local_time = (43_200.0 * sub_lon + gps_seconds_of_week) % 86_400.0
    amplitude = max(0.0, sum(parameters.alpha[i] * geomagnetic_lat**i for i in range(4)))
    period = max(72_000.0, sum(parameters.beta[i] * geomagnetic_lat**i for i in range(4)))
    phase = 2.0 * math.pi * (local_time - 50_400.0) / period
    obliquity = 1.0 + 16.0 * (0.53 - elevation_sc) ** 3
    vertical_delay = 5e-9
    if abs(phase) < 1.57:
        vertical_delay += amplitude * (1.0 - phase**2 / 2.0 + phase**4 / 24.0)
    return SPEED_OF_LIGHT * obliquity * vertical_delay


def saastamoinen_delay(latitude_rad: float, height_m: float, elevation_rad: float) -> float:
    """Simple standard-atmosphere Saastamoinen slant delay in metres."""
    height = min(max(height_m, -100.0), 10_000.0)
    elevation = max(elevation_rad, math.radians(3.0))
    pressure_hpa = 1013.25 * (1.0 - 2.2557e-5 * height) ** 5.2568
    temperature_k = 288.15 - 0.0065 * height
    water_vapour_hpa = 11.7 * math.exp(-0.0006396 * height)
    zenith = math.pi / 2.0 - elevation
    hydrostatic = 0.0022768 * pressure_hpa / (1.0 - 0.00266 * math.cos(2.0 * latitude_rad) - 0.00028 * height / 1_000.0)
    wet = 0.002277 * (1255.0 / temperature_k + 0.05) * water_vapour_hpa
    mapping = 1.0 / math.cos(zenith)
    return (hydrostatic + wet) * mapping
