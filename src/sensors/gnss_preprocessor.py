"""Turn raw epoch observations into factor-ready, corrected measurements."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from sensors.gnss_corrections import (
    KlobucharParameters, azimuth_elevation, ecef_to_geodetic,
    klobuchar_delay, saastamoinen_delay,
)
from sensors.gnss_models import AtmosphericCorrection, CorrectedGnssMeasurement, SatelliteState
from sensors.gnss_raw import ObservationEpoch
from sensors.gnss_models import BeidouEphemeris
from sensors.gnss_satellite import (
    SPEED_OF_LIGHT, EphemerisStore, classify_beidou_orbit, propagate,
    rotate_for_sagnac,
)


def prepare_epoch(
    epoch: ObservationEpoch,
    receiver_position_ecef: np.ndarray,
    ephemerides: EphemerisStore,
    ionosphere: KlobucharParameters | None = None,
    signal: str = "1C",
    secondary_signal: str | None = None,
    minimum_elevation_deg: float = 10.0,
    signals: Mapping[str, tuple[str, str | None]] | None = None,
    beidou_orbits: set[str] | None = None,
    minimum_cn0_dbhz: float | None = None,
    residual_gate_m: float | None = None,
) -> list[CorrectedGnssMeasurement]:
    """Prepare GPS/Galileo observations using the current receiver estimate.

    The receiver position is deliberately an argument: transmit time, Sagnac,
    elevation, and atmospheric delay must be recomputed as the FGO estimate
    changes (or at least once per outer optimization iteration).
    """
    receiver = np.asarray(receiver_position_ecef, dtype=float).reshape(3)
    receive_time = epoch.gps_seconds
    result: list[CorrectedGnssMeasurement] = []
    latitude, longitude, height = ecef_to_geodetic(receiver)

    for observation in epoch.satellites:
        system = observation.satellite[0]
        if signals is not None:
            if system not in signals:
                continue
            observation_signal, observation_secondary = signals[system]
        else:
            if system != "G":
                continue
            observation_signal, observation_secondary = signal, secondary_signal
        pseudorange = (
            observation.pseudorange(observation_signal)
            if observation_secondary is None
            else observation.ionosphere_free_pseudorange(
                observation_signal, observation_secondary
            )
        )
        if pseudorange is None:
            continue
        transmit_time = receive_time - pseudorange / SPEED_OF_LIGHT
        ephemeris = ephemerides.nearest(observation.satellite, transmit_time)
        orbit_type = (
            classify_beidou_orbit(ephemeris)
            if isinstance(ephemeris, BeidouEphemeris) else None
        )
        if orbit_type is not None and beidou_orbits is not None and orbit_type not in beidou_orbits:
            continue
        state = propagate(ephemeris, transmit_time)
        travel_time = receive_time - transmit_time
        position = rotate_for_sagnac(state.position_ecef, travel_time)
        velocity = rotate_for_sagnac(state.velocity_ecef, travel_time)
        state = SatelliteState(
            satellite=state.satellite, transmit_time=state.transmit_time,
            position_ecef=position, velocity_ecef=velocity,
            clock_bias_s=state.clock_bias_s, clock_drift_sps=state.clock_drift_sps,
        )
        azimuth, elevation = azimuth_elevation(receiver, position)
        if elevation < np.deg2rad(minimum_elevation_deg):
            continue
        cn0_values = [observation.cn0(observation_signal)]
        if observation_secondary is not None:
            cn0_values.append(observation.cn0(observation_secondary))
        available_cn0 = [value for value in cn0_values if value is not None]
        cn0 = min(available_cn0) if available_cn0 else None
        if minimum_cn0_dbhz is not None and cn0 is not None and cn0 < minimum_cn0_dbhz:
            continue
        # The dual-frequency linear combination removes the first-order
        # ionospheric term, so a broadcast single-frequency correction must
        # not be applied on top of it.
        ionosphere_m = 0.0 if observation_secondary is not None or ionosphere is None else klobuchar_delay(
            epoch.gps_seconds_of_week, latitude, longitude, azimuth, elevation, ionosphere
        )
        atmosphere = AtmosphericCorrection(
            ionosphere_m=ionosphere_m,
            troposphere_m=saastamoinen_delay(latitude, height, elevation),
        )
        result.append(CorrectedGnssMeasurement(
            satellite=observation.satellite, receive_time=receive_time,
            satellite_state=state, pseudorange_m=pseudorange,
            range_rate_mps=observation.range_rate(observation_signal), atmosphere=atmosphere,
            signal=(observation_signal if observation_secondary is None else
                    f"IF({observation_signal},{observation_secondary})"),
            elevation_rad=float(elevation), cn0_dbhz=cn0,
            orbit_type=orbit_type,
        ))
    if residual_gate_m is None:
        return result
    accepted: list[CorrectedGnssMeasurement] = []
    for system in {item.satellite[0] for item in result}:
        group = [item for item in result if item.satellite.startswith(system)]
        if len(group) < 3:
            accepted.extend(group)
            continue
        residuals = np.array([
            float(item.pseudorange_m)
            - (
                np.linalg.norm(item.satellite_state.position_ecef - receiver)
                - SPEED_OF_LIGHT * item.satellite_state.clock_bias_s
                + item.atmosphere.total_m
            )
            for item in group
        ])
        center = float(np.median(residuals))
        accepted.extend(
            item for item, residual in zip(group, residuals)
            if abs(float(residual) - center) <= residual_gate_m
        )
    order = {item.satellite: index for index, item in enumerate(result)}
    return sorted(accepted, key=lambda item: order[item.satellite])
