"""Run the raw-GNSS front end configured by config/tc.yaml.

This is an inspectable entry point before wiring the measurements into the
full IMU factor graph.  It validates paths, bootstraps receiver ECEF with SPP,
and prints factor-ready pseudorange/Doppler observations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from sensors.gnss_preprocessor import prepare_epoch
from sensors.gnss_raw import (
    ObservationEpoch, load_tc_config, read_tc_navigation, read_tc_observations,
)
from sensors.gnss_satellite import (
    SPEED_OF_LIGHT, EphemerisStore, propagate_gps, rotate_for_sagnac,
)


def configured_signals(options: dict) -> dict[str, tuple[str, str | None]]:
    """Load per-constellation signals, retaining the legacy GPS options."""
    configured = options.get("signals")
    if configured is None:
        secondary = options.get("secondary_signal")
        return {"G": (str(options.get("signal", "1C")),
                      None if secondary is None else str(secondary))}
    if not isinstance(configured, dict):
        raise ValueError("gnss_raw.signals must be a YAML mapping")
    result: dict[str, tuple[str, str | None]] = {}
    for system, item in configured.items():
        if not isinstance(item, dict) or not item.get("primary"):
            raise ValueError(f"gnss_raw.signals.{system} requires primary")
        secondary = item.get("secondary")
        result[str(system)] = (
            str(item["primary"]), None if secondary is None else str(secondary)
        )
    return result


def estimate_receiver_position(
    epoch: ObservationEpoch,
    ephemerides: EphemerisStore,
    signal: str = "1C",
    secondary_signal: str | None = None,
) -> tuple[np.ndarray, float]:
    """Estimate receiver ECEF and clock bias [m] from one GPS epoch."""
    usable = [
        observation for observation in epoch.satellites
        if observation.satellite.startswith("G")
        and (observation.pseudorange(signal) if secondary_signal is None else
             observation.ionosphere_free_pseudorange(signal, secondary_signal)) is not None
    ]
    if len(usable) < 4:
        raise ValueError("SPP initialization requires at least four GPS pseudoranges")

    position = np.zeros(3)
    clock_bias_m = 0.0
    for _ in range(10):
        design_rows: list[np.ndarray] = []
        innovations: list[float] = []
        for observation in usable:
            pseudorange = float(
                observation.pseudorange(signal) if secondary_signal is None else
                observation.ionosphere_free_pseudorange(signal, secondary_signal)
            )
            transmit_time = epoch.gps_seconds - pseudorange / SPEED_OF_LIGHT
            ephemeris = ephemerides.nearest(observation.satellite, transmit_time)
            satellite = propagate_gps(
                ephemeris, transmit_time,
                apply_group_delay=secondary_signal is None,
            )
            satellite_position = rotate_for_sagnac(
                satellite.position_ecef, epoch.gps_seconds - transmit_time
            )
            delta = satellite_position - position
            distance = np.linalg.norm(delta)
            line_of_sight = delta / distance
            predicted = distance + clock_bias_m - SPEED_OF_LIGHT * satellite.clock_bias_s
            design_rows.append(np.r_[-line_of_sight, 1.0])
            innovations.append(pseudorange - predicted)
        update, *_ = np.linalg.lstsq(
            np.vstack(design_rows), np.asarray(innovations), rcond=None
        )
        position += update[:3]
        clock_bias_m += update[3]
        if np.linalg.norm(update[:3]) < 1e-3:
            break
    return position, clock_bias_m


def estimate_receiver_clock_bias(
    epoch: ObservationEpoch,
    ephemerides: EphemerisStore,
    receiver_position_ecef: np.ndarray,
    signal: str = "1C",
    secondary_signal: str | None = None,
) -> float:
    """Estimate only receiver clock bias [m] at a known ECEF position."""
    position = np.asarray(receiver_position_ecef, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("receiver_position_ecef must contain three finite values")
    clock_estimates: list[float] = []
    for observation in epoch.satellites:
        if not observation.satellite.startswith("G"):
            continue
        pseudorange = (
            observation.pseudorange(signal)
            if secondary_signal is None else
            observation.ionosphere_free_pseudorange(signal, secondary_signal)
        )
        if pseudorange is None:
            continue
        transmit_time = epoch.gps_seconds - pseudorange / SPEED_OF_LIGHT
        ephemeris = ephemerides.nearest(observation.satellite, transmit_time)
        satellite = propagate_gps(
            ephemeris, transmit_time,
            apply_group_delay=secondary_signal is None,
        )
        satellite_position = rotate_for_sagnac(
            satellite.position_ecef, epoch.gps_seconds - transmit_time
        )
        geometric_range = np.linalg.norm(satellite_position - position)
        clock_estimates.append(
            float(pseudorange) - geometric_range
            + SPEED_OF_LIGHT * satellite.clock_bias_s
        )
    if len(clock_estimates) < 4:
        raise ValueError("clock initialization requires at least four GPS pseudoranges")
    # A median prevents one urban-code outlier from shifting every clock state.
    return float(np.median(clock_estimates))


def _raw_options(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    options = config.get("gnss_raw", {})
    if not isinstance(options, dict):
        raise ValueError("gnss_raw must be a YAML mapping")
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tc.yaml raw-GNSS preprocessing")
    default_config = Path(__file__).resolve().parents[1] / "config" / "tc.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs to print")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    dataset = load_tc_config(args.config)
    options = _raw_options(args.config)
    signals = configured_signals(options)
    signal, secondary_signal = signals.get("G", ("1C", None))
    minimum_elevation = float(options.get("minimum_elevation_deg", 10.0))
    beidou_orbits = {str(item) for item in options.get(
        "beidou_orbits", ["MEO", "IGSO", "GEO"]
    )}
    weighting = options.get("weighting", {})
    minimum_cn0 = weighting.get("minimum_cn0_dbhz")
    residual_gate = weighting.get("residual_gate_m")
    ephemerides = EphemerisStore.from_rinex(read_tc_navigation(args.config))
    observations = read_tc_observations(args.config)
    first_epoch = next(observations, None)
    if first_epoch is None:
        raise RuntimeError("observation file contains no usable epochs")

    configured_position = options.get("initial_position_ecef")
    if configured_position is None:
        receiver_position, receiver_clock = estimate_receiver_position(
            first_epoch, ephemerides, signal, secondary_signal
        )
        position_source = "SPP"
    else:
        receiver_position = np.asarray(configured_position, dtype=float).reshape(3)
        receiver_clock = estimate_receiver_clock_bias(
            first_epoch, ephemerides, receiver_position, signal, secondary_signal
        )
        position_source = "config"

    print(f"OBS: {dataset.observation_path}")
    print(f"NAV: {dataset.navigation_path}")
    print(f"initial ECEF ({position_source}): {receiver_position.round(3).tolist()} m")
    if np.isfinite(receiver_clock):
        print(f"initial receiver clock bias: {receiver_clock:.3f} m")

    epochs = [first_epoch]
    for _ in range(args.epochs - 1):
        epoch = next(observations, None)
        if epoch is None:
            break
        epochs.append(epoch)
    for epoch in epochs:
        prepared = prepare_epoch(
            epoch, receiver_position, ephemerides, signal=signal,
            secondary_signal=secondary_signal, minimum_elevation_deg=minimum_elevation,
            signals=signals,
            beidou_orbits=beidou_orbits,
            minimum_cn0_dbhz=(None if minimum_cn0 is None else float(minimum_cn0)),
            residual_gate_m=(None if residual_gate is None else float(residual_gate)),
        )
        code_count = sum(item.pseudorange_m is not None for item in prepared)
        doppler_count = sum(item.range_rate_mps is not None for item in prepared)
        satellites = " ".join(item.satellite for item in prepared)
        print(
            f"{epoch.time.isoformat()} code={code_count} doppler={doppler_count} "
            f"satellites=[{satellites}]"
        )


if __name__ == "__main__":
    main()
