"""Compute and plot constellation-wise post-fit pseudorange residuals."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_tc_gnss import configured_signals
from sensors.gnss_preprocessor import prepare_epoch
from sensors.gnss_raw import load_tc_config, read_rinex_observations, read_tc_navigation
from sensors.gnss_satellite import EphemerisStore, SPEED_OF_LIGHT


SYSTEM_NAMES = {"G": "GPS", "E": "Galileo", "C": "BeiDou"}
SYSTEM_COLORS = {"G": "tab:blue", "E": "tab:orange", "C": "tab:green"}


def compute_residuals(config_path: Path, trajectory_path: Path) -> pd.DataFrame:
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dataset = load_tc_config(config_path)
    raw = config["gnss_raw"]
    signals = configured_signals(raw)
    ephemerides = EphemerisStore.from_rinex(read_tc_navigation(config_path))
    trajectory = pd.read_csv(trajectory_path)
    by_time = {round(float(row.gps_time_s), 3): row for row in trajectory.itertuples()}
    first_time = float(trajectory["gps_time_s"].iloc[0])
    last_time = float(trajectory["gps_time_s"].iloc[-1])
    weighting = raw.get("weighting", {})
    minimum_cn0 = weighting.get("minimum_cn0_dbhz")
    residual_gate = weighting.get("residual_gate_m")
    beidou_orbits = {str(item) for item in raw.get("beidou_orbits", ["MEO", "IGSO", "GEO"])}
    rows: list[dict] = []
    for epoch in read_rinex_observations(dataset.observation_path):
        gps_time = epoch.gps_seconds
        if gps_time < first_time:
            continue
        if gps_time > last_time:
            break
        state = by_time.get(round(gps_time, 3))
        if state is None:
            continue
        receiver = np.array([
            state.antenna_ecef_x_m, state.antenna_ecef_y_m, state.antenna_ecef_z_m,
        ])
        prepared = prepare_epoch(
            epoch, receiver, ephemerides,
            minimum_elevation_deg=float(raw["minimum_elevation_deg"]),
            signals=signals, beidou_orbits=beidou_orbits,
            minimum_cn0_dbhz=None if minimum_cn0 is None else float(minimum_cn0),
            residual_gate_m=None if residual_gate is None else float(residual_gate),
        )
        for measurement in prepared:
            if measurement.pseudorange_m is None:
                continue
            system = measurement.satellite[0]
            isb = (
                state.galileo_isb_m if system == "E" else
                state.beidou_isb_m if system == "C" else 0.0
            )
            wet = 0.0
            if hasattr(state, "wet_delay_delta_m") and np.isfinite(state.wet_delay_delta_m):
                wet = state.wet_delay_delta_m / max(
                    np.sin(float(measurement.elevation_rad)), np.sin(np.deg2rad(3.0))
                )
            geometric_range = np.linalg.norm(
                measurement.satellite_state.position_ecef - receiver
            )
            predicted = (
                geometric_range + state.clock_bias_m + isb
                - SPEED_OF_LIGHT * measurement.satellite_state.clock_bias_s
                + measurement.atmosphere.total_m + wet
            )
            rows.append({
                "gps_time_s": gps_time,
                "elapsed_s": gps_time - first_time,
                "satellite": measurement.satellite,
                "system": system,
                "elevation_deg": np.rad2deg(measurement.elevation_rad),
                "cn0_dbhz": measurement.cn0_dbhz,
                "residual_m": predicted - measurement.pseudorange_m,
            })
    return pd.DataFrame(rows)


def save_plot(residuals: pd.DataFrame, output_path: Path) -> None:
    systems = [system for system in ("G", "E", "C") if system in set(residuals["system"])]
    figure, axes = plt.subplots(len(systems), 1, figsize=(13, 3.2 * len(systems)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, system in zip(axes, systems):
        selected = residuals[residuals["system"] == system]
        values = selected["residual_m"].to_numpy()
        bias = float(np.mean(values))
        rmse = float(np.sqrt(np.mean(values**2)))
        axis.scatter(
            selected["elapsed_s"], values, s=7, alpha=0.45,
            color=SYSTEM_COLORS[system], edgecolors="none",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhline(bias, color="red", linestyle="--", linewidth=1.0,
                     label=f"mean={bias:.2f} m")
        axis.set_ylabel("Residual [m]")
        axis.set_title(
            f"{SYSTEM_NAMES[system]} post-fit code residuals "
            f"(N={len(values)}, RMSE={rmse:.2f} m)"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Elapsed GPST [s]")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    residuals = compute_residuals(args.config, args.trajectory)
    if residuals.empty:
        raise RuntimeError("no synchronized GNSS residuals were produced")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "gnss_postfit_residuals.csv"
    plot_path = args.output_dir / "gnss_postfit_residuals.png"
    residuals.to_csv(csv_path, index=False)
    save_plot(residuals, plot_path)
    for system, selected in residuals.groupby("system"):
        values = selected["residual_m"].to_numpy()
        print(
            f"{SYSTEM_NAMES.get(system, system)}: N={len(values)} "
            f"mean={np.mean(values):.3f} m "
            f"MAE={np.mean(np.abs(values)):.3f} m "
            f"RMSE={np.sqrt(np.mean(values**2)):.3f} m"
        )
    print(f"Residual CSV: {csv_path.resolve()}")
    print(f"Residual plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
