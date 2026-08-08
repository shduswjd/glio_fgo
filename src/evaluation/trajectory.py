"""Evaluate the campus01 GNSS/IMU FGO trajectory against RTK ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution (``python agr_fgo/src/evaluation/trajectory.py``)
# without requiring callers to set PYTHONPATH=agr_fgo/src.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sensors.gnss_corrections import ecef_R_enu


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FGO = PROJECT_ROOT / "agr_fgo/results/great_data/tc_trajectory.csv"
DEFAULT_GT = PROJECT_ROOT / "great_data/groundtruth01.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "agr_fgo/results/great_data"
GPS_WEEK_SECONDS = 604_800.0


def load_trajectories(
    fgo_file: Path, gt_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FGO CSV and campus ground truth using one absolute GPST axis."""
    fgo = pd.read_csv(fgo_file)
    required = {
        "gps_time_s", "antenna_ecef_x_m", "antenna_ecef_y_m",
        "antenna_ecef_z_m", "roll", "pitch", "yaw",
    }
    missing = required.difference(fgo.columns)
    if missing:
        raise ValueError(
            "FGO CSV is missing columns (rerun tightly_coupled_fgo.py): "
            + ", ".join(sorted(missing))
        )

    # groundtruth01 columns begin with: GPS week, SOW, ECEF X/Y/Z. The
    # latitude/longitude DMS fields later on are intentionally not loaded.
    raw_gt = np.loadtxt(
        gt_file, comments="#", usecols=(0, 1, 2, 3, 4, 21, 22, 23), ndmin=2,
    )
    gt = pd.DataFrame(raw_gt, columns=[
        "gps_week", "gps_sow", "x", "y", "z",
        "heading_deg", "pitch_deg", "roll_deg",
    ])
    gt["gps_time_s"] = gt["gps_week"] * GPS_WEEK_SECONDS + gt["gps_sow"]
    fgo = fgo.sort_values("gps_time_s").drop_duplicates("gps_time_s")
    gt = gt.sort_values("gps_time_s").drop_duplicates("gps_time_s")
    return fgo, gt


def synchronize_trajectories(fgo: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Interpolate 10 Hz RTK ECEF at 1 Hz FGO times and compute local ENU error."""
    start = max(fgo["gps_time_s"].min(), gt["gps_time_s"].min())
    end = min(fgo["gps_time_s"].max(), gt["gps_time_s"].max())
    if start > end:
        raise ValueError("FGO and ground-truth GPS timestamps do not overlap")
    synced = fgo[(fgo["gps_time_s"] >= start) & (fgo["gps_time_s"] <= end)].copy()
    times = synced["gps_time_s"].to_numpy()
    gt_ecef = np.column_stack([
        np.interp(times, gt["gps_time_s"], gt[axis]) for axis in ("x", "y", "z")
    ])
    estimated_ecef = synced[[
        "antenna_ecef_x_m", "antenna_ecef_y_m", "antenna_ecef_z_m",
    ]].to_numpy()
    reference = gt_ecef[0]
    rotation = ecef_R_enu(reference).T  # ECEF delta -> ENU
    gt_enu = (rotation @ (gt_ecef - reference).T).T
    estimated_enu = (rotation @ (estimated_ecef - reference).T).T
    error = estimated_enu - gt_enu
    for index, axis in enumerate(("east", "north", "up")):
        synced[f"gt_{axis}_m"] = gt_enu[:, index]
        synced[f"fgo_antenna_{axis}_m"] = estimated_enu[:, index]
        synced[f"error_{axis}_m"] = error[:, index]
    # GT heading is clockwise from North. FGO yaw is the ENU mathematical
    # angle (counter-clockwise from East), hence yaw = 90 deg - heading.
    gt_yaw = np.unwrap(np.deg2rad(90.0 - gt["heading_deg"].to_numpy()))
    synced["gt_yaw_rad"] = np.interp(times, gt["gps_time_s"], gt_yaw)
    # The reference INS reports positive pitch nose-down, whereas GTSAM's FLU
    # RzRyRx convention is positive nose-up.
    synced["gt_pitch_rad"] = np.interp(
        times, gt["gps_time_s"], -np.deg2rad(gt["pitch_deg"]),
    )
    synced["gt_roll_rad"] = np.interp(
        times, gt["gps_time_s"], np.deg2rad(gt["roll_deg"]),
    )
    synced["error_horizontal_m"] = np.linalg.norm(error[:, :2], axis=1)
    synced["error_3d_m"] = np.linalg.norm(error, axis=1)
    return synced


def align_antenna_trajectory_svd(synced: pd.DataFrame) -> pd.DataFrame:
    """Return a diagnostic rigid-alignment ATE without changing raw metrics."""
    estimated = synced[[
        "fgo_antenna_east_m", "fgo_antenna_north_m", "fgo_antenna_up_m",
    ]].to_numpy()
    truth = synced[["gt_east_m", "gt_north_m", "gt_up_m"]].to_numpy()
    if len(estimated) < 3:
        raise ValueError("at least three synchronized positions are required for SVD")
    estimated_centered = estimated - estimated.mean(axis=0)
    truth_centered = truth - truth.mean(axis=0)
    u, _, vt = np.linalg.svd(estimated_centered.T @ truth_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = truth.mean(axis=0) - rotation @ estimated.mean(axis=0)
    aligned = (rotation @ estimated.T).T + translation
    result = synced.copy()
    result[["aligned_fgo_east_m", "aligned_fgo_north_m", "aligned_fgo_up_m"]] = aligned
    result["ate_m"] = np.linalg.norm(aligned - truth, axis=1)
    return result


def print_metrics(synced: pd.DataFrame) -> None:
    rmse = lambda name: float(np.sqrt(np.mean(synced[name].to_numpy() ** 2)))
    print(f"Synchronized samples: {len(synced)}")
    print(f"East RMSE:       {rmse('error_east_m'):.3f} m")
    print(f"North RMSE:      {rmse('error_north_m'):.3f} m")
    print(f"Up RMSE:         {rmse('error_up_m'):.3f} m")
    print(f"Horizontal RMSE: {rmse('error_horizontal_m'):.3f} m")
    print(f"3D RMSE:         {rmse('error_3d_m'):.3f} m")
    if "ate_m" in synced:
        print(f"SVD-aligned ATE RMSE (diagnostic): {rmse('ate_m'):.3f} m")


def save_plot(synced: pd.DataFrame, output_path: Path) -> None:
    elapsed = synced["gps_time_s"] - synced["gps_time_s"].iloc[0]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(synced["gt_east_m"], synced["gt_north_m"], label="RTK ground truth")
    axes[0].plot(
        synced["fgo_antenna_east_m"], synced["fgo_antenna_north_m"], label="FGO antenna",
    )
    axes[0].set(xlabel="East [m]", ylabel="North [m]", title="campus01 trajectory")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()
    for axis in ("east", "north", "up"):
        axes[1].plot(elapsed, synced[f"error_{axis}_m"], label=axis.title())
    axes[1].set(xlabel="Elapsed GPST [s]", ylabel="FGO - RTK [m]", title="Position error")
    axes[1].grid(True)
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_rpy_plot(synced: pd.DataFrame, output_path: Path) -> None:
    """Plot FGO attitude and synchronized ground-truth attitude."""
    elapsed = (synced["gps_time_s"] - synced["gps_time_s"].iloc[0]).to_numpy()
    fgo_angles = {
        "Roll": np.rad2deg(synced["roll"].to_numpy()),
        "Pitch": np.rad2deg(synced["pitch"].to_numpy()),
        "Yaw": np.rad2deg(np.unwrap(synced["yaw"].to_numpy())),
    }
    gt_angles = {
        "Roll": np.rad2deg(synced["gt_roll_rad"].to_numpy()),
        "Pitch": np.rad2deg(synced["gt_pitch_rad"].to_numpy()),
        "Yaw": np.rad2deg(synced["gt_yaw_rad"].to_numpy()),
    }
    # Put the unwrapped yaw curves on the nearest common 360-degree branch.
    gt_angles["Yaw"] += 360.0 * np.round(
        (fgo_angles["Yaw"][0] - gt_angles["Yaw"][0]) / 360.0
    )
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, name in zip(axes, ("Roll", "Pitch", "Yaw")):
        axis.plot(elapsed, fgo_angles[name], label=f"FGO {name.lower()}")
        axis.plot(elapsed, gt_angles[name], "k--", linewidth=1.1, label=f"GT {name.lower()}")
        axis.set_ylabel(f"{name} [deg]")
        axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("Elapsed GPST [s]")
    figure.suptitle("campus01 attitude (GT heading converted to ENU yaw)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fgo", type=Path, default=DEFAULT_FGO)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fgo, gt = load_trajectories(args.fgo, args.gt)
    synced = synchronize_trajectories(fgo, gt)
    if len(synced) >= 3:
        synced = align_antenna_trajectory_svd(synced)
    print_metrics(synced)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "trajectory_gal_gps_error.csv"
    plot_path = args.output_dir / "trajectory_gal_gps_comparison.png"
    rpy_path = args.output_dir / "rpy_comparison_gal_gps.png"
    synced.to_csv(csv_path, index=False)
    save_plot(synced, plot_path)
    save_rpy_plot(synced, rpy_path)
    print(f"Error CSV: {csv_path.resolve()}")
    print(f"Plot:      {plot_path.resolve()}")
    print(f"RPY plot:  {rpy_path.resolve()}")


if __name__ == "__main__":
    main()
