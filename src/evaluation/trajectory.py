"""Compare an optimized FGO trajectory with CitrusFarm RTK ground truth."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FGO_FILE = PROJECT_ROOT / "agr_fgo/results/01_13B_Jackal/trajectory.csv"
GT_FILE = PROJECT_ROOT / "01_13B_Jackal/ground_truth/gt.csv"
CONFIG_FILE = PROJECT_ROOT / "agr_fgo/config/default.yaml"
OUTPUT_DIR = PROJECT_ROOT / "agr_fgo/results/01_13B_Jackal"


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return R_world_body using the ZYX yaw-pitch-roll convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def load_trajectories(
    fgo_file: Path,
    gt_file: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and validate FGO and ground-truth trajectory files."""
    df_fgo = pd.read_csv(fgo_file)
    gt_columns = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    df_gt = pd.read_csv(
        gt_file,
        comment="#",
        header=None,
        names=gt_columns,
    )
    required_fgo = {
        "timestamp_s",
        "position_east_m",
        "position_north_m",
        "position_up_m",
        "roll",
        "pitch",
        "yaw",
    }
    missing = required_fgo.difference(df_fgo.columns)
    if missing:
        raise ValueError(f"FGO CSV is missing columns: {', '.join(sorted(missing))}")
    df_fgo = df_fgo.sort_values("timestamp_s").drop_duplicates("timestamp_s")
    df_gt = df_gt.sort_values("timestamp").drop_duplicates("timestamp")
    return df_fgo, df_gt


def synchronize_trajectories(
    df_fgo: pd.DataFrame,
    df_gt: pd.DataFrame,
    lever_arm_body: np.ndarray,
) -> pd.DataFrame:
    """Interpolate GT at FGO times and compare GPS antenna positions in ENU."""
    overlap_start = max(df_fgo["timestamp_s"].min(), df_gt["timestamp"].min())
    overlap_end = min(df_fgo["timestamp_s"].max(), df_gt["timestamp"].max())
    if overlap_start >= overlap_end:
        raise ValueError("FGO and ground-truth timestamps do not overlap")

    synced = df_fgo[
        (df_fgo["timestamp_s"] >= overlap_start)
        & (df_fgo["timestamp_s"] <= overlap_end)
    ].copy()
    timestamps = synced["timestamp_s"].to_numpy()

    # CitrusFarm rtk_path_frame uses x=North, y=-East, z=Up.
    synced["gt_east_m"] = -np.interp(timestamps, df_gt["timestamp"], df_gt["ty"])
    synced["gt_north_m"] = np.interp(timestamps, df_gt["timestamp"], df_gt["tx"])
    synced["gt_up_m"] = np.interp(timestamps, df_gt["timestamp"], df_gt["tz"])

    estimated_antenna = []
    for row in synced.itertuples():
        rotation = rotation_from_rpy(row.roll, row.pitch, row.yaw)
        base_position = np.array([
            row.position_east_m,
            row.position_north_m,
            row.position_up_m,
        ])
        estimated_antenna.append(base_position + rotation @ lever_arm_body)
    estimated_antenna = np.asarray(estimated_antenna)
    synced["fgo_antenna_east_m"] = estimated_antenna[:, 0]
    synced["fgo_antenna_north_m"] = estimated_antenna[:, 1]
    synced["fgo_antenna_up_m"] = estimated_antenna[:, 2]

    synced["error_east_m"] = synced["fgo_antenna_east_m"] - synced["gt_east_m"]
    synced["error_north_m"] = synced["fgo_antenna_north_m"] - synced["gt_north_m"]
    synced["error_up_m"] = synced["fgo_antenna_up_m"] - synced["gt_up_m"]
    synced["error_horizontal_m"] = np.hypot(
        synced["error_east_m"], synced["error_north_m"]
    )
    synced["error_3d_m"] = np.sqrt(
        synced["error_horizontal_m"] ** 2 + synced["error_up_m"] ** 2
    )
    return synced


def align_antenna_trajectory_svd(synced: pd.DataFrame) -> pd.DataFrame:
    """Rigidly align the estimated GPS-antenna trajectory to GT using SVD."""
    estimated = synced[[
        "fgo_antenna_east_m",
        "fgo_antenna_north_m",
        "fgo_antenna_up_m",
    ]].to_numpy()
    ground_truth = synced[["gt_east_m", "gt_north_m", "gt_up_m"]].to_numpy()
    if len(estimated) < 3:
        raise ValueError("at least three synchronized positions are required for SVD")

    estimated_centroid = estimated.mean(axis=0)
    ground_truth_centroid = ground_truth.mean(axis=0)
    estimated_centered = estimated - estimated_centroid
    ground_truth_centered = ground_truth - ground_truth_centroid
    u, _, vt = np.linalg.svd(estimated_centered.T @ ground_truth_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = ground_truth_centroid - rotation @ estimated_centroid
    aligned = (rotation @ estimated.T).T + translation

    result = synced.copy()
    result["aligned_fgo_east_m"] = aligned[:, 0]
    result["aligned_fgo_north_m"] = aligned[:, 1]
    result["aligned_fgo_up_m"] = aligned[:, 2]
    residual = aligned - ground_truth
    result["aligned_error_east_m"] = residual[:, 0]
    result["aligned_error_north_m"] = residual[:, 1]
    result["aligned_error_up_m"] = residual[:, 2]
    result["ate_m"] = np.linalg.norm(residual, axis=1)
    return result


def print_metrics(synced: pd.DataFrame) -> None:
    """Print raw ENU diagnostics and SVD-aligned ATE metrics."""
    rmse = lambda column: np.sqrt(np.mean(synced[column] ** 2))
    print(f"Synchronized samples: {len(synced)}")
    print(f"East RMSE:       {rmse('error_east_m'):.3f} m")
    print(f"North RMSE:      {rmse('error_north_m'):.3f} m")
    print(f"Up RMSE:         {rmse('error_up_m'):.3f} m")
    print(f"Up Max:          {synced['error_up_m'].abs().max():.3f} m")
    print(f"Horizontal RMSE: {rmse('error_horizontal_m'):.3f} m")
    print(f"3D RMSE:         {rmse('error_3d_m'):.3f} m")

    timestamps = synced["timestamp_s"].to_numpy()
    velocity_east = np.gradient(synced["gt_east_m"].to_numpy(), timestamps)
    velocity_north = np.gradient(synced["gt_north_m"].to_numpy(), timestamps)
    speed = np.hypot(velocity_east, velocity_north)
    gt_course = np.arctan2(velocity_north, velocity_east)
    yaw_error = np.arctan2(
        np.sin(synced["yaw"].to_numpy() - gt_course),
        np.cos(synced["yaw"].to_numpy() - gt_course),
    )
    moving = speed >= 0.2
    if np.any(moving):
        yaw_mae = np.rad2deg(np.mean(np.abs(yaw_error[moving])))
        yaw_rmse = np.rad2deg(np.sqrt(np.mean(yaw_error[moving] ** 2)))
        print(f"Heading/course consistency MAE:  {yaw_mae:.3f} deg")
        print(f"Heading/course consistency RMSE: {yaw_rmse:.3f} deg")

    ate = synced["ate_m"].to_numpy()
    print("SVD-aligned ATE (rigid SE(3), fixed scale):")
    print(f"  RMSE:   {np.sqrt(np.mean(ate**2)):.3f} m")
    print(f"  Mean:   {np.mean(ate):.3f} m")
    print(f"  Median: {np.median(ate):.3f} m")
    print(f"  Max:    {np.max(ate):.3f} m")


def save_plot(synced: pd.DataFrame, output_path: Path) -> None:
    """Save trajectory and raw ENU error plots."""
    elapsed = synced["timestamp_s"] - synced["timestamp_s"].iloc[0]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(synced["gt_east_m"], synced["gt_north_m"], label="Ground truth", linewidth=2)
    axes[0].plot(
        synced["fgo_antenna_east_m"], synced["fgo_antenna_north_m"],
        label="FGO antenna", linewidth=1.2,
    )
    axes[0].plot(
        synced["aligned_fgo_east_m"], synced["aligned_fgo_north_m"],
        label="FGO antenna (SVD aligned)", linewidth=1.2,
    )
    axes[0].set(xlabel="East [m]", ylabel="North [m]", title="Trajectory")
    axes[0].axis("equal")
    axes[0].grid(True)
    axes[0].legend()

    for column, label in (("error_east_m", "East"), ("error_north_m", "North"), ("error_up_m", "Up")):
        axes[1].plot(elapsed, synced[column], label=label)
    axes[1].set(xlabel="Elapsed time [s]", ylabel="FGO - GT error [m]", title="Position error")
    axes[1].grid(True)
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_rpy(synced: pd.DataFrame, output_path: Path) -> None:
    """Plot FGO RPY and position-derived GT course in degrees."""
    elapsed = (synced["timestamp_s"] - synced["timestamp_s"].iloc[0]).to_numpy()
    yaw_deg = np.rad2deg(np.unwrap(synced["yaw"].to_numpy()))
    velocity_east = np.gradient(synced["gt_east_m"].to_numpy(), elapsed)
    velocity_north = np.gradient(synced["gt_north_m"].to_numpy(), elapsed)
    speed = np.hypot(velocity_east, velocity_north)
    course_deg = np.rad2deg(np.unwrap(np.arctan2(velocity_north, velocity_east)))
    course_deg += 360.0 * np.round((yaw_deg[0] - course_deg[0]) / 360.0)
    course_deg[speed < 0.2] = np.nan

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(elapsed, np.rad2deg(synced["roll"]), color="green", label="FGO roll")
    axes[1].plot(elapsed, np.rad2deg(synced["pitch"]), color="red", label="FGO pitch")
    axes[2].plot(elapsed, yaw_deg, color="blue", label="FGO yaw")
    axes[2].plot(elapsed, course_deg, "k--", linewidth=1.2, label="GT course from position")
    axes[0].set_ylabel("Roll [deg]")
    axes[1].set_ylabel("Pitch [deg]")
    axes[2].set_ylabel("Yaw/course [deg]")
    axes[2].set_xlabel("Elapsed time [s]")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    lever_arm_body = np.asarray(config["gnss"]["lever_arm_body"], dtype=float)
    df_fgo, df_gt = load_trajectories(FGO_FILE, GT_FILE)
    synced = synchronize_trajectories(df_fgo, df_gt, lever_arm_body)
    synced = align_antenna_trajectory_svd(synced)
    print(f"FGO shape: {df_fgo.shape}")
    print(f"GT shape:  {df_gt.shape}")
    print_metrics(synced)

    error_csv = OUTPUT_DIR / "trajectory_error.csv"
    plot_file = OUTPUT_DIR / "trajectory_comparison.png"
    rpy_plot_file = OUTPUT_DIR / "rpy_changes.png"
    synced.to_csv(error_csv, index=False)
    save_plot(synced, plot_file)
    plot_rpy(synced, rpy_plot_file)
    print(f"Error CSV: {error_csv}")
    print(f"Plot:      {plot_file}")
    print(f"RPY plot:  {rpy_plot_file}")


if __name__ == "__main__":
    main()
