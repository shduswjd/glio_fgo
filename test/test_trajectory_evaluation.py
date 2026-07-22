import numpy as np
import pandas as pd

from evaluation.trajectory import align_antenna_trajectory_svd


def test_svd_alignment_recovers_rigid_transform_without_scale_change():
    estimated = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.2],
        [1.0, 2.0, 0.5],
        [-0.5, 1.0, 1.0],
    ])
    angle = np.deg2rad(35.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    ground_truth = (rotation @ estimated.T).T + np.array([4.0, -3.0, 1.5])
    synced = pd.DataFrame({
        "fgo_antenna_east_m": estimated[:, 0],
        "fgo_antenna_north_m": estimated[:, 1],
        "fgo_antenna_up_m": estimated[:, 2],
        "gt_east_m": ground_truth[:, 0],
        "gt_north_m": ground_truth[:, 1],
        "gt_up_m": ground_truth[:, 2],
    })

    aligned = align_antenna_trajectory_svd(synced)

    np.testing.assert_allclose(aligned["ate_m"], 0.0, atol=1e-12)
