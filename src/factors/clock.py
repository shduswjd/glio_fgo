"""Receiver clock states and a constant-drift process factor."""

from __future__ import annotations

import gtsam
import numpy as np


def clock_bias_key(index: int) -> int:
    return gtsam.symbol("c", index)


def clock_drift_key(index: int) -> int:
    return gtsam.symbol("d", index)


def galileo_isb_key(index: int) -> int:
    """Galileo-minus-GPS receiver inter-system code bias [m]."""
    return gtsam.symbol("e", index)


def beidou_isb_key(index: int) -> int:
    """BeiDou-minus-GPS receiver inter-system code bias [m]."""
    return gtsam.symbol("q", index)


def create_isb_random_walk_factor(
    index: int, delta_time: float, sigma_m_sqrt_s: float,
) -> gtsam.BetweenFactorDouble:
    if delta_time <= 0.0:
        raise ValueError("delta_time must be positive")
    return gtsam.BetweenFactorDouble(
        galileo_isb_key(index), galileo_isb_key(index + 1), 0.0,
        gtsam.noiseModel.Isotropic.Sigma(
            1, sigma_m_sqrt_s * np.sqrt(delta_time)
        ),
    )


def create_beidou_isb_random_walk_factor(
    index: int, delta_time: float, sigma_m_sqrt_s: float,
) -> gtsam.BetweenFactorDouble:
    if delta_time <= 0.0:
        raise ValueError("delta_time must be positive")
    return gtsam.BetweenFactorDouble(
        beidou_isb_key(index), beidou_isb_key(index + 1), 0.0,
        gtsam.noiseModel.Isotropic.Sigma(
            1, sigma_m_sqrt_s * np.sqrt(delta_time)
        ),
    )


def create_clock_dynamics_factor(
    index: int,
    delta_time: float,
    bias_process_sigma_m: float,
    drift_process_sigma_mps: float,
) -> gtsam.CustomFactor:
    """Model ``b[j] = b[i] + d[i] dt`` and ``d[j] = d[i]``."""
    if delta_time <= 0.0:
        raise ValueError("delta_time must be positive")
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        bias_process_sigma_m * np.sqrt(delta_time),
        drift_process_sigma_mps * np.sqrt(delta_time),
    ]))
    keys = [
        clock_bias_key(index), clock_drift_key(index),
        clock_bias_key(index + 1), clock_drift_key(index + 1),
    ]

    def error(_factor, values: gtsam.Values, jacobians) -> np.ndarray:
        bias_i = values.atDouble(keys[0])
        drift_i = values.atDouble(keys[1])
        bias_j = values.atDouble(keys[2])
        drift_j = values.atDouble(keys[3])
        if jacobians is not None:
            jacobians[0] = np.array([[-1.0], [0.0]])
            jacobians[1] = np.array([[-delta_time], [-1.0]])
            jacobians[2] = np.array([[1.0], [0.0]])
            jacobians[3] = np.array([[0.0], [1.0]])
        return np.array([
            bias_j - bias_i - drift_i * delta_time,
            drift_j - drift_i,
        ])

    return gtsam.CustomFactor(noise, keys, error)
