"""Zenith wet-delay correction state and temporal process model."""

from __future__ import annotations

import gtsam
import numpy as np


def wet_delay_key(index: int) -> int:
    return gtsam.symbol("w", index)


def create_wet_delay_random_walk_factor(
    index: int, delta_time: float, sigma_m_sqrt_s: float,
) -> gtsam.BetweenFactorDouble:
    if delta_time <= 0.0:
        raise ValueError("delta_time must be positive")
    if sigma_m_sqrt_s <= 0.0:
        raise ValueError("wet-delay process sigma must be positive")
    return gtsam.BetweenFactorDouble(
        wet_delay_key(index), wet_delay_key(index + 1), 0.0,
        gtsam.noiseModel.Isotropic.Sigma(
            1, sigma_m_sqrt_s * np.sqrt(delta_time)
        ),
    )
