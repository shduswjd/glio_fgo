import sys
from pathlib import Path

import gtsam
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factors.clock import (
    beidou_isb_key, clock_bias_key, clock_drift_key,
    create_beidou_isb_random_walk_factor, create_clock_dynamics_factor,
    create_isb_random_walk_factor, galileo_isb_key,
)


def test_clock_dynamics_has_zero_error_for_constant_drift():
    factor = create_clock_dynamics_factor(0, 2.0, 3.0, 0.3)
    values = gtsam.Values()
    values.insert(clock_bias_key(0), 100.0)
    values.insert(clock_drift_key(0), 4.0)
    values.insert(clock_bias_key(1), 108.0)
    values.insert(clock_drift_key(1), 4.0)
    assert factor.unwhitenedError(values) == pytest.approx(np.zeros(2))


def test_galileo_isb_random_walk_connects_adjacent_epochs():
    factor = create_isb_random_walk_factor(0, 2.0, 0.3)
    values = gtsam.Values()
    values.insert(galileo_isb_key(0), 12.5)
    values.insert(galileo_isb_key(1), 12.5)
    assert factor.unwhitenedError(values) == pytest.approx(np.zeros(1))


def test_beidou_isb_random_walk_connects_adjacent_epochs():
    factor = create_beidou_isb_random_walk_factor(0, 2.0, 0.3)
    values = gtsam.Values()
    values.insert(beidou_isb_key(0), 30.0)
    values.insert(beidou_isb_key(1), 30.0)
    assert factor.unwhitenedError(values) == pytest.approx(np.zeros(1))
