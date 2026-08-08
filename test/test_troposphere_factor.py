import sys
from pathlib import Path

import gtsam
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factors.troposphere import (
    create_wet_delay_random_walk_factor, wet_delay_key,
)


def test_wet_delay_random_walk_has_zero_error_when_constant():
    factor = create_wet_delay_random_walk_factor(0, 2.0, 0.005)
    values = gtsam.Values()
    values.insert(wet_delay_key(0), 0.12)
    values.insert(wet_delay_key(1), 0.12)
    assert factor.unwhitenedError(values) == pytest.approx(np.zeros(1))


def test_wet_delay_random_walk_rejects_invalid_process_sigma():
    with pytest.raises(ValueError):
        create_wet_delay_random_walk_factor(0, 1.0, 0.0)
