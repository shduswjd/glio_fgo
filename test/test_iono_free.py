import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensors.gnss_raw import RinexValue, SatelliteObservation


def test_gps_l1_l2_ionosphere_free_pseudorange():
    observation = SatelliteObservation(
        "G05", {"C1C": RinexValue(20_000_010.0), "C2W": RinexValue(20_000_016.469)}
    )
    # A +10 m L1 ionospheric delay corresponds to +10*(f1/f2)^2 on L2;
    # the first-order ionosphere-free combination recovers the geometric code.
    assert observation.ionosphere_free_pseudorange("1C", "2W") == pytest.approx(
        20_000_000.0, abs=2e-3
    )


def test_ionosphere_free_requires_both_codes():
    observation = SatelliteObservation("G05", {"C1C": RinexValue(20_000_010.0)})
    assert observation.ionosphere_free_pseudorange("1C", "2W") is None
