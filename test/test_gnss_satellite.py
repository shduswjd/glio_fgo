import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensors.gnss_models import BeidouEphemeris, GalileoEphemeris, GpsEphemeris
from sensors.gnss_raw import BroadcastEphemeris
from sensors.gnss_satellite import (
    BEIDOU_GPS_WEEK_OFFSET, BEIDOU_TO_GPS_SECONDS, EphemerisStore,
    beidou_ephemeris_from_rinex, classify_beidou_orbit,
    galileo_ephemeris_from_rinex, propagate_galileo, propagate_gps,
    rotate_for_sagnac,
)


def sample_ephemeris() -> GpsEphemeris:
    return GpsEphemeris(
        satellite="G01", toc=2200 * 604800.0 + 100000.0,
        af0=2e-4, af1=-4e-12, af2=0.0, iode=1.0, crs=20.0,
        delta_n=4e-9, m0=0.5, cuc=1e-6, eccentricity=0.01,
        cus=2e-6, sqrt_a=5153.795, toe=100000.0, cic=1e-7,
        omega0=1.0, cis=-1e-7, i0=0.94, crc=200.0,
        argument_of_perigee=0.7, omega_dot=-8e-9, idot=1e-10,
        gps_week=2200, tgd=-2e-9,
    )


def test_broadcast_propagation_has_physical_position_and_velocity():
    state = propagate_gps(sample_ephemeris(), 2200 * 604800.0 + 100100.0)
    assert np.linalg.norm(state.position_ecef) == pytest.approx(26_560_000.0, rel=0.03)
    assert 1_000.0 < np.linalg.norm(state.velocity_ecef) < 5_000.0
    assert np.isfinite(state.clock_bias_s)


def test_ephemeris_store_selects_nearest_toe():
    ephemeris = sample_ephemeris()
    store = EphemerisStore([ephemeris])
    assert store.nearest("G01", ephemeris.toc + 10.0) is ephemeris
    with pytest.raises(KeyError):
        store.nearest("G02", ephemeris.toc)


def test_sagnac_rotation_preserves_norm():
    vector = np.array([20e6, 10e6, 5e6])
    assert np.linalg.norm(rotate_for_sagnac(vector, 0.075)) == pytest.approx(np.linalg.norm(vector))


def test_galileo_broadcast_propagation_has_physical_state():
    gps = sample_ephemeris()
    ephemeris = GalileoEphemeris(
        satellite="E11", toc=gps.toc, af0=gps.af0, af1=gps.af1,
        af2=gps.af2, iodnav=gps.iode, crs=gps.crs,
        delta_n=gps.delta_n, m0=gps.m0, cuc=gps.cuc,
        eccentricity=gps.eccentricity, cus=gps.cus, sqrt_a=5440.6,
        toe=gps.toe, cic=gps.cic, omega0=gps.omega0, cis=gps.cis,
        i0=0.99, crc=gps.crc, argument_of_perigee=gps.argument_of_perigee,
        omega_dot=gps.omega_dot, idot=gps.idot, gps_week=gps.gps_week,
        bgd_e5a_e1=-1.3e-8, bgd_e5b_e1=-1.4e-8,
    )
    state = propagate_galileo(ephemeris, ephemeris.toc + 100.0)
    assert np.linalg.norm(state.position_ecef) == pytest.approx(29_600_000.0, rel=0.03)
    assert 1_000.0 < np.linalg.norm(state.velocity_ecef) < 5_000.0
    assert state.satellite == "E11"


def test_galileo_rinex_bgd_fields_do_not_use_transmission_time():
    values = [0.0] * 24
    values[7] = 5440.6
    values[18] = 2129.0
    values[21] = -1.3e-8
    values[22] = -1.4e-8
    values[23] = 177634.0
    raw = BroadcastEphemeris(
        "E11", datetime(2020, 10, 27, tzinfo=timezone.utc),
        (0.0, 0.0, 0.0), tuple(values),
    )
    ephemeris = galileo_ephemeris_from_rinex(raw)
    assert ephemeris.bgd_e5a_e1 == pytest.approx(-1.3e-8)
    assert ephemeris.bgd_e5b_e1 == pytest.approx(-1.4e-8)


def sample_beidou(sqrt_a: float, inclination_deg: float) -> BeidouEphemeris:
    gps = sample_ephemeris()
    return BeidouEphemeris(
        satellite="C01", toc=gps.toc, af0=0.0, af1=0.0, af2=0.0,
        aode=1.0, crs=0.0, delta_n=0.0, m0=0.0, cuc=0.0,
        eccentricity=0.001, cus=0.0, sqrt_a=sqrt_a, toe=100000.0,
        cic=0.0, omega0=0.0, cis=0.0, i0=np.deg2rad(inclination_deg),
        crc=0.0, argument_of_perigee=0.0, omega_dot=0.0, idot=0.0,
        gps_week=2200, tgd_b1_b3=0.0, tgd_b2_b3=0.0,
    )


@pytest.mark.parametrize(("sqrt_a", "inclination", "expected"), [
    (5282.0, 55.0, "MEO"),
    (6493.0, 55.0, "IGSO"),
    (6493.0, 5.0, "GEO"),
])
def test_beidou_orbit_classification(sqrt_a, inclination, expected):
    assert classify_beidou_orbit(sample_beidou(sqrt_a, inclination)) == expected


def test_beidou_rinex_time_and_tgd_conversion():
    values = [0.0] * 26
    values[7] = 6493.0
    values[8] = 172800.0
    values[18] = 773.0
    values[22] = -9.2e-9
    values[23] = 2.3e-9
    raw = BroadcastEphemeris(
        "C13", datetime(2020, 10, 27, tzinfo=timezone.utc),
        (0.0, 0.0, 0.0), tuple(values),
    )
    ephemeris = beidou_ephemeris_from_rinex(raw)
    assert ephemeris.gps_week == 773 + BEIDOU_GPS_WEEK_OFFSET
    assert ephemeris.toe == 172800.0
    assert ephemeris.toc % 604800.0 == pytest.approx(172800.0 + BEIDOU_TO_GPS_SECONDS)
    assert ephemeris.tgd_b1_b3 == pytest.approx(-9.2e-9)
    assert ephemeris.tgd_b2_b3 == pytest.approx(2.3e-9)
