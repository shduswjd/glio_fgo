# Tightly-coupled IMU/GNSS FGO structure

## Data representation

Use a `dataclass` when fields have a fixed meaning, unit, and coordinate frame.
Use a `dict` only as an index.  For example, a pseudorange should not be
`{"pr": ...}`; it belongs in `CorrectedGnssMeasurement`.  Looking up all GPS
ephemerides for `G05`, however, is a good use of `dict[str, list[GpsEphemeris]]`.

The processing direction is:

```text
RINEX OBS ──> ObservationEpoch ─┐
                               ├─> prepare_epoch ─> raw GNSS factors ─┐
RINEX NAV ──> GpsEphemeris ────┘                                     ├─> FGO
IMU ─────────> preintegration factors ────────────────────────────────┘
```

Modules have one responsibility:

- `sensors/gnss_raw.py`: RINEX syntax only
- `sensors/gnss_models.py`: typed interfaces and units
- `sensors/gnss_satellite.py`: ephemeris selection, GPS orbit/clock, Sagnac
- `sensors/gnss_corrections.py`: azimuth/elevation, Klobuchar, Saastamoinen
- `sensors/gnss_preprocessor.py`: builds observations for factors
- `factors/gnss_raw_factors.py`: residuals and Jacobians only

## Minimal use

```python
from sensors.gnss_preprocessor import prepare_epoch
from sensors.gnss_raw import read_rinex_navigation, read_rinex_observations
from sensors.gnss_satellite import EphemerisStore

navigation_records = read_rinex_navigation(nav_path)
ephemerides = EphemerisStore.from_rinex(navigation_records)

for epoch in read_rinex_observations(obs_path):
    measurements = prepare_epoch(
        epoch=epoch,
        receiver_position_ecef=current_position_estimate,
        ephemerides=ephemerides,
        ionosphere=klobuchar_parameters,  # None means no ionosphere correction
    )
    for measurement in measurements:
        graph.add(create_raw_pseudorange_factor(
            X(k), C(k), measurement, pseudorange_noise, lever_arm_body
        ))
        if measurement.range_rate_mps is not None:
            graph.add(create_doppler_factor(
                X(k), V(k), D(k), measurement, doppler_noise, lever_arm_body
            ))
```

`C(k)` is the GPS-referenced receiver clock bias in metres and `D(k)` is clock
drift in m/s. `E(k)` and `Q(k)` are the Galileo-minus-GPS and BeiDou-minus-GPS
receiver inter-system code biases.
Add a between-factor clock process model between epochs.  The current
preprocessor supports GPS, Galileo, and BeiDou broadcast ephemerides. BeiDou
MEO/IGSO use the common Kepler propagation and GEO uses its ICD-specific frame
rotation. GLONASS still needs numerical state-vector propagation and FDMA
channel frequencies.

Atmosphere and elevation depend on receiver position. For best accuracy,
rebuild these prepared measurements after a coarse first optimization (or move
the calculations into a factor). A practical first implementation uses one
outer relinearization pass.

## Running `tc.yaml`

From the repository root, activate the included environment and run:

```bash
source .venv/bin/activate
python agr_fgo/src/run_tc_gnss.py --epochs 5
```

The default config is `agr_fgo/config/tc.yaml`; it is resolved from the script
location, so the script can also be invoked from another working directory.
To select another config explicitly:

```bash
PYTHONPATH=agr_fgo/src .venv/bin/python agr_fgo/src/run_tc_gnss.py \
  --config agr_fgo/config/tc.yaml --epochs 30
```

The command loads all navigation records, uses the first observation epoch for
a GPS single-point-positioning initialization, applies the configured elevation
mask, and prints the number of factor-ready code and Doppler measurements.
Set `data.duration: 60.0` to limit input to the first minute. Set
`gnss_raw.initial_position_ecef` to a known `[x, y, z]` value to bypass SPP.

This command runs and validates the raw-GNSS front end. It does not yet launch
the combined IMU optimizer; the returned `CorrectedGnssMeasurement` objects are
the inputs to `create_raw_pseudorange_factor` and `create_doppler_factor` shown
above.

## Running the tightly-coupled graph

```bash
source .venv/bin/activate
python agr_fgo/src/tightly_coupled_fgo.py
```

Start with `data.duration: 10.0` so the first run finishes quickly.
Increase the duration gradually after checking the trajectory. The current
prototype deliberately rejects a full `null` result if graph error does not
decrease; production-scale full data needs fixed-lag marginalization rather
than one growing batch. Output is written to
`agr_fgo/results/tc_trajectory.csv`.

Each one-second GNSS epoch creates these variables:

```text
X(k): body pose in local ENU
V(k): body velocity in local ENU [m/s]
B(k): accelerometer and gyroscope biases
C(k): GPS-referenced receiver clock bias [m]
D(k): receiver clock drift [m/s]
E(k): Galileo-minus-GPS inter-system code bias [m]
Q(k): BeiDou-minus-GPS inter-system code bias [m]
```

Adjacent epochs are connected by an IMU preintegration factor, an IMU-bias
random-walk factor, and a constant-drift receiver-clock factor. Every usable
satellite independently contributes one pseudorange factor and one Doppler
factor. Huber loss is applied to both GNSS measurement types.

The navigation state stays in local ENU, where the current GTSAM preintegration
model has a fixed up-axis gravity vector. GNSS factors convert antenna position
and velocity to ECEF internally. Do not change the navigation pose directly to
ECEF while using `PreintegrationParams.MakeSharedU()`; a proper ECEF mechanization
would additionally require position-dependent gravity, Earth rotation, and
Coriolis terms.

The TXT column layout in `tc.yaml` matches `Tactical_imu_data.txt`:

```text
time, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
```

Noise values currently provide runnable defaults, not a sensor calibration.
Replace them with datasheet noise density and Allan-variance bias random walk.
The current Tactical-IMU configuration uses the published layout estimate
`[-0.04, -0.33, 0.16]` m for sensor 5 to sensor 6. Confirm the signs against
the raw IMU axis convention and replace it with surveyed calibration when
available. Initial yaw is weakly constrained because this
TXT file has no absolute attitude; add a heading prior, dual-antenna heading, or
motion-based initialization for production use.
