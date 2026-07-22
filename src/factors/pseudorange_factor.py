import numpy as np
import gtsam

from sensors.gnss_raw import ObservationEpoch, BroadcastEphemeris

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ])

def create_pseudorange_factor(
        pose_key: int, 
        clock_bias_key: int,
        satellite_position_ecef: np.ndarray,
        measured_pseudorange: float,
        noise_model: gtsam.noiseModel.Base,
        lever_arm: np.ndarray | None = None,
) -> gtsam.CustomFactor:
    """Create a pseudorange factor connected to a receiver pose and clock bias.

    Positions and the receiver clock bias are expressed in metres.  The pose is
    expected to place the body frame in ECEF, and ``lever_arm`` is the antenna
    position expressed in the body frame.
    """

    satellite_position = np.asarray(
        satellite_position_ecef, dtype=float
    ).reshape(3)
    pseudorange = float(measured_pseudorange)

    if lever_arm is None: 
        lever = np.zeros(3)
    else:
        lever = np.asarray(lever_arm, dtype=float).reshape(3)

    if not np.all(np.isfinite(satellite_position)):
        raise ValueError("satellite_position_ecef must contain finite values")
    if not np.isfinite(pseudorange) or pseudorange <= 0.0:
        raise ValueError("measured_pseudorange must be finite and positive")
    if not np.all(np.isfinite(lever)):
        raise ValueError("lever_arm must contain finite values")

    def error_func(
            this: gtsam.CustomFactor,
            values: gtsam.Values,
            jacobians: list[np.ndarray],
    ):

        pose = values.atPose3(pose_key)
        clock_bias = float(values.atDouble(clock_bias_key))

        R_ecef_body = pose.rotation().matrix()
        t_ecef = np.asarray(pose.translation(), dtype=float).reshape(3)
        antenna_position_ecef = t_ecef + R_ecef_body @ lever

        delta = satellite_position - antenna_position_ecef
        geometric_range = np.linalg.norm(delta)
        if geometric_range < 1e-6:
            raise ValueError("satellite and receiver positions must be distinct")

        # LOS points from the receiver antenna towards the satellite.
        los = delta / geometric_range
        predicted = geometric_range + clock_bias
        residual = np.array([predicted - pseudorange], dtype=float)

        if jacobians is not None:
            # Pose3 retract uses body-frame increments.  The antenna-position
            # Jacobian is [-R[lever]x, R], while d(range)/d(antenna)=-LOS.
            H_antenna_pose = np.hstack(
                [-R_ecef_body @ skew(lever), R_ecef_body]
            )
            jacobians[0] = -los.reshape(1, 3) @ H_antenna_pose
            jacobians[1] = np.array([[1.0]])

        return residual
    return gtsam.CustomFactor(
        noise_model, [pose_key, clock_bias_key], error_func
    )

def create_pseudorange_factor_from_epoch(
        epoch: ObservationEpoch,
        ephemerides: None,
        pose_key: int,
        clock_bias_key: int
) -> list[gtsam.NonlinearFactor]:
    pass
        
def create_pseudorange_noise():
    pass

 