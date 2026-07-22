import numpy as np
import gtsam
import pymap3d as pm



from sensors.gnss import GnssMeasurement
# from gtsam.utils.numerical_derivative import numericalDerivative11

# receiver clock bias
def C(key: int) -> int:
    return gtsam.symbol("c", key)

# receiver clock drift
def D(key: int) -> int:
    return gtsam.symbol("d", key)

# receiver ecef position Point3
def P(key: int) -> int:
    return gtsam.symbol("p", key)

def llh_to_enu(
    latitude: float,
    longitude: float,
    altitude: float,
    reference_latitude: float,
    reference_longitude: float,
    reference_altitude: float,
):
    e, n, u = pm.geodetic2enu(
        latitude,
        longitude,
        altitude,
        reference_latitude,
        reference_longitude,
        reference_altitude,
    )
    return e, n, u # np.concatenate((e, n, u))

def create_gnss_noise(
    position_covariance: np.ndarray,
    minimum_sigma: float = 0.1,
) -> gtsam.noiseModel.Base:
    
    covariance = np.asarray(position_covariance, dtype=float)

    if covariance.shape != (3, 3):
        raise ValueError(
            f"position_covariance must have shape (3, 3), but got {covariance.shape}"
        )
    
    if not np.all(np.isfinite(covariance)):
        raise ValueError("position_covariance must contain finite values")
    
    if minimum_sigma <= 0.0:
        raise ValueError("minimum_sigma must be positive")
    
    # 수치 오차 때문에 비대칭 일 수 있으므로 대칭화
    covariance = 0.5 * (covariance + covariance.T)

    # 지나치게 작은 고윳값과 음수 고윳값을 제한 
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, minimum_sigma**2)

    safe_covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return gtsam.noiseModel.Gaussian.Covariance(safe_covariance)

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ])

def create_gnss_position_factor(
    pose_key: int,
    position_enu: gtsam.Point3,
    noise_model: gtsam.noiseModel.Base,
    lever_arm: np.ndarray | None = None
) -> gtsam.CustomFactor: # gtsam.NonlinearFactor:
    """state의 pose와 GNSS 위치 측정을 연결."""

    # measurements
    z = np.asarray(position_enu, dtype=float).reshape(3)
    if lever_arm is None:
        lever = np.zeros(3)
    else:
        lever = np.asarray(lever_arm, dtype=float).reshape(3)

    def error_func(
            this: gtsam.CustomFactor,
            values: gtsam.Values,
            jacobians: list[np.ndarray],
    ) -> np.ndarray:
        pose = values.atPose3(pose_key)
        # print("[updating] estimated pose: ", pose.translation())
        # print("[updating] estimated rotation (rpy):", np.rad2deg(pose.rotation().rpy()))
        predicted = np.asarray(pose.transformFrom(lever)).reshape(3)

        residual = predicted - z
        if jacobians is not None:
            # H_pose = np.zeros((3, 6), dtype=float) # pos (3) + 최적화 변수 Pose3(6)
            # pose.traformFrom(lever, H_pose) # lever local frame -> world frame && lever에 대한 jacobian 저장
            # jacobians[0] = H_pose
            R = pose.rotation().matrix()

            H_pose = np.zeros((3, 6), dtype=float)

            # d(R*lever + T) / d(R)
            H_pose[:, 0:3] = -R @ skew(lever)
            
            # d(R * lever + T) / d(T) 
            H_pose[:, 3:6] = R
            
            jacobians[0] = H_pose
        return residual
    
    return gtsam.CustomFactor(
        noise_model, [pose_key], error_func
    )
        


def create_factor_from_measurement(
        pose_key:int,
        measurement: GnssMeasurement,
        reference_llh: np.ndarray,
        lever_arm_body: np.ndarray | None = None,
        minimum_sigma: float = 0.1,
) -> gtsam.NonlinearFactor:
    """bag에서 읽은 GnssMeasurement를 바로 factor로 변환."""
    
    lat, lon, alt = measurement.latitude, measurement.longitude, measurement.altitude
    e, n, u = llh_to_enu(lat, lon, alt, *reference_llh)
    meas_enu = np.array([e, n, u])
    noise_model = create_gnss_noise(
        position_covariance=measurement.position_covariance,
        minimum_sigma=minimum_sigma,
    )
    return create_gnss_position_factor(pose_key, meas_enu, noise_model, lever_arm_body)


def predicted_position(
        pose: gtsam.Pose3,
        lever: np.ndarray,
)-> np.ndarray:
    return np.asarray(
        pose.transformFrom(lever),
        dtype = float
    ).reshape(3)
