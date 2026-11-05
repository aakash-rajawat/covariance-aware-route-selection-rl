"""Small, explicit KF and unicycle EKF. States use SI units and radians."""
import numpy as np


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def joseph_update(mean, cov, measurement, H, R, angle_index=None):
    """Numerically stable Gaussian linear measurement update (no explicit inverse)."""
    innovation = measurement - H @ mean
    if angle_index is not None:
        innovation[angle_index] = wrap(innovation[angle_index])
    S = H @ cov @ H.T + R
    K = np.linalg.solve(S, H @ cov).T
    result = mean + K @ innovation
    if angle_index is not None:
        result[angle_index] = wrap(result[angle_index])
    I_KH = np.eye(len(mean)) - K @ H
    new_cov = I_KH @ cov @ I_KH.T + K @ R @ K.T
    return result, 0.5 * (new_cov + new_cov.T)


class ConstantVelocityKF:
    """One obstacle: [x, y, vx, vy], point observation with covariance R."""
    def __init__(self, position, measurement_std=0.15):
        self.mean = np.array([*position, 0.0, 0.0], dtype=float)
        self.cov = np.diag([measurement_std**2] * 2 + [0.6**2] * 2)
        self.missed = 0

    def predict(self, dt, acceleration_std=0.35):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt**2/2, 0], [0, dt**2/2], [dt, 0], [0, dt]])
        self.mean = F @ self.mean
        self.cov = F @ self.cov @ F.T + acceleration_std**2 * (G @ G.T)
        self.cov = (self.cov + self.cov.T)/2
        self.missed += 1

    def update(self, position, measurement_std):
        H = np.eye(4)[:2]
        R = np.eye(2) * measurement_std**2
        self.mean, self.cov = joseph_update(self.mean, self.cov, np.asarray(position), H, R)
        self.missed = 0


class UnicycleEKF:
    """Ego estimate [x,y,yaw]; commanded v and w, intermittent pose fixes."""
    def __init__(self, initial):
        self.mean = np.array(initial, dtype=float)
        self.cov = np.diag([0.08**2, 0.08**2, 0.05**2])

    def predict(self, dt, v, w):
        theta = self.mean[2]
        F = np.eye(3)
        F[0, 2] = -v * dt * np.sin(theta)
        F[1, 2] = v * dt * np.cos(theta)
        self.mean += [v * dt * np.cos(theta), v * dt * np.sin(theta), w * dt]
        self.mean[2] = wrap(self.mean[2])
        self.cov = F @ self.cov @ F.T + np.diag([0.012**2, 0.012**2, 0.01**2])
        self.cov = (self.cov + self.cov.T)/2

    def update(self, measured_pose, position_std=0.08, heading_std=0.05):
        R = np.diag([position_std**2] * 2 + [heading_std**2])
        self.mean, self.cov = joseph_update(self.mean, self.cov, np.array(measured_pose), np.eye(3), R, angle_index=2)
