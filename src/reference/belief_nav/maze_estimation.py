"""Wheel-odometry EKF integration for the converted MuJoCo robot.

Milestone 7C.1.

Prediction:
    Noisy wheel-joint angle increments.

Correction:
    Intermittent simulated absolute-pose measurements.

MuJoCo ground truth is used only for synthetic measurements
and initial-prior simulation.

This module does not modify the original UnicycleEKF.
"""

import math

import numpy as np

from belief_nav.filter import UnicycleEKF, wrap


class MuJoCoPoseEstimator:

    def __init__(
        self,
        backend,
        encoder_delta_std=0.002,
        pose_position_std=0.075,
        pose_heading_std=0.04,
        pose_fix_every=5,
        pose_dropout=0.30,
        initial_position_std=0.08,
        initial_heading_std=0.05,
    ):

        self.backend = backend

        self.encoder_delta_std = float(
            encoder_delta_std
        )

        self.pose_position_std = float(
            pose_position_std
        )

        self.pose_heading_std = float(
            pose_heading_std
        )

        self.pose_fix_every = int(
            pose_fix_every
        )

        self.pose_dropout = float(
            pose_dropout
        )

        self.initial_position_std = float(
            initial_position_std
        )

        self.initial_heading_std = float(
            initial_heading_std
        )

        if (
            self.encoder_delta_std < 0
            or self.pose_position_std <= 0
            or self.pose_heading_std <= 0
            or self.pose_fix_every < 1
            or not 0 <= self.pose_dropout <= 1
            or self.initial_position_std < 0
            or self.initial_heading_std < 0
        ):
            raise ValueError(
                "Invalid estimator noise or timing parameters."
            )

        model = backend.model

        self.right_qpos = int(
            model.jnt_qposadr[
                backend.right_joint_id
            ]
        )

        self.left_qpos = int(
            model.jnt_qposadr[
                backend.left_joint_id
            ]
        )

        self.filter = None

        self.last_angles = None
        self.last_time = None

        self.steps = 0
        self.fix_count = 0
        self.dropout_count = 0

    def _wheel_angles(self):

        return np.array(
            [
                self.backend.data.qpos[self.right_qpos],
                self.backend.data.qpos[self.left_qpos],
            ],
            dtype=np.float64,
        )

    def reset(self, seed=None):
        """Initialize a noisy pose prior and independent sensor RNGs."""

        seed_sequence = np.random.SeedSequence(seed)

        children = seed_sequence.spawn(4)

        self.initial_rng = np.random.default_rng(
            children[0]
        )

        self.encoder_rng = np.random.default_rng(
            children[1]
        )

        self.fix_rng = np.random.default_rng(
            children[2]
        )

        self.dropout_rng = np.random.default_rng(
            children[3]
        )

        # Synthetic initial localization prior.
        true_pose = self.backend.pose()

        initial_noise = self.initial_rng.normal(
            0.0,
            [
                self.initial_position_std,
                self.initial_position_std,
                self.initial_heading_std,
            ],
        )

        initial_estimate = (
            true_pose + initial_noise
        )

        initial_estimate[2] = wrap(
            initial_estimate[2]
        )

        self.filter = UnicycleEKF(
            initial_estimate
        )

        # Match the prior covariance to the declared
        # initialization uncertainty.
        self.filter.cov = np.diag(
            [
                self.initial_position_std**2,
                self.initial_position_std**2,
                self.initial_heading_std**2,
            ]
        )

        self.last_angles = self._wheel_angles()

        self.last_time = (
            self.backend.simulation_time
        )

        self.steps = 0
        self.fix_count = 0
        self.dropout_count = 0

        return (
            self.filter.mean.copy(),
            self.filter.cov.copy(),
        )

    def update(self):
        """Process one completed MuJoCo control interval."""

        if self.filter is None:
            raise RuntimeError(
                "Call estimator.reset() first."
            )

        current_time = (
            self.backend.simulation_time
        )

        dt = current_time - self.last_time

        if not math.isfinite(dt) or dt <= 0:
            raise RuntimeError(
                "Simulation time has not advanced."
            )

        current_angles = self._wheel_angles()

        delta_angles = (
            current_angles - self.last_angles
        )

        # Simulated incremental wheel-encoder noise.
        measured_delta = (
            delta_angles
            + self.encoder_rng.normal(
                0.0,
                self.encoder_delta_std,
                size=2,
            )
        )

        delta_right = float(
            measured_delta[0]
        )

        delta_left = float(
            measured_delta[1]
        )

        radius = (
            self.backend.WHEEL_RADIUS
        )

        separation = (
            self.backend.WHEEL_SEPARATION
        )

        delta_distance = (
            0.5
            * radius
            * (delta_right + delta_left)
        )

        delta_heading = (
            radius
            * (delta_right - delta_left)
            / separation
        )

        measured_v = (
            delta_distance / dt
        )

        measured_omega = (
            delta_heading / dt
        )

        # EKF prediction uses wheel odometry,
        # not the requested navigation command.
        self.filter.predict(
            dt,
            measured_v,
            measured_omega,
        )

        prior_covariance = (
            self.filter.cov.copy()
        )

        self.steps += 1

        fix_attempted = (
            self.steps % self.pose_fix_every == 0
        )

        fix_received = False
        measurement = None

        if fix_attempted:

            dropped = (
                self.dropout_rng.random()
                < self.pose_dropout
            )

            if dropped:

                self.dropout_count += 1

            else:

                # Synthetic absolute-pose measurement.
                true_pose = self.backend.pose()

                noise = self.fix_rng.normal(
                    0.0,
                    [
                        self.pose_position_std,
                        self.pose_position_std,
                        self.pose_heading_std,
                    ],
                )

                measurement = (
                    true_pose + noise
                )

                measurement[2] = wrap(
                    measurement[2]
                )

                self.filter.update(
                    measurement,
                    position_std=self.pose_position_std,
                    heading_std=self.pose_heading_std,
                )

                self.fix_count += 1
                fix_received = True

        self.last_angles = current_angles
        self.last_time = current_time

        if not (
            np.all(np.isfinite(self.filter.mean))
            and np.all(np.isfinite(self.filter.cov))
        ):
            raise FloatingPointError(
                "Non-finite EKF state."
            )

        covariance = self.filter.cov.copy()

        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-10,
        ):
            raise RuntimeError(
                "EKF covariance is not symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if np.min(eigenvalues) < -1e-10:
            raise RuntimeError(
                "EKF covariance is not positive semidefinite."
            )

        return {
            "mean": self.filter.mean.copy(),
            "covariance": covariance,
            "dt": float(dt),
            "odometry_v": measured_v,
            "odometry_omega": measured_omega,
            "fix_attempted": fix_attempted,
            "fix_received": fix_received,
            "measurement": (
                None if measurement is None
                else measurement.copy()
            ),
            "prior_covariance_trace": float(
                np.trace(prior_covariance)
            ),
            "posterior_covariance_trace": float(
                np.trace(covariance)
            ),
        }

    @property
    def mean(self):

        return self.filter.mean.copy()

    @property
    def covariance(self):

        return self.filter.cov.copy()
