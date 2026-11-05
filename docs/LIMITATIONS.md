# Limitations

- One maze and two prescribed localization uncertainty regimes.
- High-level range measurements deliberately excluded from both policies.
- Fixed low-level waypoint controller.
- EKF covariance calibration is imperfect.
- Three independent training seeds but the same 48 evaluation scenarios.
- Relocalize-then-wide outperforms both trained DQN policies.
- No statistical significance or general navigation optimality claimed.
- No cross-maze or Gazebo validation.
