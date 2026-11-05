# Covariance-Aware Route Selection with Deep Reinforcement Learning

A MuJoCo proof of concept comparing Point Double DQN and Belief Double DQN
for uncertainty-dependent navigation decisions.

Both policies select between:
1. A short, narrow route
2. A longer, wider route
3. A paid localization correction followed by route selection

The same predefined EKF-based waypoint controller executes the chosen route.
The neural networks learn high-level decisions, not wheel commands.

## Experimental Results

Three independently trained policy pairs were evaluated on
48 shared, previously unseen simulation scenarios.

| Strategy | Success | Mean reward |
|---|---:|---:|
| Point DQN | 29/48 | 2.338 |
| Belief DQN | 39/48 | 9.815 |
| Always wide | 39/48 | 9.356 |
| Relocalize then wide | 45/48 | 12.805 |

All three Belief DQNs learned to select the short route under low covariance
and the wide route under high covariance.

Point DQN consistently selected the short route.

However, the relocalize-then-wide reference strategy outperformed
both learned agents.

## Repository Structure

- `src/` — MuJoCo, EKF, controllers and Double DQN implementation
- `checkpoints/` — six trained PyTorch models
- `results/` — training and evaluation results
- `scripts/` — reproducibility commands
- `docs/` — methodology, limitations and provenance

## Reproduction

Run:

    bash scripts/re_evaluate.sh smoke

For complete evaluation without retraining:

    bash scripts/re_evaluate.sh full

The scripts use the existing Python environment from
`~/belief-aware-robot-navigation/.venv/`.

## Limitations

This is a high-level covariance-conditioned route-selection demonstration.

It uses one maze, two prescribed uncertainty regimes, fixed low-level
controllers and deliberately masked high-level range measurements.

No cross-maze generalization, Gazebo validation or wheel-level RL
performance is claimed.

The covariance comes from the simulated robot EKF, not an analytical
optical-tracking covariance model.
