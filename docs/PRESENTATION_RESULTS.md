# M7d results: interpretation and boundaries

The summary table and figures in `README.md` are generated from:

- `results/seed0/m7d_eval_seed0.json` (Point, Belief, covariance ablations)
- `results/seed1/m7d_eval_seed1.json` and `results/seed2/m7d_eval_seed2.json` (independently trained policies)
- `results/fixed_baselines/m7d_fixed_baselines.json` (fixed route and relocalization strategies)
- `src/m3b_gate.json` and `src/mini_scene_m3b.xml` (maze geometry)

There are three independent training seeds; the same 48 fresh evaluation scenarios
were used for all three seeds, so they are not 144 independent tests.

The main learned result is a covariance-conditioned route rule: narrow at low
covariance, wide at high covariance. The stronger 'relocalize, then wide'
fixed strategy outperforms Belief in both success and mean reward.

The Point/Belief comparison intentionally masks high-level range inputs for
both agents. This isolates covariance in a restricted observation setting.
It cannot establish that covariance improves a controller with unrestricted
sensor inputs. The high-level DQN does not learn wheel commands.

The diagram shows *reference waypoints*, not individual robot trajectories.
The evaluation results come from physical MuJoCo simulation of the executed
route options, not path-length proxies. No confidence intervals or claims of
statistical significance are made from this small benchmark.
