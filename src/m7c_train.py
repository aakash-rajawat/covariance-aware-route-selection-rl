"""M7c high-level Point/Belief matched RL on M7b conditional initial belief.

No expert route decisions or demonstrations. The fixed route controllers are
shared by both policies; route option selection and one paid pose fix are learned.
At the policy's initial decision, scan features are zeroed for BOTH agents,
so the intended comparison isolates covariance rather than global pose's
otherwise identifiable deviation from a known physical starting location.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

import m7b_initialization as m7b

core = m7b.core
RUN = Path(__file__).resolve().parent
ORIGINAL_MAKE_OBSERVATION = core.make_observation


def high_level_observation(estimated, covariance, truth, boxes, mode, can_fix):
    obs = ORIGINAL_MAKE_OBSERVATION(
        estimated, covariance, truth, boxes, mode, can_fix
    ).copy()
    # Range sensors are excluded from BOTH high-level route selectors.
    # Same MuJoCo robot, obstacles, contacts, estimation and option execution.
    obs[4:20] = 0.0
    return obs


core.make_observation = high_level_observation


def audit():
    sim = core.BotWSMuJoCoBackend(scene_path=core.SCENE)
    boxes = core.m2.scene_boxes(core.SCENE)
    examples = {}
    for regime in ('low', 'high'):
        for mode in ('point', 'belief'):
            estimator, obs = core.reset_case(
                sim, boxes, mode, regime, seed=970001
            )
            assert obs.shape == (27,)
            assert np.all(np.isfinite(obs))
            assert np.array_equal(obs[4:20], np.zeros(16))
            assert obs[-1] == 1.0
            if mode == 'point':
                assert np.array_equal(obs[20:26], np.zeros(6))
            else:
                assert np.linalg.norm(obs[20:26]) > 0
            examples[mode, regime] = dict(
                observation=obs.tolist(),
                initial_covariance_trace=float(np.trace(estimator.filter.cov)),
            )
    point_low = np.array(examples['point', 'low']['observation'])
    point_high = np.array(examples['point', 'high']['observation'])
    assert np.array_equal(point_low, point_high), (
        'Point still observes a difference between uncertainty regimes'
    )
    belief_low = np.array(examples['belief', 'low']['observation'])
    belief_high = np.array(examples['belief', 'high']['observation'])
    assert np.array_equal(belief_low[:20], belief_high[:20])
    assert not np.array_equal(belief_low[20:26], belief_high[20:26])
    assert examples['belief', 'high']['initial_covariance_trace'] > (
        examples['belief', 'low']['initial_covariance_trace']
    )
    report = dict(
        status='PASS',
        high_level_ranges_excluded_for_both_policies=True,
        original_M7b_initialization_preserved=True,
        point_initial_observation_identical_across_regimes=True,
        belief_initial_observation_differs_only_in_covariance=True,
        observation_dimension=27,
        first_action_options=['short', 'wide', 'paid_fix'],
        post_fix_action_options=['short', 'wide'],
        constraints=[
            'Only high-level route selection is learned; low-level controller uses EKF.',
            'Low/high uncertainty is directly encoded by covariance; this is a restricted observation comparison.',
            'MuJoCo physics and fixed M3b route geometry remain unchanged.',
            'No guarantee of trained Belief advantage; small expected information value.',
        ],
        examples={f'{k[0]}_{k[1]}': v for k, v in examples.items()},
    )
    (RUN / 'm7c_observation_audit.json').write_text(
        json.dumps(report, indent=2) + '\n'
    )
    print('M7c audit PASS: identical point observations; ranges masked for both.', flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-episodes', type=int, default=240)
    ap.add_argument('--eval-per-regime', type=int, default=24)
    ap.add_argument('--report-every', type=int, default=40)
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args()
    args.preflight_only = False  # Required by the reused M7 training entry point.
    core.check_inputs()
    audit()
    if args.audit_only:
        print('Audit-only mode: no training run', flush=True)
        return 0
    # Use M7b patched conditional reset_case. DO NOT run the unpatched M7 script.
    code = core.run(args)
    if code != 0:
        return code
    result_path = RUN / 'm7_results.json'
    result = json.loads(result_path.read_text())
    result['m7c_design'] = {
        'initialization': 'M7b conditional P: fixed estimated mean, sampled physical truth',
        'high_level_sensing': '4 goal-relative features, 16 zeros, 6 covariance-or-zeros, 1 can_fix bit',
        'range_sensing': 'Excluded only from high-level policy; MuJoCo physical geometry unchanged',
        'training_source': 'M7 masked Double DQN, both policies initialized from random weights',
        'covariance_interpretation': 'The experimental regimes differ in initial uncertainty and sensor noise. ' +
             'Point does not receive the regime; Belief receives P.',
        'causal_limitation': 'This tests information value with deliberately restricted high-level sensors, ' +
             'not the benefit of covariance over all possible observations.',
    }
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print('M7c COMPLETE. B-P return:', round(result['return_delta_belief_minus_point'], 4),
          '; B live-fixed:', round(result['return_delta_belief_live_minus_fixed'],4),flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
