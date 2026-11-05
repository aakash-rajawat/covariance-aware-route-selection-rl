"""M7b: physically consistent conditional initial belief, plus correctness audit.

Run from an isolated copy of the M7 preflight archive. M7 learning algorithm,
maze, reference route controller, costs, and network architecture are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import m7_hierarchical_rl as core

RUN = Path(__file__).resolve().parent
ANCHOR = core.START.copy()
SAFE_START = dict(xmin=0.26, xmax=1.03, ymin=0.27, ymax=2.93)


def draw_start(regime: str, seed: int):
    """Draw truth conditional on the common estimated pose and its covariance.

    Same underlying random normals used for the two regimes under a given seed.
    Reject physically infeasible starts (report rejection rate). This truncation
    slightly changes the theoretical Gaussian distribution and is NOT concealed.
    """
    p = core.m2.REGIMES[regime]
    sig = np.array([p['initial_position_std'], p['initial_position_std'],
                    p['initial_heading_std']], dtype=np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 473891]))
    for attempts in range(1, 101):
        z = rng.standard_normal(3)
        true = ANCHOR + sig * z
        if (SAFE_START['xmin'] <= true[0] <= SAFE_START['xmax'] and
            SAFE_START['ymin'] <= true[1] <= SAFE_START['ymax'] and
            abs(core.m2.wrap(true[2])) <= math.pi):
            true[2] = core.m2.wrap(true[2])
            return true, sig, attempts
    raise RuntimeError(f'Cannot sample a safe initial truth after 100 draws: {regime}, {seed}')


def reset_case(sim, boxes, mode, regime, seed):
    true, sig, attempts = draw_start(regime, seed)
    sim.reset(true)
    actual = np.asarray(sim.pose(), dtype=float)
    if np.linalg.norm(actual[:2]-true[:2]) > 0.025:
        raise AssertionError('MuJoCo reset settled far from requested start')
    if sim.wall_contact():
        raise AssertionError('Initial contact after reset; unsafe initial draw')
    estimator = core.MuJoCoPoseEstimator(sim, pose_fix_every=5,
                                        **core.m2.REGIMES[regime])
    estimator.reset(seed=seed)
    # Set the conditional belief: E[x | belief] = same anchor for every regime.
    # The actual world position is sampled from the declared covariance.
    estimator.filter.mean[:] = ANCHOR
    estimator.filter.cov[:] = np.diag(sig ** 2)
    obs = core.make_observation(estimator.filter.mean, estimator.filter.cov,
                                actual, boxes, mode, True)
    return estimator, obs

# Patching only the isolated copy's function reference. No edits to original projects.
core.reset_case = reset_case


def initial_audit(n_pairs=100):
    """Verify consistency, paired sampling, and shared-observation leakage."""
    sim = core.BotWSMuJoCoBackend(scene_path=core.SCENE)
    boxes = core.m2.scene_boxes(core.SCENE)
    rows = []
    pobs = {}
    bobs = {}
    for k in range(n_pairs):
        seed = 970000 + k
        for regime in ('low', 'high'):
            ef, op = reset_case(sim, boxes, 'point', regime, seed)
            truth = np.asarray(sim.pose(), dtype=float).copy()
            mu = ef.filter.mean.copy()
            cov = ef.filter.cov.copy()
            est_err = truth - mu
            est_err[2] = core.m2.wrap(est_err[2])
            nees = float(est_err @ np.linalg.solve(cov, est_err))
            assert np.all(np.isfinite(op))
            assert np.array_equal(op[20:26], np.zeros(6, dtype=np.float32))
            assert np.allclose(mu, ANCHOR, atol=1e-12)
            assert np.all(np.linalg.eigvalsh(cov) > 0)
            bf, ob = reset_case(sim, boxes, 'belief', regime, seed)
            assert np.allclose(bf.filter.mean, mu)
            assert np.allclose(bf.filter.cov, cov)
            assert np.allclose(sim.pose(), truth, atol=1e-9)
            assert np.array_equal(op[:20], ob[:20])
            assert np.linalg.norm(op[20:26]) == 0
            assert np.linalg.norm(ob[20:26]) > 0
            pobs[regime, k] = op.copy()
            bobs[regime, k] = ob.copy()
            rows.append(dict(seed=seed, regime=regime, true_x=float(truth[0]),
                    true_y=float(truth[1]), true_yaw=float(truth[2]),
                    estimate_x=float(mu[0]), estimate_y=float(mu[1]),
                    estimate_yaw=float(mu[2]), nees=nees,
                    position_error=float(np.linalg.norm(est_err[:2])),
                    covariance_trace=float(np.trace(cov)),
                    ranges_l2_from_nominal=float(np.linalg.norm(op[4:20] -
                                  core.m2.scan(ANCHOR, boxes)/core.m2.RANGE_MAX))))
    # The same seed gives paired noise z before feasible-area rejection.
    for k in range(n_pairs):
        assert np.array_equal(pobs['low', k][:4], pobs['high', k][:4]), (
            'Point goal feature leakage')
        assert np.array_equal(pobs['low', k][20:], pobs['high', k][20:])
        assert not np.array_equal(bobs['low', k][20:26], bobs['high', k][20:26])

    # A deliberately simple regime-prediction test from shared range features:
    # threshold the distance of a scan from the nominal scan. Half held out.
    train = [r for r in rows if r['seed'] < 970000 + n_pairs//2]
    test = [r for r in rows if r['seed'] >= 970000 + n_pairs//2]
    scores = np.array([r['ranges_l2_from_nominal'] for r in train])
    cuts = np.unique(np.r_[0.0, scores, float(np.max(scores) + 1e-8)])
    def accuracy(rs, cutoff):
        return float(np.mean([((r['ranges_l2_from_nominal'] > cutoff) ==
                               (r['regime'] == 'high')) for r in rs]))
    threshold = float(max(cuts, key=lambda c: accuracy(train, c)))
    heldout_accuracy = accuracy(test, threshold)
    with (RUN/'m7b_initial_draws.csv').open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={
        'status':'PASS_INITIALIZATION',
        'num_paired_initial_conditions':n_pairs,
        'mean_start_poses':{regime:list(map(float,np.mean([
            [r['estimate_x'],r['estimate_y'],r['estimate_yaw']]
            for r in rows if r['regime']==regime],axis=0))) for regime in ('low','high')},
        'mean_nees_by_regime':{regime:float(np.mean([r['nees'] for r in rows if r['regime']==regime]))
                             for regime in ('low','high')},
        'mean_start_position_error_by_regime':{regime:float(np.mean([r['position_error'] for r in rows if r['regime']==regime]))
                             for regime in ('low','high')},
        'point_goal_features_identical_across_regimes':True,
        'point_covariance_features_all_zero':True,
        'belief_covariance_features_distinguish_regimes':True,
        'shared_range_leakage_heldout_accuracy':heldout_accuracy,
        'shared_range_leakage_baseline_accuracy':0.5,
        'range_leakage_test':'Single-threshold classifier on distance from nominal 16-ray scan; '
                             'holdout half of paired start seeds; exploratory, not a proof of no information leakage.',
        'limitations':[
          'Random physical starts are sampled from an EKF-centered Gaussian subject to free-space truncation.',
          'Real range sensors remain shared; Point may infer uncertainty statistically from ranges.',
          'EKF process-noise calibration from M2 remains unresolved; this audits only the initial belief.',
          'The controller still uses fixed reference waypoints; the DQN learns route selection only.'
        ]
    }
    (RUN/'m7b_initial_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('M7b initial-belief audit PASS;',n_pairs,'matched seeds; initial mean fixed',flush=True)
    print('Initial NEES low/high:',{k:round(v,3) for k,v in report['mean_nees_by_regime'].items()},flush=True)
    print('Point range-only regime prediction accuracy (held out):',round(heldout_accuracy,3),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--preflight-seeds',type=int,default=12)
    ap.add_argument('--initial-pairs',type=int,default=100)
    ap.add_argument('--train-episodes',type=int,default=240)
    ap.add_argument('--eval-per-regime',type=int,default=24)
    ap.add_argument('--report-every',type=int,default=40)
    args=ap.parse_args()
    core.check_inputs()
    audit=initial_audit(args.initial_pairs)
    if args.preflight_only:
        code=core.run_preflight_only(args)
        result=json.loads((RUN/'m7_preflight_result.json').read_text())
        result['m7b_initial_belief_audit']=audit
        (RUN/'m7b_preflight_report.json').write_text(json.dumps(result,indent=2)+'\n')
        print('M7b preflight decision-value (regime oracle vs constant):',
              round(result['regime_label_information_gain'],4),flush=True)
        return code
    else:
        # Only an explicit later command should invoke training, after examining preflight.
        raise RuntimeError('Training intentionally disabled in M7b preflight. Review results first.')

if __name__=='__main__':
    sys.exit(main())
