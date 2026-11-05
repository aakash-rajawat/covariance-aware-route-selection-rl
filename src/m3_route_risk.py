"""M3: Uncertainty-sensitive two-route gate, using ESTIMATED pose for control.

This is an offline diagnostic. Reference routes are NEVER supplied to an RL
policy or its reward function. There is NO training in M3.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

from m2_validate import (
    START, GOAL, ROUTES, REGIMES, scan, scene_boxes, wrap,
)
from botws_mujoco_backend import BotWSMuJoCoBackend
from belief_nav.maze_estimation import MuJoCoPoseEstimator

RUN = Path(sys.argv[1]).resolve()
SCENE = RUN / 'mini_scene.xml'
N_SEEDS = 24
HORIZON = 375  # M1's wide route used ~235 steps; allow noisy navigation margin.
GOAL_RADIUS = 0.18
WAYPOINT_RADIUS = 0.16


def action(estimated, target):
    """Reference-only fixed controller: uses EKF mean, NEVER ground truth."""
    delta = np.asarray(target, dtype=float) - estimated[:2]
    bearing = math.atan2(delta[1], delta[0])
    error = wrap(bearing - estimated[2])
    v = 0.12 if abs(error) < 0.40 else 0.0
    w = float(np.clip(2.5 * error, -0.6, 0.6))
    return v, w


def rollout(sim, boxes, regime, route, seed, horizon=HORIZON, store_trajectory=False):
    sim.reset(START)
    estimator = MuJoCoPoseEstimator(sim, pose_fix_every=5, **REGIMES[regime])
    est, _ = estimator.reset(seed=int(seed))
    targets = ROUTES[route]
    target_index = 1
    collision = False
    success = False
    track = []
    pos_errors = []
    nees_values = []
    last_trace = None
    initial_error = float(np.linalg.norm(est[:2]-sim.pose()[:2]))
    min_obstacle_range = float('inf')
    near_obstacle_decisions = 0
    for step in range(1, horizon + 1):
        # Reference waypoints exist ONLY for offline route-risk diagnostics.
        # Controller's decision and waypoint advancement use EKF state only.
        if (target_index < len(targets)-1 and
            np.linalg.norm(est[:2]-np.asarray(targets[target_index])) < WAYPOINT_RADIUS):
            target_index += 1
        v, w = action(est, targets[target_index])
        sim.advance(v, w, duration=0.2)
        truth = sim.pose()
        collision = bool(sim.wall_contact())
        filt = estimator.update()
        est = filt['mean']
        cov = filt['covariance']
        last_trace = float(np.trace(cov))
        ranges = scan(truth, boxes)
        min_obstacle_range = min(min_obstacle_range, float(ranges.min()))
        near_obstacle_decisions += int(float(ranges.min()) < 0.23)
        err = np.asarray(est-truth, dtype=float)
        err[2] = wrap(err[2])
        pos_errors.append(float(np.linalg.norm(err[:2])))
        nees = float(err @ np.linalg.solve(cov, err))
        nees_values.append(nees)
        if store_trajectory and (step % 5 == 0 or step == 1 or collision):
            track.append(dict(regime=regime, route=route, seed=seed, step=step,
                true_x=float(truth[0]), true_y=float(truth[1]), true_yaw=float(truth[2]),
                est_x=float(est[0]), est_y=float(est[1]), est_yaw=float(est[2]),
                cov_trace=last_trace, pos_error=pos_errors[-1],
                min_range=float(ranges.min()), target_index=target_index,
                fix=int(filt['fix_received']), contact=int(collision)))
        success = bool((not collision) and np.linalg.norm(truth[:2]-GOAL) < GOAL_RADIUS)
        if success or collision:
            break
    # Identical illustrative outcome reward for both *reference* routes.
    # This is NOT a path-based RL reward.
    ret = (20.0 if success else -20.0 if collision else 0.0) - 0.02*step
    return dict(regime=regime,route=route,seed=seed,steps=step,
        success=int(success),collision=int(collision),timeout=int(not success and not collision),
        reward=ret,initial_pos_err_m=initial_error,
        mean_pos_err_m=float(np.mean(pos_errors)),
        max_pos_err_m=float(np.max(pos_errors)),
        mean_nees=float(np.mean(nees_values)),
        nees_95_coverage=float(np.mean(np.asarray(nees_values)<=7.815)),
        final_cov_trace=float(last_trace),
        min_range_m=float(min_obstacle_range),
        near_obstacle_steps=near_obstacle_decisions,
        corrections=estimator.fix_count,missed_corrections=estimator.dropout_count), track


def summary(rows):
    result = {}
    for reg in ('low','high'):
        result[reg] = {}
        for route in ('short','wide'):
            r = [x for x in rows if x['regime']==reg and x['route']==route]
            assert len(r)==N_SEEDS
            result[reg][route] = {
                'episodes':len(r),
                'successes':sum(x['success'] for x in r),
                'collisions':sum(x['collision'] for x in r),
                'timeouts':sum(x['timeout'] for x in r),
                'success_rate':float(np.mean([x['success'] for x in r])),
                'collision_rate':float(np.mean([x['collision'] for x in r])),
                'mean_return':float(np.mean([x['reward'] for x in r])),
                'mean_steps':float(np.mean([x['steps'] for x in r])),
                'mean_pos_err_m':float(np.mean([x['mean_pos_err_m'] for x in r])),
                'mean_nees':float(np.mean([x['mean_nees'] for x in r])),
                'mean_nees_95_coverage':float(np.mean([x['nees_95_coverage'] for x in r])),
            }
    return result


def plot(results, tracks):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=(7,7))
        ax.add_patch(Rectangle((1.25,1.05),.70,1.45,facecolor='.55', alpha=.7))
        ax.set_xlim(0,3.2);ax.set_ylim(0,3.2)
        for reg, route, seed, style in [
            ('low','short',41000,'-'),('high','short',41000,'--'),
            ('low','wide',41000,'-'),('high','wide',41000,'--')]:
            t=[x for x in tracks if x['regime']==reg and x['route']==route and x['seed']==seed]
            if t:ax.plot([x['true_x'] for x in t],[x['true_y'] for x in t],style,
                         label=f'{reg}: {route}', linewidth=1.7)
        ax.plot([START[0]],[START[1]],'o',label='start')
        ax.plot([GOAL[0]],[GOAL[1]],'*',markersize=13,label='goal')
        ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend()
        ax.set_title('M3: reference-controller paths using EKF estimated pose')
        fig.tight_layout();fig.savefig(RUN/'m3_trajectories.png',dpi=140);plt.close(fig)
        fig, axes=plt.subplots(1,2,figsize=(11,4))
        for a, key, title in [(axes[0],'collision_rate','Collision fraction'),
                              (axes[1],'mean_return','Mean navigation return')]:
            labels=['low short','low wide','high short','high wide']
            values=[results[reg][r][key] for reg in ('low','high') for r in ('short','wide')]
            a.bar(labels,values)
            a.tick_params(axis='x',labelrotation=35)
            a.set_title(title)
            a.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(RUN/'m3_route_risk.png',dpi=140);plt.close(fig)
    except Exception as exc:
        (RUN/'m3_plot_warning.txt').write_text(repr(exc))


def main():
    print('M3: physical two-route risk test. References use EKF mean, NOT truth.',flush=True)
    boxes=scene_boxes(SCENE)
    sim=BotWSMuJoCoBackend(scene_path=SCENE)
    # Integration smoke test: paired one-seed cases, then the locked study.
    for reg in ('low','high'):
        for route in ('short','wide'):
            out,_=rollout(sim,boxes,reg,route,41000,horizon=HORIZON)
            print('  smoke',reg,route,'success',out['success'],
                  'collision',out['collision'],'steps',out['steps'],flush=True)
            assert math.isfinite(out['final_cov_trace']) and out['steps']>0
    rows=[];tracks=[]
    for reg in ('low','high'):
        for route in ('short','wide'):
            for index in range(N_SEEDS):
                seed=41000+index
                data,t=rollout(sim,boxes,reg,route,seed,
                               store_trajectory=(index in (0,1,2)))
                rows.append(data)
                tracks.extend(t)
            r=[x for x in rows if x['regime']==reg and x['route']==route]
            print('  done',reg,route,'success',sum(x['success'] for x in r),
                  'collisions',sum(x['collision'] for x in r),
                  'timeouts',sum(x['timeout'] for x in r),flush=True)
    with (RUN/'m3_episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    with (RUN/'m3_sample_trajectories.csv').open('w',newline='') as f:
        if tracks:
            writer=csv.DictWriter(f,fieldnames=list(tracks[0]))
            writer.writeheader();writer.writerows(tracks)
    scores=summary(rows)
    low_diff=scores['low']['short']['mean_return']-scores['low']['wide']['mean_return']
    high_diff=scores['high']['wide']['mean_return']-scores['high']['short']['mean_return']
    # Reversal is measured but not guaranteed; this is a gate, NOT a forced conclusion.
    risk_reversal=(low_diff>0.5 and high_diff>0.5)
    result={
        'status':'PASS' if risk_reversal else 'FAIL_PREMISE',
        'gate':'low uncertainty prefers short; high uncertainty prefers wide, each by >0.5 mean return',
        'low_short_minus_wide_return':low_diff,
        'high_wide_minus_short_return':high_diff,
        'rollout_count':len(rows),
        'seeds':list(range(41000,41000+N_SEEDS)),
        'results':scores,
        'reference_controller_inputs':'EKF mean and offline fixed route target only; no MuJoCo pose',
        'controlled_variables':'Same initial pose and same estimator seed for both routes in each regime',
        'contact_detection':'MuJoCo contact checked at the end of each 0.2-second control interval',
        'methodology':'Reference route testing ONLY; no policy training, no imitation, no RL waypoint or reward guidance.',
        'limitations':[
          'Fixed route reference controller is not an optimized path follower; failure can reflect its tracking limitations.',
          'Perfect analytic ranges are diagnostic only and not used by this reference controller.',
          'The original EKF is overconfident by M2 NEES, requiring calibration before training.',
          'The test establishes route-dependent risk, not an actual learned Belief advantage.',
          'All routes share one map, not held-out geometric variants.'
        ],
        'recommendation': 'Proceed to calibrated M4 RL interface' if risk_reversal else
            'Do not train; inspect the route failure patterns and adjust geometry or the offline reference controller once.'
    }
    (RUN/'m3_gate.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(scores,tracks)
    print('M3 GATE:',result['status'],flush=True)
    print('low(short-wide) mean return:',round(low_diff,3),flush=True)
    print('high(wide-short) mean return:',round(high_diff,3),flush=True)
    print('Next:',result['recommendation'],flush=True)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        (RUN/'m3_error.txt').write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(2)
