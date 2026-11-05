"""M7: High-level Point-vs-Belief DQN over real MuJoCo reference-controller outcomes.

This is OPTION-SELECTION RL: low-level wheel commands come from the same fixed,
EKF-based route controller for both learners. No demonstrations or optimal route
labels are supplied to either DQN. Initial observations never include ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from collections import deque

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
sys.path.insert(0, str(RUN / 'reference'))

from botws_mujoco_backend import BotWSMuJoCoBackend
from belief_nav.maze_estimation import MuJoCoPoseEstimator
from maze_dqn import MazeDQNAgent

class MaskedMazeDQNAgent(MazeDQNAgent):
    """Double DQN with action 2 disabled after a localization request."""
    def learn(self, rng, batch_size=32):
        if len(self.replay) < batch_size:
            return None
        idx = rng.choice(len(self.replay), size=batch_size, replace=False)
        batch = [self.replay[int(i)] for i in idx]
        obs=torch.as_tensor(np.stack([b[0] for b in batch]), device=self.device)
        action=torch.as_tensor([b[1] for b in batch],dtype=torch.int64,device=self.device).unsqueeze(1)
        reward=torch.as_tensor([b[2] for b in batch],dtype=torch.float32,device=self.device)
        nxt=torch.as_tensor(np.stack([b[3] for b in batch]),device=self.device)
        terminated=torch.as_tensor([b[4] for b in batch],dtype=torch.float32,device=self.device)
        current=self.online(obs).gather(1,action).squeeze(1)
        with torch.no_grad():
            q=self.online(nxt)
            q[nxt[:,-1]<.5,2]=-1e9
            selected=q.argmax(dim=1,keepdim=True)
            next_val=self.target(nxt).gather(1,selected).squeeze(1)
            target=reward+self.gamma*(1.-terminated)*next_val
        loss=torch.nn.functional.smooth_l1_loss(current,target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(),10.)
        self.optimizer.step()
        self.updates+=1
        if self.updates%self.target_update_steps==0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
import m2_validate as m2
import m3_route_risk as m3

SCENE = RUN / 'mini_scene_m3b.xml'
START = np.array([.50, 2.20, 0.0], dtype=float)
GOAL = np.array([2.65, 2.20], dtype=float)
ROUTES = {
    'short': [[.50,2.20],[.85,2.86],[2.35,2.86],[2.65,2.20]],
    'wide': [[.50,2.20],[.85,1.15],[2.35,1.15],[2.35,2.20],[2.65,2.20]],
}
# Explicitly reproduce the tested M3b route geometry from its saved gate.
# The correct wide penultimate waypoint is [2.35, 2.15], verified below.
ROUTES['wide'][-2] = [2.35, 2.15]

m2.GOAL = GOAL
m3.START = START
m3.GOAL = GOAL
m3.ROUTES = ROUTES
m3.SCENE = SCENE
m3.HORIZON = 375
m3.WAYPOINT_RADIUS = 0.16

GOAL_RADIUS = 0.18
HORIZON = 375
STEP_COST = 0.02
SUCCESS_REWARD = 20.0
COLLISION_PENALTY = -20.0
FIX_COST = 1.0
FIX_POSITION_STD = 0.025
FIX_HEADING_STD = 0.020
DIM = 27
N_ACTIONS = 3
RANGE_COUNT = 16


def check_inputs():
    gate = json.loads((RUN / 'm3b_gate.json').read_text())
    if gate.get('status') != 'PASS':
        raise ValueError('M3b geometry gate was not PASS')
    rg = gate['precommitted_geometry']['routes']
    if rg != ROUTES:
        raise ValueError(f'M3b reference route specification changed: {rg!r} != {ROUTES!r}')
    if not SCENE.is_file():
        raise FileNotFoundError(SCENE)
    if not (RUN / 'reference/belief_nav/maze_estimation.py').is_file():
        raise FileNotFoundError('EKF source unavailable')
    if not (RUN / 'maze_dqn.py').is_file():
        raise FileNotFoundError('DQN source unavailable')
    summary = gate['results']
    pay = {reg:{act:summary[reg][act]['mean_return'] for act in ('short','wide')}
           for reg in ('low','high')}
    oracle = .5 * sum(max(pay[reg].values()) for reg in pay)
    best_fixed = max(.5 * sum(pay[reg][act] for reg in pay)
                     for act in ('short','wide'))
    report = dict(geometry_gate='PASS', empirical_reference_return=pay,
                  oracle_regime_aware_return=oracle,
                  best_uncertainty_blind_fixed_route_return=best_fixed,
                  extra_information_return=oracle-best_fixed,
                  caveat='Reference controllers and balanced low/high regime assumption. '
                         'Not a bound on all possible learned policies; observation-based '
                         'Point policy can sometimes infer regime indirectly.')
    (RUN / 'm7_value_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Reference-only value of regime information:',round(oracle-best_fixed,4),
          '(small; no guaranteed Belief win)',flush=True)
    return report


def make_observation(estimated, covariance, truth, boxes, mode, can_fix):
    raw = m2.observation(estimated, covariance, m2.scan(truth,boxes), mode)
    assert raw.shape == (26,)
    obs = np.concatenate((raw,np.array([float(can_fix)],dtype=np.float32)))
    if obs.shape != (DIM,) or not np.isfinite(obs).all():
        raise AssertionError('invalid policy observation')
    if mode == 'point' and not np.array_equal(obs[20:26],np.zeros(6)):
        raise AssertionError('Point observation leaked covariance')
    return obs


def choose_action(agent,obs,rng,eps,can_fix):
    valid = np.array([0,1,2] if can_fix else [0,1],dtype=np.int64)
    if rng.random() < eps:
        return int(rng.choice(valid))
    with torch.no_grad():
        q=agent.online(torch.as_tensor(obs,device=agent.device).unsqueeze(0))
        q=q[0,valid].detach().cpu().numpy()
    return int(valid[int(np.argmax(q))])


def reset_case(sim, boxes, mode, regime, seed):
    sim.reset(START)
    estimator = MuJoCoPoseEstimator(sim,pose_fix_every=5,**m2.REGIMES[regime])
    est,P = estimator.reset(seed=seed)
    return estimator,make_observation(est,P,sim.pose(),boxes,mode,True)


def ask_for_fix(sim,estimator,rng):
    """One explicit paid active localization observation; no free motion or truth leak."""
    measurement = sim.pose() + rng.normal(0,[FIX_POSITION_STD,FIX_POSITION_STD,FIX_HEADING_STD])
    measurement[2] = m2.wrap(measurement[2])
    estimator.filter.update(measurement,position_std=FIX_POSITION_STD,
                            heading_std=FIX_HEADING_STD)
    return estimator.filter.mean.copy(),estimator.filter.cov.copy()


def execute_option(sim,estimator,route,boxes):
    """Fixed route execution, with wheel commands derived from EKF MEAN only."""
    targets=ROUTES[route]
    target_index=1
    success=False;collision=False
    min_range=float('inf')
    for steps in range(1,HORIZON+1):
        estimated=estimator.filter.mean.copy()
        if target_index < len(targets)-1 and (
            np.linalg.norm(estimated[:2]-np.asarray(targets[target_index]))<m3.WAYPOINT_RADIUS
        ):
            target_index+=1
        v,w = m3.action(estimated,targets[target_index])
        sim.advance(v,w,duration=0.2)
        truth=sim.pose()
        collision=bool(sim.wall_contact())
        filt=estimator.update()
        rng=m2.scan(truth,boxes)
        min_range=min(min_range,float(rng.min()))
        success=bool(not collision and np.linalg.norm(truth[:2]-GOAL)<GOAL_RADIUS)
        if success or collision:
            break
    rew=(SUCCESS_REWARD if success else COLLISION_PENALTY if collision else 0.)-STEP_COST*steps
    return dict(route=route,steps=steps,success=int(success),collision=int(collision),
                timeout=int(not success and not collision),reward=float(rew),
                min_range=float(min_range),final_cov_trace=float(np.trace(estimator.filter.cov)))


def run_episode(agent, sim, boxes, mode,regime,seed,choice_rng,fix_rng,epsilon,
                training,updates_per_episode=8,obs_mode='live'):
    estimator,obs=reset_case(sim,boxes,mode,regime,seed)
    initial_cov = float(np.trace(estimator.filter.cov))
    if obs_mode != 'live' and mode=='belief':
        obs=obs.copy()
        if obs_mode=='zero':obs[20:26]=0
        elif obs_mode=='fixed':obs[20:26]=np.tanh(np.array([.006,.006,.002,0,0,0])/m2.COV_SCALES)
        else:raise ValueError(obs_mode)
    first=choose_action(agent,obs,choice_rng,epsilon,can_fix=True)
    did_fix=(first==2)
    if did_fix:
        new_est,new_cov=ask_for_fix(sim,estimator,fix_rng)
        obs1=make_observation(new_est,new_cov,sim.pose(),boxes,mode,False)
        if obs_mode!='live' and mode=='belief':
            obs1=obs1.copy()
            if obs_mode=='zero':obs1[20:26]=0
            elif obs_mode=='fixed':obs1[20:26]=np.tanh(np.array([.006,.006,.002,0,0,0])/m2.COV_SCALES)
        second=choose_action(agent,obs1,choice_rng,epsilon,can_fix=False)
        if training:
            agent.store(obs,2,-FIX_COST,obs1,terminated=False)
    else:
        obs1=obs
        second=first
    route='short' if second==0 else 'wide'
    out=execute_option(sim,estimator,route,boxes)
    if training:
        agent.store(obs1,int(second),out['reward'],obs1,terminated=True)
        for _ in range(updates_per_episode):
            agent.learn(choice_rng,batch_size=32)
    out.update(mode=mode,regime=regime,seed=seed,request=int(did_fix),
               first_action=first,route_action=second,net_return=out['reward']-FIX_COST*int(did_fix),
               initial_cov_trace=initial_cov,obs_mode=obs_mode)
    return out


def write_csv(path,records):
    if not records:
        return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def stats(records):
    result={}
    for regime in ('low','high','all'):
        subset=[r for r in records if regime=='all' or r['regime']==regime]
        if not subset: continue
        result[regime]=dict(n=len(subset),success=sum(r['success'] for r in subset),
                            collisions=sum(r['collision'] for r in subset),
                            timeout=sum(r['timeout'] for r in subset),
                            mean_return=float(np.mean([r['net_return'] for r in subset])),
                            request_rate=float(np.mean([r['request'] for r in subset])),
                            narrow_rate=float(np.mean([r['route']=='short' for r in subset])))
    return result


def run_preflight_only(args):
    value=check_inputs()
    sim=BotWSMuJoCoBackend(scene_path=SCENE)
    boxes=m2.scene_boxes(SCENE)
    records=[]
    for regime in ('low','high'):
        for condition in ('normal','fix'):
            for route in ('short','wide'):
                for k in range(args.preflight_seeds):
                    seed=785000+k
                    estimator,obs=reset_case(sim,boxes,'belief',regime,seed)
                    if condition=='fix':
                        ask_for_fix(sim,estimator,np.random.default_rng(seed+777))
                    result=execute_option(sim,estimator,route,boxes)
                    result.update(regime=regime,condition=condition,seed=seed,
                                  net_return=result['reward']-(FIX_COST if condition=='fix' else 0))
                    records.append(result)
                sub=records[-args.preflight_seeds:]
                print('Preflight',regime,condition,route,
                      'success',sum(x['success'] for x in sub),'/',len(sub),
                      'mean return',round(float(np.mean([x['net_return'] for x in sub])),3),flush=True)
    write_csv(RUN/'m7_preflight_episodes.csv',records)
    means={regime:{f'{condition}_{route}':float(np.mean([r['net_return'] for r in records
                if r['regime']==regime and r['condition']==condition and r['route']==route]))
            for condition in ('normal','fix') for route in ('short','wide')}
            for regime in ('low','high')}
    opts=list(means['low'])
    best_fixed=max(.5*sum(means[regime][a] for regime in ('low','high')) for a in opts)
    oracle=.5*sum(max(means[regime].values()) for regime in ('low','high'))
    v=dict(status='PASS_EXECUTION', reference=means, oracle_regime_label=oracle,
           best_constant_macro=best_fixed, regime_label_information_gain=oracle-best_fixed,
           note='Only a regime-label-versus-constant macro diagnostic, NOT a proven '
                'Point-vs-Belief information advantage. Shared observations may reveal reliability. '
                'A real MuJoCo trajectory with EKF estimated-pose controller is used for each action.',
           preflight_seeds=args.preflight_seeds, original_m3b_value=value)
    (RUN/'m7_preflight_result.json').write_text(json.dumps(v,indent=2)+'\n')
    print('Preflight value gap over best constant macro:',round(oracle-best_fixed,3),flush=True)
    return 0


def run(args):
    if args.preflight_only:
        return run_preflight_only(args)
    if args.train_episodes < 32 or args.eval_per_regime < 2:
        raise ValueError('Need >=32 training episodes and >=2 per regime evaluation')
    torch.set_num_threads(1)
    preflight=check_inputs()
    sim=BotWSMuJoCoBackend(scene_path=SCENE)
    boxes=m2.scene_boxes(SCENE)
    # Cheap real-physics smoke checks with no learning.
    checks=[]
    for regime in ('low','high'):
        for mode in ('point','belief'):
            estimator,obs=reset_case(sim,boxes,mode,regime,seed=90201)
            checks.append(dict(mode=mode,regime=regime,observation_shape=list(obs.shape),
                               trace=float(np.trace(estimator.filter.cov)),
                               point_zero_cov=bool(np.all(obs[20:26]==0))))
    (RUN/'m7_smoke.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert all(x['observation_shape']==[DIM] for x in checks)
    assert all(x['point_zero_cov'] for x in checks if x['mode']=='point')
    assert next(x['trace'] for x in checks if x['mode']=='belief' and x['regime']=='high') > next(x['trace'] for x in checks if x['mode']=='belief' and x['regime']=='low')
    print('M7 smoke: PASS (27 features, covariance gated, uncertainty ordering)',flush=True)
    statuses={}; evaluations=[]
    train_schedule = np.array(['low']*(args.train_episodes//2)+
                              ['high']*(args.train_episodes-args.train_episodes//2))
    np.random.default_rng(11442).shuffle(train_schedule)
    for mode in ('point','belief'):
        agent=MaskedMazeDQNAgent(observation_dim=DIM,n_actions=N_ACTIONS,
                           gamma=.995, lr=4e-4, replay_capacity=4000,
                           target_update_steps=120,seed=92810,device='cpu')
        crng=np.random.default_rng(20321)
        frng=np.random.default_rng(18372)
        trainrows=[]
        for i,regime in enumerate(train_schedule):
            eps=max(.06,1.0-.94*(i/max(1,args.train_episodes-1)))
            out=run_episode(agent,sim,boxes,mode,str(regime),seed=720000+i,
                    choice_rng=crng,fix_rng=frng,epsilon=eps,training=True)
            trainrows.append(out)
            if (i+1)%args.report_every==0 or i+1==args.train_episodes:
                recent=trainrows[-args.report_every:]
                print(f'{mode} training {i+1}/{args.train_episodes}: '
                      f'success {sum(x["success"] for x in recent)}/{len(recent)}; '
                      f'narrow {sum(x["route"]=="short" for x in recent)}; '
                      f'fixes {sum(x["request"] for x in recent)}',flush=True)
        write_csv(RUN/f'm7_train_{mode}.csv',trainrows)
        agent.save(RUN/f'm7_{mode}_policy.pt')
        statuses[mode]={'training':stats(trainrows),'updates':agent.updates}
        for observation_mode in (('live','fixed','zero') if mode=='belief' else ('live',)):
            evrows=[]
            for regime in ('low','high'):
                for j in range(args.eval_per_regime):
                    seed=890000+(0 if regime=='low' else 100000)+j
                    out=run_episode(agent,sim,boxes,mode,regime,seed=seed,
                                    choice_rng=np.random.default_rng(3000+j),
                                    fix_rng=np.random.default_rng(6000+j),
                                    epsilon=0.0,training=False,obs_mode=observation_mode)
                    evrows.append(out)
                    evaluations.append(out)
            statuses[f'{mode}_{observation_mode}_eval']=stats(evrows)
            print(mode,observation_mode,'evaluation',stats(evrows)['all'],flush=True)
    write_csv(RUN/'m7_evaluation.csv',evaluations)
    report={'status':'PASS_EXECUTION','hypothesis_supported':None,
            'interpretation':'PASS_EXECUTION only means models trained and were evaluated, not Belief > Point.',
            'note':'One high-level choice, optionally followed by paid localization request. '
                   'The fixed controller uses route waypoints; policy does not learn wheel steering.',
            'train_episodes_per_policy':args.train_episodes,
            'eval_episodes_per_regime':args.eval_per_regime,
            'preflight':preflight,'results':statuses,
            'limitations':['Single M3b geometry; no cross-map generalization.',
                           'The EKF was previously overconfident by NEES.',
                           'One training seed, exploratory evidence.',
                           'Route execution uses fixed, EKF-based waypoint controllers.',
                           'Point may infer part of reliability indirectly through shared sensors.']}
    point=statuses['point_live_eval']['all']['mean_return']
    belief=statuses['belief_live_eval']['all']['mean_return']
    belief_fixed=statuses['belief_fixed_eval']['all']['mean_return']
    report['return_delta_belief_minus_point']=belief-point
    report['return_delta_belief_live_minus_fixed']=belief-belief_fixed
    (RUN/'m7_results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4))
        for key,label in [('point_live_eval','Point'),('belief_live_eval','Belief'),('belief_fixed_eval','Belief fixed P')]:
            data=statuses[key]
            ax.plot([0,1],[data['low']['mean_return'],data['high']['mean_return']],marker='o',label=label)
        ax.set_xticks([0,1],['Low','High']);ax.set_ylabel('Mean navigation return')
        ax.set_title('M7 high-level decision: preliminary results');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(RUN/'m7_comparison.png',dpi=160);plt.close(fig)
    except Exception as exc:
        (RUN/'m7_plot_warning.txt').write_text(repr(exc))
    print('M7 FINISHED: execution pass; Belief - Point return',round(belief-point,3),
          'Belief live - fixed',round(belief-belief_fixed,3),flush=True)
    return 0

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--preflight-seeds',type=int,default=12)
    ap.add_argument('--train-episodes',type=int,default=240)
    ap.add_argument('--eval-per-regime',type=int,default=24)
    ap.add_argument('--report-every',type=int,default=40)
    args=ap.parse_args()
    try:
        sys.exit(run(args))
    except Exception:
        traceback.print_exc()
        (RUN/'m7_error.txt').write_text(traceback.format_exc())
        sys.exit(2)
