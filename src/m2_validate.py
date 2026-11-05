"""M2: Ray sensing, EKF covariance and matched Point/Belief observation checks.

Uses M1 physical MuJoCo scene and original EKF/robot source copied into the
isolated M2 run. OFFLINE truth-guided reference controls are ONLY for simulator
and estimator validation; neither is used as a policy, expert demo or reward.
"""
import csv
import json
import math
import sys
import traceback
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

RUN = Path(sys.argv[1]).resolve()
SCENE = RUN / 'mini_scene.xml'
METRICS = RUN / 'm2_metrics.json'
START = np.array([0.50, 2.50, 0.0])
GOAL = np.array([2.70, 2.50])
N_RAYS = 16
RANGE_MAX = 2.5
WORLD_DIAGONAL = (2 * 3.2**2)**0.5
COV_SCALES = np.array([0.02, 0.02, 0.01, 0.02, 0.02, 0.02], dtype=np.float64)
ROUTES = {
    'short': [[.50,2.50],[.90,2.83],[2.30,2.83],[2.70,2.50]],
    'wide': [[.50,2.50],[.90,.65],[2.30,.65],[2.70,2.50]],
}
REGIMES = {
    'low': dict(pose_dropout=.10, encoder_delta_std=.0015, pose_position_std=.035,
                pose_heading_std=.025, initial_position_std=.035, initial_heading_std=.025),
    'moderate': dict(pose_dropout=.50, encoder_delta_std=.0025, pose_position_std=.075,
                pose_heading_std=.050, initial_position_std=.075, initial_heading_std=.050),
    'high': dict(pose_dropout=.90, encoder_delta_std=.0050, pose_position_std=.120,
                pose_heading_std=.080, initial_position_std=.120, initial_heading_std=.080),
}


def wrap(theta):
    return (float(theta) + math.pi) % (2*math.pi) - math.pi


def scene_boxes(xml):
    """Exact axis-aligned 2-D AABBs of planar world wall geoms."""
    world = ET.parse(xml).getroot().find('worldbody')
    boxes = []
    for elem in world.findall('geom'):
        if elem.get('type') != 'box':
            continue
        name = elem.get('name')
        if name not in {'border_south','border_north','border_west','border_east','central_block'}:
            continue
        pos = np.fromstring(elem.get('pos'), sep=' ')
        half = np.fromstring(elem.get('size'), sep=' ')
        boxes.append((name, pos[0]-half[0],pos[0]+half[0],pos[1]-half[1],pos[1]+half[1]))
    if len(boxes) != 5:
        raise AssertionError(f'Expected 5 static walls; found {boxes}')
    return boxes


def ray_box(origin, direction, box):
    """Slab intersection distance of ray and 2-D AABB; +inf if none."""
    _, xmin,xmax,ymin,ymax = box
    lo = -float('inf')
    hi = float('inf')
    for origin_i, direct_i, bmin, bmax in ((origin[0],direction[0],xmin,xmax),
                                           (origin[1],direction[1],ymin,ymax)):
        if abs(direct_i) < 1e-14:
            if origin_i < bmin or origin_i > bmax:
                return float('inf')
            continue
        t1 = (bmin-origin_i)/direct_i
        t2 = (bmax-origin_i)/direct_i
        lo=max(lo,min(t1,t2))
        hi=min(hi,max(t1,t2))
    if hi < max(lo,0.0):
        return float('inf')
    return max(lo,0.0)


def scan(true_pose, boxes):
    """16 idealized planar ranges from TRUE sensor pose, never from EKF mean."""
    xy = np.asarray(true_pose[:2])
    yaw = float(true_pose[2])
    vals=[]
    for i in range(N_RAYS):
        angle = yaw + 2*math.pi*i/N_RAYS
        direction=(math.cos(angle), math.sin(angle))
        vals.append(min(RANGE_MAX, *(ray_box(xy,direction,b) for b in boxes)))
    return np.asarray(vals,dtype=np.float32)


def observation(est_pose, P, ranges, mode):
    """Common 26-input network. Last six features zero for Point."""
    dx,dy = GOAL-np.asarray(est_pose[:2])
    yaw=float(est_pose[2])
    forward=dx*math.cos(yaw)+dy*math.sin(yaw)
    lateral=-dx*math.sin(yaw)+dy*math.cos(yaw)
    dist=math.hypot(dx,dy)
    bearing=math.atan2(lateral,forward)
    goal=np.array([forward/WORLD_DIAGONAL,lateral/WORLD_DIAGONAL,
                   dist/WORLD_DIAGONAL,bearing/math.pi],dtype=np.float32)
    cov=np.asarray(P,dtype=np.float64)
    ordered=np.array([cov[0,0],cov[1,1],cov[2,2],cov[0,1],cov[0,2],cov[1,2]])
    if mode=='point':
        extra=np.zeros(6,dtype=np.float32)
    elif mode=='belief':
        extra=np.tanh(ordered/COV_SCALES).astype(np.float32)
    else:
        raise ValueError(mode)
    out=np.concatenate((goal,np.asarray(ranges,dtype=np.float32)/RANGE_MAX,extra))
    if out.shape!=(26,) or not np.all(np.isfinite(out)):
        raise AssertionError('Bad observation')
    if not (np.max(np.abs(out))<=1.001):
        raise AssertionError('Observation exceeded normalization bounds')
    return out


def offline_reference_action(truth, target):
    """Privileged route-following for validation ONLY; never for DQN."""
    dif=np.asarray(target)-truth[:2]
    heading_error=wrap(math.atan2(dif[1],dif[0])-truth[2])
    w=float(np.clip(2.5*heading_error,-.6,.6))
    v=.12 if abs(heading_error)<.40 else 0.
    return v,w


def run_trace(sim, boxes, Estimator, regime, seed, steps=125, route='short', collect=False):
    sim.reset(START)
    filt=Estimator(sim,pose_fix_every=5,**REGIMES[regime])
    filt.reset(seed=seed)
    nodes=ROUTES[route]
    target_i=1
    trows=[]
    prior_diag=np.diag(filt.covariance).copy()
    check_cov=0
    correction_check=0
    point_last=belief_last=None
    reached=False
    collision=False
    for t in range(steps):
        ground=sim.pose()
        target=np.asarray(nodes[target_i])
        if np.linalg.norm(ground[:2]-target)<.14 and target_i<len(nodes)-1:
            target_i+=1
            target=np.asarray(nodes[target_i])
        v,w=offline_reference_action(ground,target)
        sim.advance(v,w)
        truth=sim.pose()
        collision=bool(sim.wall_contact())
        result=filt.update()
        P=result['covariance']
        mean=result['mean']
        eigs=np.linalg.eigvalsh(P)
        assert np.all(np.isfinite(P)) and np.min(eigs)>-1e-10
        assert np.max(np.abs(P-P.T))<1e-10
        check_cov+=1
        if result['fix_received']:
            assert result['posterior_covariance_trace']<result['prior_covariance_trace']
            correction_check+=1
        rs=scan(truth,boxes)
        assert rs.shape==(16,) and np.all((rs>=0)&(rs<=RANGE_MAX))
        obs_p=observation(mean,P,rs,'point')
        obs_b=observation(mean,P,rs,'belief')
        assert np.array_equal(obs_p[:20],obs_b[:20])
        assert np.all(obs_p[20:]==0)
        point_last=obs_p
        belief_last=obs_b
        err=np.array([mean[0]-truth[0],mean[1]-truth[1],wrap(mean[2]-truth[2])])
        try:
            nees=float(err@np.linalg.solve(P,err))
        except np.linalg.LinAlgError:
            nees=float('nan')
        trows.append(dict(regime=regime,seed=seed,step=t+1,route=route,
                true_x=truth[0],true_y=truth[1],true_heading=truth[2],
                est_x=mean[0],est_y=mean[1],est_heading=mean[2],
                pos_err_m=float(np.linalg.norm(err[:2])),heading_err_rad=err[2],
                pxx=P[0,0],pyy=P[1,1],pyaw=P[2,2],
                ptrace=float(np.trace(P)),fix=int(result['fix_received']),
                dropped=int(result['fix_attempted'] and not result['fix_received']),
                nees=nees, min_range_m=float(rs.min()),
                collision=int(collision)))
        if collision or np.linalg.norm(truth[:2]-GOAL)<.18:
            reached=(not collision and np.linalg.norm(truth[:2]-GOAL)<.18)
            break
    return dict(rows=trows,steps=len(trows),reached=reached,collision=collision,
                fixes=filt.fix_count,dropped=filt.dropout_count,
                final_point=point_last,final_belief=belief_last,
                initial_cov_diag=prior_diag,corrections_verified=correction_check,
                covariance_steps_verified=check_cov)


def main():
    from botws_mujoco_backend import BotWSMuJoCoBackend
    from belief_nav.maze_estimation import MuJoCoPoseEstimator
    boxes=scene_boxes(SCENE)
    sim=BotWSMuJoCoBackend(scene_path=SCENE)
    checks={}
    def chk(name,value,extra=None):
        checks[name]={'pass':bool(value),'details':extra}
        print(' ',name,'PASS' if value else 'FAIL',extra if extra is not None else '',flush=True)
        if not value: raise AssertionError(name)

    print('M2: sensor + covariance validation',flush=True)
    at_start=scan(START,boxes)
    chk('ray_count',len(at_start)==16 and np.all(np.isfinite(at_start)),at_start.round(3).tolist())
    # Bearing 0: obstacle directly east ~.75m; west boundary ~.4m;
    # north boundary ~.6m; south boundary ~2.4m.
    chk('range_direction_consistency',
        abs(float(at_start[0])-.75)<.03 and abs(float(at_start[4])-.60)<.03 and
        abs(float(at_start[8])-.40)<.03 and abs(float(at_start[12])-2.40)<.03,
        dict(east=float(at_start[0]),north=float(at_start[4]),west=float(at_start[8]),south=float(at_start[12])))
    chk('angle_rotation',np.allclose(scan([.5,2.5,math.pi/2],boxes),np.roll(at_start,-4),atol=1e-5))
    # Covariance-only changes must never influence Point values or shared first 20 values.
    mean=START+np.array([.05,-.02,.03])
    p1=np.diag([.001,.001,.001]); p2=np.diag([.08,.08,.04])
    point1=observation(mean,p1,at_start,'point');point2=observation(mean,p2,at_start,'point')
    belief1=observation(mean,p1,at_start,'belief');belief2=observation(mean,p2,at_start,'belief')
    chk('covariance_only_ablation',np.array_equal(point1,point2) and
        np.array_equal(belief1[:20],belief2[:20]) and
        not np.array_equal(belief1[20:],belief2[20:]))
    chk('point_padding',np.array_equal(point1[20:],np.zeros(6,dtype=np.float32)))

    # Seed repeatability, paired identical MuJoCo truth controls but independent EKF draws.
    a=run_trace(sim,boxes,MuJoCoPoseEstimator,'moderate',20260,steps=65)
    b=run_trace(sim,boxes,MuJoCoPoseEstimator,'moderate',20260,steps=65)
    chk('seeded_ekf_repeatability',a['rows']==b['rows'] and
        np.array_equal(a['final_belief'],b['final_belief']))
    records=[];aggregates={}; total_corrections=0
    for regime in REGIMES:
        runs=[]
        # Pilot multiple complete paths or partial paths; deterministic reference
        # only for isolating sensor/estimator error, not used by either RL agent.
        for index in range(10):
            seed=30000+index
            route='short' if index%2==0 else 'wide'
            out=run_trace(sim,boxes,MuJoCoPoseEstimator,regime,seed,
                          steps=125,route=route)
            records.extend(out['rows'])
            runs.append(out)
            total_corrections+=out['corrections_verified']
            if index==0:
                print(' ',regime,'first-run',dict(steps=out['steps'],fixes=out['fixes'],
                        dropped=out['dropped'],collision=out['collision']),flush=True)
        arr=[r for r in records if r['regime']==regime]
        e=np.asarray([r['pos_err_m'] for r in arr])
        ptr=np.asarray([r['ptrace'] for r in arr])
        pxx=np.asarray([r['pxx'] for r in arr])
        pyy=np.asarray([r['pyy'] for r in arr])
        # A radial 2-sigma test is heuristic; use NEES for joint calibration.
        nees=np.asarray([r['nees'] for r in arr]); nees=nees[np.isfinite(nees)]
        aggregates[regime]=dict(rollouts=len(runs),samples=len(arr),
            successes=sum(x['reached'] for x in runs),
            collisions=sum(x['collision'] for x in runs),
            fix_count=sum(x['fixes'] for x in runs),
            dropped_count=sum(x['dropped'] for x in runs),
            mean_pos_error_m=float(e.mean()),p95_pos_error_m=float(np.quantile(e,.95)),
            mean_position_variance_m2=float(np.mean(pxx+pyy)),
            mean_trace=float(ptr.mean()),
            mean_nees=float(nees.mean()),
            nees_fraction_under_7_815=float(np.mean(nees<=7.815)),
            estimated_position_std_m=float(np.sqrt(np.mean(pxx+pyy))),
            mean_final_cov_trace=float(np.mean([x['rows'][-1]['ptrace'] for x in runs])))
    chk('all_covariances_finite',len(records)>500 and all(math.isfinite(x['ptrace']) for x in records),len(records))
    chk('pose_updates_verified',total_corrections>50,total_corrections)
    chk('increased_uncertainty',aggregates['high']['mean_trace']>aggregates['moderate']['mean_trace']>
        aggregates['low']['mean_trace'],{k:round(v['mean_trace'],5) for k,v in aggregates.items()})
    chk('nonzero_error',all(v['mean_pos_error_m']>0.001 for v in aggregates.values()))
    # Calibration is reported, not hidden behind a PASS that overstates quality.
    # Do not use ground truth as an input to the future policy.
    with (RUN/'m2_trace.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    with (RUN/'m2_observation_example.json').open('w') as f:
        json.dump(dict(point=point1.tolist(),belief_low=belief1.tolist(),
                       belief_high=belief2.tolist(),covariance_scales=COV_SCALES.tolist(),
                       physical_truth_not_in_observation=True),f,indent=2)
    output=dict(status='PASS',uncertainty_regimes=REGIMES,tests=checks,
                aggregates=aggregates,samples=len(records),
                methodology='Reference routes/truth used ONLY for EKF diagnostic; neither policy is trained or receives planner guidance.',
                limitations=['Analytic idealized 16-ray range sensor, not simulated LiDAR noise.',
                             'Fixed covariance process noise in original EKF; assess NEES calibration before M3.',
                             'Reference controller uses ground truth only for controlled physical motion.',
                             'No DQN training performed.'])
    METRICS.write_text(json.dumps(output,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n')
    print('M2 OVERALL PASS',flush=True)
    for k,v in aggregates.items():
        print(' ',k, 'position_rmse-ish mean error', round(v['mean_pos_error_m'],4),
              'mean trace',round(v['mean_trace'],5),'mean NEES',round(v['mean_nees'],2),flush=True)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True)
        for k in REGIMES:
            lines=[x for x in records if x['regime']==k and x['seed']==30000]
            axs[0].plot([x['step'] for x in lines],[x['pos_err_m'] for x in lines],label=k)
            axs[1].plot([x['step'] for x in lines],[x['ptrace'] for x in lines],label=k)
        axs[0].set_ylabel('Position estimation error (m)')
        axs[1].set_ylabel('Covariance trace'); axs[1].set_xlabel('Decision step')
        for ax in axs: ax.grid(alpha=.2); ax.legend()
        fig.suptitle('M2 — same physical reference route, different localization regimes')
        fig.tight_layout();fig.savefig(RUN/'m2_error_covariance.png',dpi=140);plt.close(fig)
    except Exception as exc:
        (RUN/'plot_warning.txt').write_text(repr(exc))
    return 0


if __name__=='__main__':
    try: sys.exit(main())
    except Exception as ex:
        (RUN/'m2_error.txt').write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(2)
