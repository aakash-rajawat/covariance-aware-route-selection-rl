"""M7d: independent-seed replication + fixed-strategy baselines for M7c.

Run in a copied directory containing the verified M7c sources and M3b scene.
NO modification of the original robotics repositories, no new maze layout.

Uses M7c's patched M7b conditional initialization, M7c's shared high-level
range masking, and the same Double DQN learning algorithm / hyperparameters.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch

# Crucial: import M7c FIRST to activate its observation and M7b reset patches.
import m7c_train as m7c

core = m7c.core
RUN = Path(__file__).resolve().parent
EVAL_COUNT = 24
EVAL_START = 990000  # Disjoint from original 890000 eval, and 720000 training.
SEED_STRIDE = 10000
SEED_MIX = 9973


def write_csv(path: Path, records: list[dict]):
    if not records:
        raise ValueError('No records to write')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def audit() -> dict:
    core.check_inputs()
    # M7c's audit confirms no range/covariance leakage in Point observations.
    info = m7c.audit()
    if info.get('status') != 'PASS':
        raise RuntimeError('Original M7c observation audit failed')
    return info


def train(seed_index: int, train_episodes: int = 240) -> dict:
    if seed_index not in (1, 2):
        raise ValueError('Only predeclared replication seeds 1, 2 are supported')
    if train_episodes != 240:
        raise ValueError('Protocol frozen at 240 train episodes per policy')
    audit()
    torch.set_num_threads(1)
    sim = core.BotWSMuJoCoBackend(scene_path=core.SCENE)
    boxes = core.m2.scene_boxes(core.SCENE)
    schedule = np.array(['low'] * 120 + ['high'] * 120)
    np.random.default_rng(11442 + seed_index * SEED_MIX).shuffle(schedule)
    statuses = {}
    for mode in ('point', 'belief'):
        agent = core.MaskedMazeDQNAgent(
            observation_dim=27, n_actions=3,
            gamma=.995, lr=4e-4, replay_capacity=4000,
            target_update_steps=120,
            seed=92810 + seed_index * SEED_MIX,
            device='cpu',
        )
        choice_rng = np.random.default_rng(20321 + seed_index * SEED_MIX)
        fix_rng = np.random.default_rng(18372 + seed_index * SEED_MIX)
        records = []
        for i, regime in enumerate(schedule):
            eps = max(.06, 1.0 - .94 * i / 239)
            outcome = core.run_episode(
                agent, sim, boxes, mode, str(regime),
                seed=720000 + seed_index * SEED_STRIDE + i,
                choice_rng=choice_rng, fix_rng=fix_rng,
                epsilon=eps, training=True,
            )
            records.append(outcome)
            if (i + 1) % 40 == 0:
                last = records[-40:]
                print(f'seed {seed_index} {mode} training {i+1}/240: '
                      f'success={sum(x["success"] for x in last)}/40, '
                      f'wide={sum(x["route"] == "wide" for x in last)}, '
                      f'fix={sum(x["request"] for x in last)}', flush=True)
        write_csv(RUN / f'm7d_train_{mode}.csv', records)
        agent.save(RUN / f'm7d_{mode}_policy.pt')
        statuses[mode] = {
            'training': core.stats(records),
            'updates': int(agent.updates),
            'weight_seed': 92810 + seed_index * SEED_MIX,
            'training_seed_offset': seed_index * SEED_STRIDE,
        }
    report = {
        'status': 'PASS_EXECUTION',
        'seed_index': seed_index,
        'train_episodes_per_policy': train_episodes,
        'results': statuses,
        'protocol': {
            'algorithm': 'Double DQN with masked relocalization action',
            'high_level_observations': 'M7c: estimated goal (4), 16 zero ranges, 6 cov-or-zero, fix flag',
            'true_start': 'M7b: uncertainty-scaled random true pose around fixed estimated pose',
            'actions': ['short', 'wide', 'paid_localization_then_route'],
            'evaluation_seeds': f'low={EVAL_START}+j, high={EVAL_START+100000}+j, 0<=j<24',
            'decision_budget': 240,
            'hyperparameters': 'Same as original M7c',
            'train_schedule_seed': 11442 + seed_index * SEED_MIX,
        },
    }
    (RUN / 'm7d_train_result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def fresh_eval_seed(regime: str, j: int) -> int:
    return EVAL_START + (100000 if regime == 'high' else 0) + j


def evaluate_models(
    point_path: Path,
    belief_path: Path,
    label: str,
    eval_per_regime: int = EVAL_COUNT,
) -> dict:
    if eval_per_regime != EVAL_COUNT:
        raise ValueError('Evaluation count is frozen at 24 per regime')
    audit()
    torch.set_num_threads(1)
    sim = core.BotWSMuJoCoBackend(scene_path=core.SCENE)
    boxes = core.m2.scene_boxes(core.SCENE)
    models = {
        'point': core.MaskedMazeDQNAgent.load(point_path, device='cpu'),
        'belief': core.MaskedMazeDQNAgent.load(belief_path, device='cpu'),
    }
    records = []
    summary = {}
    for mode, obs_modes in (('point', ('live',)), ('belief', ('live', 'fixed', 'zero'))):
        for obs_mode in obs_modes:
            sub = []
            for regime in ('low', 'high'):
                for j in range(eval_per_regime):
                    seed = fresh_eval_seed(regime, j)
                    out = core.run_episode(
                        models[mode], sim, boxes, mode, regime, seed,
                        choice_rng=np.random.default_rng(3000 + j),
                        fix_rng=np.random.default_rng(6000 + j),
                        epsilon=0.0, training=False,
                        obs_mode=obs_mode,
                    )
                    out['policy_label'] = label
                    out['source_checkpoint'] = f'{mode}_{obs_mode}'
                    sub.append(out)
                    records.append(out)
            key = f'{mode}_{obs_mode}'
            summary[key] = core.stats(sub)
            print(label, key, summary[key]['all'], flush=True)
    write_csv(RUN / f'm7d_eval_{label}.csv', records)
    report = {
        'status': 'PASS_EXECUTION',
        'label': label,
        'test_seeds': 'fresh, common across all independent policies and fixed baselines',
        'results': summary,
        'note': 'Fixed-cov and zero-cov ablations are intervention probes, not separately trained policies.',
    }
    (RUN / f'm7d_eval_{label}.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def eval_fixed_baselines(eval_per_regime: int = EVAL_COUNT) -> dict:
    if eval_per_regime != EVAL_COUNT:
        raise ValueError('Evaluation count is frozen at 24 per regime')
    audit()
    sim = core.BotWSMuJoCoBackend(scene_path=core.SCENE)
    boxes = core.m2.scene_boxes(core.SCENE)
    records = []
    summary = {}
    for option in ('always_short', 'always_wide', 'fix_then_wide'):
        sub = []
        for regime in ('low', 'high'):
            for j in range(eval_per_regime):
                seed = fresh_eval_seed(regime, j)
                estimator, _ = core.reset_case(sim, boxes, 'point', regime, seed)
                fix = option.startswith('fix_then_')
                if fix:
                    core.ask_for_fix(sim, estimator, np.random.default_rng(6000 + j))
                route = 'short' if option == 'always_short' else 'wide'
                out = core.execute_option(sim, estimator, route, boxes)
                out.update(
                    mode='reference_fixed', regime=regime, seed=seed,
                    request=int(fix), first_action=2 if fix else (0 if route=='short' else 1),
                    route_action=0 if route == 'short' else 1,
                    net_return=out['reward'] - core.FIX_COST * int(fix),
                    initial_cov_trace=None, obs_mode=option,
                    policy_label='fixed_baseline',source_checkpoint=option,
                )
                records.append(out)
                sub.append(out)
        summary[option] = core.stats(sub)
        print('fixed baseline', option, summary[option]['all'], flush=True)
    write_csv(RUN / 'm7d_fixed_baselines.csv', records)
    report = {
        'status': 'PASS_EXECUTION',
        'test_seeds': 'same fresh seeds as all DQN evaluations',
        'results': summary,
        'precommitted_strong_baseline': 'always_wide',
        'note': 'The policy-always-wide baseline is executed directly, independently of any learned network.',
    }
    (RUN / 'm7d_fixed_baselines.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def smoke() -> dict:
    audit()
    sim = core.BotWSMuJoCoBackend(scene_path=core.SCENE)
    boxes = core.m2.scene_boxes(core.SCENE)
    cases = []
    for regime in ('low', 'high'):
        for mode in ('point', 'belief'):
            estimator, obs = core.reset_case(sim, boxes, mode, regime, seed=986000)
            assert obs.shape == (27,)
            assert np.all(np.isfinite(obs))
            assert np.array_equal(obs[4:20], np.zeros(16))
            assert (np.array_equal(obs[20:26],np.zeros(6))) == (mode=='point')
            cases.append({'mode':mode,'regime':regime,
                          'cov_trace':float(np.trace(estimator.filter.cov))})
    report = {'status':'PASS','cases':cases}
    (RUN / 'm7d_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print('M7d smoke PASS', flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',required=True,choices=('smoke','train','eval','fixed_baselines'))
    parser.add_argument('--seed-index',type=int,choices=(1,2))
    parser.add_argument('--point',type=Path)
    parser.add_argument('--belief',type=Path)
    parser.add_argument('--label',type=str)
    args = parser.parse_args()
    if args.mode == 'smoke':
        smoke()
    elif args.mode == 'train':
        if args.seed_index is None:
            parser.error('--seed-index is required for train')
        train(args.seed_index)
    elif args.mode == 'eval':
        if not args.point or not args.belief or not args.label:
            parser.error('--point, --belief, --label required for eval')
        evaluate_models(args.point,args.belief,args.label)
    else:
        eval_fixed_baselines()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
