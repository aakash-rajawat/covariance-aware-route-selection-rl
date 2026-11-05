#!/usr/bin/env python3
"""Build data-derived, static GitHub presentation assets for the frozen M7d release.

Only writes README.md, docs/PRESENTATION_RESULTS.md and figures/*.svg.
Does not change source, checkpoints, results, or original projects.
"""
from __future__ import annotations
import json
from pathlib import Path
from html import escape
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
GATE = json.loads((ROOT / 'src/m3b_gate.json').read_text())['precommitted_geometry']
POLICIES = {str(i): json.loads((ROOT / f'results/seed{i}/m7d_eval_seed{i}.json').read_text())['results'] for i in range(3)}
BASELINES = json.loads((ROOT / 'results/fixed_baselines/m7d_fixed_baselines.json').read_text())['results']
assert GATE['world_width_m'] == 3.2
for key in ('point_live','belief_live','belief_fixed','belief_zero'):
    assert all(POLICIES[s][key]['all']['n'] == 48 for s in POLICIES)
    assert all(POLICIES[s][key]['all'] == POLICIES['0'][key]['all'] for s in ('1','2'))
assert all(BASELINES[name]['all']['n'] == 48 for name in BASELINES)
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

NAVY = '#14263D'; MUTED = '#526476'; LINE = '#DEE5EC'; SKY = '#087EAB'; AMBER = '#C97717'; TEAL = '#087E77'; RED = '#AD4F4B'
def tx(x,y,content,size=16,weight=400,fill=NAVY,anchor='start'):
    return f'<text x="{x}" y="{y}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(str(content))}</text>'
def rect(x,y,w,h,fill,rx=0,stroke='none',sw=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
def line(x1,y1,x2,y2,color=LINE,sw=1,dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" stroke-dasharray="{dash}"/>'
def circle(x,y,r,fill,stroke='none',sw=1):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
def svg(file,w,h,parts):
    body='\n'.join(parts)
    content=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">\n{body}\n</svg>\n'
    ET.fromstring(content)
    (FIG/file).write_text(content,encoding='utf-8')

def maze():
    w,h=1050,620
    p=[rect(0,0,w,h,'#FFFFFF'),tx(38,52,'The navigation decision',28,700),tx(38,82,'M3b MuJoCo maze and reference waypoint options (top view)',15,400,MUTED)]
    x0,y0,scale=62,115,136
    side=3.2*scale
    toxy=lambda v:(round(x0+v[0]*scale,1),round(y0+(3.2-v[1])*scale,1))
    p.append(rect(x0,y0,side,side,'#F7FAFC',10,'#DCE5ED',2))
    # scene walls read from actual model
    world=ET.parse(ROOT/'src/mini_scene_m3b.xml').getroot().find('worldbody')
    for g in world.findall('geom'):
        if g.get('name') not in {'border_north','border_south','border_east','border_west','central_block'}: continue
        cx,cy,_=map(float,g.attrib['pos'].split()); sx,sy,_=map(float,g.attrib['size'].split())
        left,top=toxy((cx-sx,cy+sy))
        p.append(rect(left,top,round(2*sx*scale,1),round(2*sy*scale,1),'#667A8F',2))
    for name,color in [('wide',TEAL),('short',AMBER)]:
        route=GATE['routes'][name]
        points=' '.join(f'{a},{b}' for a,b in (toxy(v) for v in route))
        p.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>')
        for v in route[1:-1]:
            a,b=toxy(v);p.append(circle(a,b,4.5,'#FFFFFF',color,2))
    x,y=toxy(GATE['start'][:2]);p.append(circle(x,y,12,SKY,'#FFFFFF',3));p.append(tx(x-14,y-23,'Start',16,700))
    x,y=toxy(GATE['goal']);p.append(circle(x,y,13,'#FFFFFF',SKY,3));p.append(circle(x,y,6,SKY));p.append(tx(x+18,y-19,'Goal',16,700))
    p.extend([line(584,112,584,561,LINE,2),tx(623,158,'Two options, one goal',23,700),
        line(623,188,670,188,AMBER,7),tx(690,194,'Short / narrow',18,700),
        tx(690,220,'More direct, less tolerant of error',15,400,MUTED),
        line(623,280,670,280,TEAL,7),tx(690,286,'Wide / long',18,700),
        tx(690,312,'Longer detour with more clearance',15,400,MUTED),
        tx(623,382,'What the network decides',18,700),
        tx(623,409,'Choose short, wide, or request',15,400,MUTED),
        tx(623,432,'a paid localization correction.',15,400,MUTED),
        rect(613,474,389,73,'#F3F6F9',9),tx(629,501,'Route lines are reference waypoints,',14,400,MUTED),
        tx(629,525,'not measured robot trajectories.',14,400,MUTED)])
    svg('maze_routes.svg',w,h,p)

def compare():
    w,h=1120,630
    specs=[('Point DQN',POLICIES['0']['point_live']['all'],SKY),
           ('Belief DQN',POLICIES['0']['belief_live']['all'],TEAL),
           ('Always wide',BASELINES['always_wide']['all'],'#8192A2'),
           ('Fix, then wide',BASELINES['fix_then_wide']['all'],AMBER)]
    p=[rect(0,0,w,h,'#FFFFFF'),tx(40,52,'Performance on the shared evaluation scenarios',27,700),
       tx(40,81,'48 matched cases: 24 low uncertainty + 24 high uncertainty; identical evaluation seeds',15,400,MUTED)]
    p.extend([tx(45,137,'Navigation success',21,700),tx(575,137,'Mean episode return',21,700)])
    # vertical split separator
    p.append(line(558,116,558,542,'#DDE5EB',2))
    for idx,(name,v,color) in enumerate(specs):
        y=196+idx*89
        p.extend([tx(45,y-14,name,16,700),rect(45,y,375,22,'#EDF2F6',6),
          rect(45,y,round(375*v['success']/48,1),22,color,6),
          tx(433,y+18,f"{v['success']}/48 ({100*v['success']/48+1e-9:.1f}%)",15,700)])
        # min -20, max +20 displayed from x650 through x1017; axis crossing 0
        left,right=650,1017
        zero=left+(right-left)/2
        reward=float(v['mean_return']); end=zero+reward/20*((right-left)/2)
        p.extend([tx(582,y-14,name,16,700),line(zero,y-4,zero,y+27,'#CDD9E2',2),
                  rect(min(zero,end),y,abs(end-zero),22,color,5),
                  tx(min(1080,max(590,end+12)),y+18,f'{reward:+.2f}',15,700)])
    p.extend([line(45,549,1080,549,LINE,1),
      tx(45,580,'Belief learns covariance-dependent routes, but fix-then-wide is the strongest tested strategy.',15,700),
      tx(45,605,'Same 48 cases are reused across all 3 training seeds; these are not 144 independent test scenarios.',13,400,MUTED)])
    svg('performance.svg',w,h,p)

def route_choices():
    w,h=1030,430
    p=[rect(0,0,w,h,'#FFFFFF'),tx(35,50,'Learned route choice at the two tested uncertainty levels',24,700),
       tx(35,78,'Greedy evaluation on 48 common held-out cases; all 3 trained seeds agree',14,400,MUTED)]
    # draw conditional rule as two rows
    p.extend([tx(47,146,'Policy',17,700),tx(433,146,'Low covariance',17,700,anchor='middle'),tx(822,146,'High covariance',17,700,anchor='middle')])
    rows=[('Point DQN', ['SHORT','SHORT']),('Belief DQN',['SHORT','WIDE']),('Belief + zero covariance',['SHORT','SHORT']),('Belief + fixed covariance',['WIDE','WIDE'])]
    for k,(name,opts) in enumerate(rows):
        y=174+k*51
        p.append(rect(32,y-15,962,46,('#F7FAFC' if k%2==0 else '#FFFFFF'),7))
        p.append(tx(47,y+13,name,16,700 if k<2 else 400))
        for j,opt in enumerate(opts):
            x=355 if j==0 else 742
            color=AMBER if opt=='SHORT' else TEAL
            p.extend([rect(x,y-5,158,32,color,9),tx(x+79,y+17,opt,15,700,'#FFFFFF','middle')])
    p.extend([line(33,390,994,390,LINE),tx(43,416,'No learned policy selected the paid localization action in these final evaluations.',13,400,MUTED)])
    svg('route_choices.svg',w,h,p)

maze(); compare();route_choices()
P=POLICIES['0']; D=BASELINES
fmt=lambda d:f"{d['success']}/48 ({d['success']/48*100+1e-9:.1f}%)"
rows=[('Point DQN',P['point_live']['all']),('Belief DQN',P['belief_live']['all']),('Always short',D['always_short']['all']),('Always wide',D['always_wide']['all']),('Relocalize, then wide',D['fix_then_wide']['all'])]
tab='\n'.join(f"| {name} | {fmt(d)} | {d['mean_return']:.3f} |" for name,d in rows)
readme=f'''# Covariance-Aware Route Selection with Deep Reinforcement Learning

**Can localization uncertainty change the navigation decisions learned by a robot?**

A controlled **MuJoCo proof of concept** comparing two high-level Double DQN policies: **Point** (estimated goal, no covariance) and **Belief** (same information plus EKF pose covariance). Both control the same differential-drive robot through the same waypoint-following controller; **wheel commands are not learned**.

![M3b maze with short and wide route options](figures/maze_routes.svg)

## Results at a glance

Three independently trained Point/Belief pairs converged to the same tested decision rule on **48 shared held-out scenarios** (24 low- and 24 high-uncertainty cases). These are **48 distinct scenarios, not 144 independent test situations**.

![Navigation success and reward comparison](figures/performance.svg)

| Strategy | Successful episodes | Mean episode return |
| --- | ---: | ---: |
{tab}

**Main finding:** Belief learned to choose the **short route at low uncertainty** and the **wide route at high uncertainty**. Point chose the short route in both regimes. Belief achieved {fmt(P['belief_live']['all'])} versus {fmt(P['point_live']['all'])} for Point.

**Essential qualification:** Always-wide matched Belief's number of successes (39/48), with a slightly lower mean return ({D['always_wide']['all']['mean_return']:.3f} versus {P['belief_live']['all']['mean_return']:.3f}). **Relocalize-then-wide outperformed both learned policies** ({fmt(D['fix_then_wide']['all'])}, mean return {D['fix_then_wide']['all']['mean_return']:.3f}). This study demonstrates covariance-dependent learning, **not optimal navigation**.

## How it works

```mermaid
flowchart LR
    E["EKF pose estimate + goal"] --> P["Point Double DQN"]
    E --> B["Belief Double DQN"]
    C["EKF pose covariance"] --> B
    P --> A["High-level route decision"]
    B --> A
    A --> R["Shared EKF-based waypoint controller"]
    R --> M["MuJoCo differential-drive robot"]
    M --> F["Outcome reward: goal / collision / time"]
```

The high-level action set is **short route**, **wide route**, or **paid localization update followed by route selection**. Both networks have the same 27-input architecture: 4 goal-relative features, 16 range-input slots deliberately set to zero for both networks, 6 covariance-input slots (zero for Point), and 1 localization-available flag. The only intended information difference at the first decision is covariance. That restriction is deliberate; this experiment does **not** show that covariance beats an agent using all raw sensors.

### Learned decisions and covariance ablation

![Route choices for Point, Belief and covariance ablations](figures/route_choices.svg)

- **Training:** 240 episodes per policy per seed, with two added independent replications (6 trained models in total).
- **Evaluation:** same 48 fresh simulation scenarios for each model, seed-matched to the fixed-strategy baselines.
- **Covariance ablation:** at zero covariance, Belief falls back to short; with fixed covariance, it chooses wide, confirming the input changes its decision.
- **Active localization:** no trained policy chose the paid localization action in the final evaluations, even though it was available and the fixed relocalize-then-wide baseline was stronger.

## Repository contents

```text
src/            MuJoCo scene, EKF, shared controller, DQN and evaluation code
checkpoints/    Point and Belief policy weights for seeds 0, 1 and 2
results/        Original matched evaluation CSV/JSON and training traces
figures/        Diagrams generated from the checked-in geometry and results
scripts/        Smoke/full evaluation and figure-generation scripts
docs/           Limitations, provenance and results interpretation
```

## Verify locally

```bash
sha256sum -c SHA256SUMS.txt
bash scripts/re_evaluate.sh smoke
# Full evaluation on the saved models, no retraining:
bash scripts/re_evaluate.sh full
# Rebuild SVG figures from saved data (standard library only):
python3 scripts/build_presentation.py
```

**Environment note:** The existing reproduction launcher currently defaults to `~/belief-aware-robot-navigation/.venv/bin/python` or accepts `M7_PYTHON=/path/to/python`. A clean-clone installation and fully pinned dependencies have **not yet been validated**; see the pending publication audit. The figure generator itself uses only Python's standard library.

## Scope and limitations

This is **high-level option-selection RL**, not end-to-end navigation, autonomous route planning, or learned motor control. It uses one small maze, two prescribed uncertainty regimes, a shared fixed low-level controller, and masked high-level range inputs. EKF covariance calibration is imperfect; intermediate uncertainty levels, new maps, Gazebo transfer, and generalization are **not established**. The covariance comes from this robot's simulated EKF, not an analytical optical-tracking covariance model.

See [results interpretation](docs/PRESENTATION_RESULTS.md), [limitations](docs/LIMITATIONS.md), and [source provenance](docs/PROVENANCE.md). **No license is assigned yet pending source-provenance review.**
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
(ROOT/'docs/PRESENTATION_RESULTS.md').write_text('''# M7d results: interpretation and boundaries

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
''',encoding='utf-8')
print('PRESENTATION PASS: README, 3 data-derived SVG figures, results interpretation')
for f in sorted(FIG.glob('*.svg')):print(f.relative_to(ROOT),f.stat().st_size,'bytes')
