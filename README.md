# qlearn_int_minimal

**Pure-integer reinforcement learning in C** — a reusable, environment-agnostic
agent (Q8.8 fixed point, **no floating point anywhere**), eleven classic RL
algorithms behind one interface, several environments (gridworlds, a phone
CPU-frequency tuner, a physics "swirl-escape" task), a discriminating contest,
and a path visualizer. Designed to drop straight into an embedded/kernel
frequency tuner.

---

## Quick start

```sh
./play.sh            # interactive launcher (menus; recommended)
```
The launcher asks, in order: **demonstration method → test/environment →
agent(s) → grid size** (3–4 prompts) and runs it.

Or run pieces directly (no `make` needed):

```sh
./build.sh                       # builds the core demos
./run.sh 8 12 14 7               # gridworld (rows cols mines seed) + SVG plots
```

> **Toolchain note (this machine = WSL2).** There is no plain `gcc`/`make`, but
> `gcc-12` is present. Every command below works by substituting `gcc-12` for
> `cc`. A `Makefile` is provided for environments that have `make`. CUDA needs
> `nvcc -ccbin g++-12` (see GPU section).

---

## What you can run (options)

### 1. Interactive launcher — `./play.sh`
| Prompt | Choices |
|--------|---------|
| Demonstration method | Scoreboard/ranking (console) · Plot (SVG) · Path visualizer (SVG) |
| Test / environment | windy maze · cliff arena · frozen lake · shifting maze · swirl escape · phone tuner · contest |
| Agent(s) | `all` (rank everyone) or any one of the 11 |
| Grid size | small / medium / large / custom |

### 2. `make` targets (or the equivalent direct command shown earlier)
| Target | What it does |
|--------|--------------|
| `make run` | gridworld demo → `results/*.svg` |
| `make plots` | gridworld + regenerate plots |
| `make phone` | phone DVFS tuner demo → `results/phone.svg` |
| `make swirl` | swirl-escape (partially observed) → `results/swirl.svg` |
| `make contest` | all 11 agents on the cliff arena → `results/contest_*.svg` |
| `make tournament` | all 11 across the 3 harder envs (arena/frozen/shift) |
| `make viz` | **path visualizer**: every agent's path vs the BFS-optimal |
| `make tests` | every algorithm's windy-maze + frequency-tuning unit test |
| `make agent` | the feature-vector agent example |
| `make plug` | the plug-and-play async-controller (tuner integration) example |
| `make run-gpu` | CUDA ensemble of agents (needs an NVIDIA GPU) |

### 3. Flexible runner — `runner <env> <agent|all> [sizes]`
```sh
gcc-12 -O2 -std=c99 -Iinclude -o runner contest/runner.c contest/reg_*.c
./runner swirl all                 # rank all 11 agents on the swirl task
./runner windy dynaq 10 10 18      # one agent on a 10x10 maze with 18 mines
```
`env` ∈ `windy | arena | frozen | shift | swirl | phone`.

---

## Sample expected results

**Gridworld** (`./cliff 8 12 14 7`) — reward climbs to the BFS optimum and holds:
```
  episode   steps   reward   reached_goal
  0         120     -120     no
  1280      18      -18      yes
  7679      18      -18      yes        <- optimal = 18 steps
```

**Phone DVFS tuner** (`make phone`) — matches the oracle's jank while saving power:
```
  policy          jank%     energy    avgMHz
  performance        8.7       100%     2620
  powersave         92.6        58%     1239
  qlearn             8.7        90%     2328  <- learned
  optimal            9.2        82%     2076  <- oracle
```

**Swirl escape, partially observed** (`make swirl`) — beats the reactive oracle
by learning a hedge against the noisy sensor:
```
  policy               reward  caught%
  match-reading          -533      26%  (reactive ORACLE)
  qlearn (learned)       -242       0%  <- learned
  omniscient             -191       0%  (full-info bound)
```

**Contest** (`make contest`) — reproduces Sutton & Barto Fig. 6.4 with integers:
```
   1. Actor-Critic   online -11  eval -11  falls    0
   ...
   6. Q-learning     online -29  eval  -9  falls 2387   <- off-policy: optimal but risky
  11. Monte-Carlo    online -35  eval -13  falls  522   <- slowest
```

**All agents on swirl** (`./runner swirl all`) — best are the one-step value methods:
```
   1. qlearn       reward=-242 caught=0%   (oracle -533, omni -191)
   1. (tie) sarsa, doubleq, qlambda, dynaq  reward=-242 caught=0%
  10. mc           reward=-447 caught=20%
  11. actorcritic  reward=-533 caught=26%   (only matched the reading)
```

**Path visualizer** (`make viz`) — trains all 11, draws each learned path vs the
BFS-optimal, plus reward and Bellman-δ curves:
```
  qlearn       path=14 steps reached=yes
  ...
  actorcritic  path=16 steps reached=yes
  Saved: results/viz_paths.svg, viz_rewards.svg, viz_delta.svg  (optimal=14 steps)
```

All plots are pure-stdlib SVG (no matplotlib/gnuplot). Open any `results/*.svg`
in a browser or IDE.

---

## The agents & the tuner API

Every algorithm implements one vtable (`include/rl_core.h` → `rl_agent`), all
integer-only, caller-owned memory:

```c
rl_agent ag = qlearn_agent_make(spec, n_features, n_actions, seed);  // init once
int action  = ag.step(&ag, reward_prev, features, done, explore);    // learn + act
int action  = ag.act_greedy(&ag, features);                          // act only
```

**Plug-and-play into an async, init-once controller** (the frequency tuner) —
`include/rl_controller.h`:
```c
rl_controller_init(&ctrl, &agent, my_reward_fn, user);     // qlm_init / qlm_start  (once)
int action = rl_controller_tick(&ctrl, features);          // handle_load_change   (each event)
```
See `tunner/tuner_plug_example.c` (`make plug`) for the full lifecycle, and
`tunner/environment_description.md` for the real `dfc_tuner_qlearn_misc.c` mapping.

**The 11 algorithms** (each `include/algo_*.h` + `docs/rl/*.md` write-up):
Q-learning · SARSA · Expected SARSA · Double-Q · Q(λ) · SARSA(λ) · Dyna-Q ·
Prioritized Sweeping · Monte-Carlo · n-step SARSA · Actor-Critic.
See **[`docs/rl/README.md`](docs/rl/README.md)** for the zoo, the contest, and
the harder environments. Concepts: **[`docs/q-learning.md`](docs/q-learning.md)**.

---

## Repository layout

```
include/        rl_core.h (interface) · rl_env.h (umbrella) · algo_*.h (11 agents)
                env_*.h (windy/arena/frozen/shift/swirl/phone) · rl_controller.h · qlearn.h
src/            cliff_wind_qlearn.c (gridworld) · phone_sim.c · swirl_sim.c · *_gpu.cu
contest/        contest.c · tournament.c · runner.c · pathviz.c · reg_*.c · qval_*.c
tunner/         qlearn_tunner.c (real tuner) · *_example.c · environment_description.md
tools/          plot_svg.py (pure-stdlib SVG)
docs/           q-learning.md · rl/*.md
play.sh build.sh run.sh Makefile
```

Everything is integer-only and dependency-free (libc + optional `python3` for
SVGs). Verified: no `float`/`double` in any algorithm.
