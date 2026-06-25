# qlearn_int_minimal

Pure-integer (Q8.8 fixed-point) reinforcement learning in C. The project provides
an environment-agnostic agent interface, eleven classic tabular RL algorithms,
several environments (gridworlds, a CPU-frequency tuner, a partially-observed
control task), a contest/benchmark harness, and a migration path into a real
frequency tuner. No floating-point arithmetic is used anywhere.

## Build

This machine has no `cc` or `make`; `gcc-12` is installed. Set `CC` accordingly:

    CC=gcc-12 ./build.sh        # builds the core demonstrations

Each demonstration is also a single self-contained compilation (see below), so
`make` is optional. A `Makefile` is provided for environments that have `make`.

## Running

Interactive launcher (prompts for: method, environment, agent, grid size):

    ./play.sh

Equivalent direct commands (copy-paste; no menu required):

    # Gridworld demonstration: rows cols mines seed
    gcc-12 -O2 -std=c99 -Iinclude -o cliff src/cliff_wind_qlearn.c
    ./cliff 8 12 14 7

    # CPU-frequency tuner demonstration
    gcc-12 -O2 -std=c99 -Iinclude -o phone_sim src/phone_sim.c
    ./phone_sim

    # Any single agent, or `all` (ranked), on any environment
    gcc-12 -O2 -std=c99 -Iinclude -o runner contest/runner.c contest/reg_*.c
    ./runner swirl all
    ./runner windy dynaq 10 10 18

    # Eleven-algorithm contest on the slippery-cliff arena
    gcc-12 -O2 -std=c99 -Iinclude -o contest_run contest/contest.c contest/reg_*.c
    ./contest_run

    # Path visualizer: each agent's learned path versus the BFS-optimal path
    gcc-12 -O2 -std=c99 -Iinclude -o pathviz contest/pathviz.c contest/reg_*.c contest/qval_*.c
    ./pathviz 6 9 6
    python3 tools/plot_svg.py results/metrics.csv results

    # Kernel-migration simulator: every agent behind one init/select/destroy vtable
    gcc-12 -O2 -std=c99 -Iinclude -o dfc_sim tunner/dfc_qlearn_sim.c contest/reg_*.c
    ./dfc_sim            # rank all agents
    ./dfc_sim sweep      # hyperparameter sweep -> Pareto front of jank vs power

`runner` environments: `windy`, `arena`, `frozen`, `shift`, `swirl`, `phone`.
Plots are written to `results/*.svg` (pure-stdlib SVG). Example outputs are
committed under `docs/img/` for reference.

## Algorithms

Every algorithm is a single self-contained header implementing the common agent
interface in `include/rl_core.h`, in pure integer arithmetic. Each has an
accompanying explanation document. Start from the interface, then read any
implementation alongside its document.

| Algorithm | Implementation | Explanation |
|-----------|----------------|-------------|
| Q-learning (reference) | [include/algo_qlearn.h](include/algo_qlearn.h) | [docs/q-learning.md](docs/q-learning.md) |
| SARSA | [include/algo_sarsa.h](include/algo_sarsa.h) | [docs/rl/sarsa.md](docs/rl/sarsa.md) |
| Expected SARSA | [include/algo_expsarsa.h](include/algo_expsarsa.h) | [docs/rl/expsarsa.md](docs/rl/expsarsa.md) |
| Double Q-learning | [include/algo_doubleq.h](include/algo_doubleq.h) | [docs/rl/doubleq.md](docs/rl/doubleq.md) |
| Watkins's Q(lambda) | [include/algo_qlambda.h](include/algo_qlambda.h) | [docs/rl/qlambda.md](docs/rl/qlambda.md) |
| SARSA(lambda) | [include/algo_sarsalambda.h](include/algo_sarsalambda.h) | [docs/rl/sarsalambda.md](docs/rl/sarsalambda.md) |
| Dyna-Q | [include/algo_dynaq.h](include/algo_dynaq.h) | [docs/rl/dynaq.md](docs/rl/dynaq.md) |
| Prioritized Sweeping | [include/algo_psweep.h](include/algo_psweep.h) | [docs/rl/psweep.md](docs/rl/psweep.md) |
| Monte-Carlo control | [include/algo_mc.h](include/algo_mc.h) | [docs/rl/mc.md](docs/rl/mc.md) |
| n-step SARSA | [include/algo_nstep.h](include/algo_nstep.h) | [docs/rl/nstep.md](docs/rl/nstep.md) |
| One-step Actor-Critic | [include/algo_actorcritic.h](include/algo_actorcritic.h) | [docs/rl/actorcritic.md](docs/rl/actorcritic.md) |

Overview of the algorithm family, the contest, and the harder environments:
[docs/rl/README.md](docs/rl/README.md). Theory of the base method:
[docs/q-learning.md](docs/q-learning.md).

## Environments

| Environment | Source | Description |
|-------------|--------|-------------|
| Windy maze | [include/env_windy.h](include/env_windy.h) | gridworld with mines and per-column wind |
| Cliff arena | [include/env_arena.h](include/env_arena.h) | slippery cliff; the contest discriminator |
| Frozen lake | [include/env_frozen.h](include/env_frozen.h) | slippery grid with deadly holes |
| Shifting maze | [include/env_shift.h](include/env_shift.h) | non-stationary; the gap moves mid-training |
| Swirl escape | [include/env_swirl.h](include/env_swirl.h) | partially observed control task |
| Phone tuner | [include/env_phone.h](include/env_phone.h) | simulated DVFS frequency control |

## Agent interface

All algorithms expose the same vtable (`rl_agent` in `include/rl_core.h`):

    rl_agent ag = qlearn_agent_make(spec, n_features, n_actions, seed);   /* construct */
    int action  = ag.step(&ag, reward_prev, features, done, explore);     /* learn + act */
    int action  = ag.act_greedy(&ag, features);                           /* act only */
    ag.destroy(&ag);

A feature vector of any dimensionality is quantized into a Q-table; memory is
caller-owned; there is no floating point.

## Migration into the frequency tuner

The agents are intended to drop into `dfc_tuner_qlearn_misc.c`. They are exposed
through a three-pointer vtable — initialize, select a frequency from a feature
struct, destroy — so any algorithm can be swapped without changing the tuner:

    freq_agent fa = freq_agent_make("dynaq", reg_dynaq, &prop);   /* init */
    unsigned int khz = fa.select(&fa, &features);                 /* features -> frequency */
    fa.destroy(&fa);                                              /* destroy */

Features used: `avg_load`, `curr_refresh_rate`, `curr_power`, `frame_budget` (a
subset of `struct __sched_ind_qlearn_features`). Relevant files:

- [tunner/dfc_qlearn_sim.c](tunner/dfc_qlearn_sim.c) — kernel-like data structures and a
  simulated tuner that swaps in and ranks every agent.
- [tunner/tuner_plug_example.c](tunner/tuner_plug_example.c) — the init-once / per-event
  controller lifecycle (`include/rl_controller.h`).
- [tunner/environment_description.md](tunner/environment_description.md) — mapping to the
  real kernel interface.

## Repository layout

    include/    rl_core.h (agent interface), rl_env.h (umbrella), algo_*.h (11 agents),
                env_*.h (6 environments), rl_controller.h, rl_harness.h, qlearn.h
    src/        cliff_wind_qlearn.c, phone_sim.c, swirl_sim.c, *_gpu.cu
    contest/    contest.c, tournament.c, runner.c, pathviz.c, reg_*.c, qval_*.c
    tunner/     dfc_qlearn_sim.c, *_example.c, environment_description.md
    tools/      plot_svg.py (pure-stdlib SVG)
    docs/       q-learning.md, rl/*.md, img/ (example outputs)

## Notes

- Pure integer (Q8.8). Verified: no `float` or `double` in any algorithm.
- Toolchain on this machine: WSL2 with `gcc-12`; CUDA is optional and uses
  `nvcc -ccbin g++-12`.
