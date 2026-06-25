# tunner/ — frequency-tuner integration

How the integer agents migrate into the real CPU-frequency tuner
(`dfc_tuner_qlearn_misc.c`).

| File | Purpose |
|------|---------|
| `qlearn_tunner.c` | The real (kernel) tuner. Currently a heuristic; now also emits a trace line per event for offline tuning. Uses kernel headers (does **not** build on a host). |
| `dfc_qlearn_sim.c` | Host-buildable simulator. Wraps every agent behind one `init/select/destroy` vtable and runs them through a simulated tuner with kernel-like structs. |
| `tuner_plug_example.c` | The init-once / per-event controller lifecycle with a reward function (`rl_controller.h`). |
| `qlearn_agent_example.c` | Minimal feature-vector usage of a single agent. |
| `environment_description.md` | The real `__sched_ind_qlearn_features` / `dfc_prop` interface and reward design. |

## Features used

`avg_load`, `curr_refresh_rate`, `curr_power`, `frame_budget` — a subset of
`struct __sched_ind_qlearn_features`.

## Run the simulator and the sweep

    gcc-12 -O2 -std=c99 -Iinclude -o dfc_sim tunner/dfc_qlearn_sim.c contest/reg_*.c
    ./dfc_sim            # rank all 11 agents behind the swappable vtable
    ./dfc_sim sweep      # hyperparameter sweep -> Pareto front of jank vs power

## Harvest real traces for offline tuning

`qlearn_tunner.c` logs one line per event, e.g.

    dfs load=512 rr=2 power=430 fbud=-120 jank=1843 freq=2150000

Collect and convert on device:

    dmesg | grep ' dfs ' > qltrace.log

These traces are the input for calibrating `dfc_qlearn_sim.c` and running the
sweep on real workloads (low gamma + moderate alpha were best in simulation).
