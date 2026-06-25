# include/ — headers (the library)

Everything here is header-only, pure integer (Q8.8 fixed point), and depends only
on `<stdint.h>`/`<stdlib.h>`.

## Core

| Header | Role |
|--------|------|
| `rl_core.h` | Fixed-point type `rl_fp`, feature spec, the `rl_env` and **`rl_agent`** interfaces. Start here. |
| `rl_env.h` | Umbrella that includes the core, the harness, and all environments. |
| `rl_harness.h` | Generic training (`rl_train`) and evaluation (`rl_eval_many`) loops. |
| `rl_controller.h` | Plug-and-play wrapper for an async, init-once controller (e.g. a frequency tuner). |
| `qlearn.h` | The original standalone, malloc-free Q-learning agent (predates the `rl_agent` zoo). |

## Algorithms (`algo_*.h`)

Each implements the `rl_agent` vtable and is documented in `docs/rl/`:
`algo_qlearn.h` (reference), `algo_sarsa.h`, `algo_expsarsa.h`, `algo_doubleq.h`,
`algo_qlambda.h`, `algo_sarsalambda.h`, `algo_dynaq.h`, `algo_psweep.h`,
`algo_mc.h`, `algo_nstep.h`, `algo_actorcritic.h`.

## Environments (`env_*.h`)

`env_windy.h`, `env_arena.h`, `env_frozen.h`, `env_shift.h`, `env_swirl.h`,
`env_phone.h`. Each exposes `*_make()` / `reset` / `step` and a feature spec.

---

## Agent parameters explained

All numbers are **Q8.8 fixed point**: an integer `x` means the real value
`x / 256`. Helpers: `RL_INT(n)` = `n*256`, `RL_FRAC(a,b)` = `a*256/b`.

A `qlearn_agent_make(spec, n_features, n_actions, seed)` (and every `*_agent_make`)
takes:

- **`spec`** — an array of `rl_feature { lo, hi, bins }`, one per input. Each raw
  feature value in `[lo, hi]` is quantized into `bins` buckets; the agent's state
  is the combination of all buckets. Choosing `bins` is the key modelling
  decision: too few cannot tell situations apart, too many explode the table and
  slow learning. Total states = product of all `bins`.
- **`n_features`** — input dimensionality (length of `spec`).
- **`n_actions`** — output dimensionality (e.g. number of frequency levels).
- **`seed`** — PRNG seed (reproducibility; exploration tie-breaks).

Tunable hyperparameters (set in the constructor; the reference agent also exposes
`qlearn_agent_set(agent, alpha, gamma, eps)`):

| Parameter | Meaning | Typical | Guidance |
|-----------|---------|---------|----------|
| **alpha** (learning rate) | how strongly a new sample overrides the old estimate | `RL_FRAC(1,4)` = 0.25 | smaller = more stable/slower; constant (no decay) tracks non-stationary signals |
| **gamma** (discount) | weight on future vs immediate reward | `230` ≈ 0.90 | high for long-horizon mazes; **low (0–0.8) for near-bandit control like the frequency tuner** |
| **epsilon** (exploration) | probability (per-mille) of a random action | `100` = 10% | anneal high→low while training; keep tiny or guided on real hardware |

Algorithm-specific parameters live in their own headers and docs: `lambda`
(trace decay) for Q(λ)/SARSA(λ); `n` (lookahead) for n-step SARSA; planning steps
for Dyna-Q; priority threshold for Prioritized Sweeping; actor/critic step sizes
for Actor-Critic.

The online contract (`rl_agent.step`) and how to drive it are documented at the
top of `rl_core.h`.
