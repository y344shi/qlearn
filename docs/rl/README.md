# The Integer RL Algorithm Zoo & Contest

Eleven classic reinforcement-learning control algorithms, each implemented in
**pure-integer Q8.8 fixed point** behind one common interface
([`include/rl_core.h`](../../include/rl_core.h) → `rl_agent`), so they can be
trained and compared on identical tasks. No floating point appears in any
algorithm.

## The algorithms

| Algorithm | Header | Write-up | Family |
|-----------|--------|----------|--------|
| Q-learning (reference) | [`algo_qlearn.h`](../../include/algo_qlearn.h) | [q-learning.md](../q-learning.md) | off-policy TD |
| SARSA | [`algo_sarsa.h`](../../include/algo_sarsa.h) | [sarsa.md](sarsa.md) | on-policy TD |
| Expected SARSA | [`algo_expsarsa.h`](../../include/algo_expsarsa.h) | [expsarsa.md](expsarsa.md) | TD (expectation) |
| Double Q-learning | [`algo_doubleq.h`](../../include/algo_doubleq.h) | [doubleq.md](doubleq.md) | off-policy, unbiased |
| Watkins's Q(λ) | [`algo_qlambda.h`](../../include/algo_qlambda.h) | [qlambda.md](qlambda.md) | off-policy + traces |
| SARSA(λ) | [`algo_sarsalambda.h`](../../include/algo_sarsalambda.h) | [sarsalambda.md](sarsalambda.md) | on-policy + traces |
| Dyna-Q | [`algo_dynaq.h`](../../include/algo_dynaq.h) | [dynaq.md](dynaq.md) | model-based planning |
| Prioritized Sweeping | [`algo_psweep.h`](../../include/algo_psweep.h) | [psweep.md](psweep.md) | model-based planning |
| Monte-Carlo control | [`algo_mc.h`](../../include/algo_mc.h) | [mc.md](mc.md) | Monte-Carlo |
| n-step SARSA | [`algo_nstep.h`](../../include/algo_nstep.h) | [nstep.md](nstep.md) | n-step TD |
| One-step Actor-Critic | [`algo_actorcritic.h`](../../include/algo_actorcritic.h) | [actorcritic.md](actorcritic.md) | policy-based |

Each algorithm passes the two correctness tasks (`tests/test_<algo>.c`):
the **windy maze** (`env_windy.h`) and **frequency tuning** (`env_phone.h`).
Run them all: `make tests`.

## The contest arena

[`include/env_arena.h`](../../include/env_arena.h) is a **cliff-walk** designed
to *distinguish* the algorithms rather than merely be solved. With persistent
ε-greedy exploration on the deterministic cliff, the methods split exactly along
their theory (Sutton & Barto, Fig. 6.4):

```
=== CONTEST SCOREBOARD : deterministic cliff, persistent 10% exploration ===
  algorithm         online     eval  success    falls/2  steps2bar
   1. Actor-Critic      -11      -11     100%          0      10000
   2. Expected-SARSA    -15      -11     100%        370      10000
   3. SARSA(lambda)     -18      -13     100%        357      10000
   4. SARSA             -18      -13     100%        308      20000
   5. n-step-SARSA      -21      -13     100%        361      10000
   6. Q-learning        -29       -9     100%       2387      10000
   7. Dyna-Q            -29       -9     100%       2392      10000
   8. Prio-Sweeping     -30       -9     100%       2433      10000
   9. Q(lambda)         -31       -9     100%       2438      10000
  10. Double-Q          -32       -9     100%       2615      30000
  11. Monte-Carlo       -35      -13     100%        522      50000
```

How to read it — the discrimination is the point:
- **online** = average reward per episode *while still exploring*. **On-policy**
  methods (SARSA, SARSA(λ), Expected-SARSA, n-step, and the policy-based
  Actor-Critic) win here: they account for their own ε-exploration and learn a
  **safe path** away from the edge, so they rarely fall (`falls/2` ≈ 300).
- **eval** = greedy return after training. **Off-policy** methods (Q-learning,
  Double-Q, Q(λ)) and the **planners** (Dyna-Q, Prioritized Sweeping) win here
  (−9, the *optimal edge path*) — but pay for it with ~2400 cliff falls while
  exploring (worst **online**).
- **steps2bar** = sample efficiency (steps until greedy eval clears −30).
  **Monte-Carlo** is slowest (no bootstrapping, high variance).

This is the textbook cliff-walking result, reproduced with integer arithmetic.

## Harder environments & the grand tournament

Three escalating challenges ([`make tournament`](../../contest/tournament.c)),
each stressing a different weakness — all integer, all on the shared `rl_env`:

| Env | Header | Stresses |
|-----|--------|----------|
| Slippery cliff (5×10) | `env_arena.h` | robustness to action noise |
| Frozen lake (8×8, 12 deadly holes, slips) | [`env_frozen.h`](../../include/env_frozen.h) | hazardous exploration |
| Shifting maze (the gap moves mid-training) | [`env_shift.h`](../../include/env_shift.h) | adaptability (Dyna-Q+ blocking, S&B 8.3) |

Ranked by average rank across the three (lower = better):

```
   1. Dyna-Q          arena -18  frozen  -85  shift  -13   <- planning generalises best
   2. SARSA(lambda)   arena -24  frozen  -60  shift  -13
   ...
  11. Monte-Carlo     arena-199  frozen -364  shift -120   <- slow everywhere
```

The harder envs spread the field far more than the clean cliff: **Dyna-Q**
(model-based planning) is the most robust all-rounder; **eligibility-trace** and
**expected** methods generalise well; **Monte-Carlo** and as-tuned **SARSA** are
the most fragile under stochasticity. The shifting maze additionally separates
*adaptation speed* — Dyna-Q, traces and Q-learning recover the moved path fast,
while Double-Q, SARSA and Monte-Carlo lag.

## Swirl escape — the phone task as physics

[`env_swirl.h`](../../include/env_swirl.h) recasts the DVFS challenge physically:
unexpected **swirls** hit a craft, which must **thrust** hard to escape (≥ swirl
strength, like a frequency burst to beat a deadline) — but thrust burns a
**limited fuel** budget (power). The learned policy fires bursts to escape and
coasts when calm, beating the naive governors; the fuel budget is exactly what
sinks "always-max" (runs the tank dry) and "coast" (caught by everything).
`make swirl` → `results/swirl.svg`.

## Run it

```sh
make contest        # 11 agents on the cliff arena -> results/contest_*.svg
make tournament     # 11 agents across the 3 harder envs (arena/frozen/shift)
make swirl          # the fuel-limited swirl-escape physics demo -> swirl.svg
make tests          # every algorithm's windy-maze + frequency-tuning test
```

Demonstrable output:
- `results/contest_scores.svg` — ranked bar chart (coloured by family).
- `results/contest_curves.svg` — learning curves (eval return vs training step).
