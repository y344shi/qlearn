# n-step SARSA: Bridging TD and Monte Carlo (Integer-Only)

This note explains **n-step SARSA**, the on-policy multi-step generalisation of
SARSA, in the same notation as [`../q-learning.md`](../q-learning.md) and its
one-step sibling [`sarsa.md`](sarsa.md). It then shows the integer-only (Q8.8
fixed-point) implementation used in this contest
([`../../include/algo_nstep.h`](../../include/algo_nstep.h)). If one-step SARSA
bootstraps after a *single* reward, n-step SARSA waits for *n* real rewards
before bootstrapping — and that one knob slides the algorithm continuously
between one-step TD and Monte Carlo.

---

## 1. The idea: look further before you bootstrap

A one-step TD method (SARSA, Q-learning) estimates the return after a single
transition by *bootstrapping*: it replaces the unknown tail of the return with
its own current estimate of the next state-action value. Monte-Carlo (MC) does
the opposite — it waits until the episode ends and uses the **actual** observed
return, never bootstrapping. Both are extremes of one spectrum:

- one-step TD uses **1** real reward then bootstraps — low variance, but biased
  by whatever error currently sits in the value table;
- MC uses **all** real rewards and never bootstraps — unbiased, but every
  reward along a long, noisy path adds variance.

**n-step methods** (Sutton & Barto, Chapter 7) sit *between* these two: use the
next **n** real rewards, then bootstrap off the value n steps ahead. $n$ is the
dial; $n=1$ is one-step TD, $n=\infty$ is Monte Carlo, and intermediate $n$
usually beats *both* endpoints.

---

## 2. The n-step return

Following the MDP setup of the Q-learning note (states $\mathcal{S}$, actions
$\mathcal{A}$, rewards, discount $\gamma\in[0,1)$, return
$G_t=\sum_{k\ge0}\gamma^kR_{t+k+1}$), define the **n-step return** as the first
$n$ discounted rewards plus a bootstrap off the estimated value $n$ steps later.
For the *action-value* (control) case this is:

$$
\boxed{\;
G_{t:t+n} \;=\; R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1}R_{t+n}
            \;+\; \gamma^{n}\,Q(S_{t+n}, A_{t+n})
\;}
$$

i.e. $G_{t:t+n} = \sum_{i=0}^{n-1}\gamma^{i}R_{t+i+1} + \gamma^{n}Q(S_{t+n},A_{t+n})$,
for $n\ge 1$ and $t+n$ before the terminal step. If $t+n$ reaches or passes the
terminal step, the bootstrap term vanishes and $G_{t:t+n}=G_t$ — the **full**
Monte-Carlo return. Two sanity checks:

- $n=1$: $G_{t:t+1} = R_{t+1} + \gamma Q(S_{t+1},A_{t+1})$ — exactly the
  one-step SARSA target.
- $n\to\infty$ (or $t+n\ge T$): $G_{t:t+n}=R_{t+1}+\gamma R_{t+2}+\dots$ — the
  MC return, no bootstrap.

Because it is **on-policy**, the bootstrap value $Q(S_{t+n},A_{t+n})$ is read at
the action $A_{t+n}$ the behaviour policy **actually took** — not a `max` (that
would be the off-policy / n-step tree-backup variant).

---

## 3. The n-step SARSA update

n-step SARSA nudges the value of the state-action pair *n steps in the past*
toward the n-step return:

$$
\boxed{\;
Q(S_\tau, A_\tau) \;\leftarrow\; Q(S_\tau, A_\tau)
   \;+\; \alpha\,\big[\, G_{\tau:\tau+n} - Q(S_\tau, A_\tau) \,\big],
\qquad \tau = t-n+1
\;}
$$

The index $\tau=t-n+1$ is the **bookkeeping** core of every n-step method: at
the moment we *finish* observing $R_{t+1}$ (so we hold rewards
$R_{\tau+1}\dots R_{\tau+n}$ and the state-action pair $S_{t+n},A_{t+n}$… in
practice $S_{t},A_{t}$ at the current step), the pair whose value we can finally
update is the one taken at time $\tau$. The bracket
$\delta = G_{\tau:\tau+n} - Q(S_\tau,A_\tau)$ is the **n-step TD error**.

The algorithm makes **no updates during the first $n-1$ steps** of an episode
(no pair is yet $n$ steps old). To compensate, when the episode terminates it
performs an **equal number of catch-up updates** for the still-buffered tail —
each a shrinking-horizon MC return with no bootstrap. This flush is what makes
the boundary case smoothly become Monte Carlo.

### Error-reduction property

The reason intermediate $n$ helps is the **error-reduction property** (Sutton &
Barto §7.1): the worst-case error of the *expected* n-step return is at most
$\gamma^{n}$ times the worst-case error of the current value estimate. Looking
$n$ steps ahead shrinks the bootstrap's contribution to the target by
$\gamma^{n}$, so more of the target is grounded in real reward — provably
tightening the estimate, which is why n-step prediction converges for all $n$.

---

## 4. Bias / variance — the spectrum made concrete

| $n$ | rewards used | bootstrap | bias | variance | credit reach / update |
|-----|--------------|-----------|------|----------|-----------------------|
| $1$ (TD) | 1 | yes | high (whole tail is an estimate) | low | 1 cell |
| middle | $n$ | yes | moderate | moderate | $n$ cells |
| $\infty$ (MC) | all | no | none (unbiased) | high | whole path |

- **Bias** comes from the bootstrap: replacing the tail with a possibly-wrong
  $Q$. More real rewards (larger $n$) → less bias.
- **Variance** comes from summing many random rewards/transitions. Fewer real
  rewards (smaller $n$) → less variance.
- **Credit propagation speed.** A one-step method moves reward information back
  exactly **one** cell per visit, so on a long path it must replay the path many
  times for credit to crawl to the start. n-step moves it back **n** cells per
  update — the practical reason this repo picks $n>1$ for the long, sparse
  windy maze (see §7).

The sweet spot is task-dependent and empirical; n-step is the tool that lets you
*tune* the trade-off instead of being stuck at an endpoint.

---

## 5. Pseudocode (Sutton & Barto §7.2) and the tau bookkeeping

```
Initialize Q(s,a) for all s,a   (optimistically — see §7)
Algorithm parameters: step size α∈(0,1], small ε, integer n
for each episode:
    Initialize and store S_0 ; choose A_0 ~ ε-greedy(·|S_0) ; store A_0
    T ← ∞
    for t = 0, 1, 2, ... :
        if t < T:
            take A_t ; observe and store R_{t+1}, S_{t+1}
            if S_{t+1} terminal: T ← t+1
            else: choose A_{t+1} ~ ε-greedy(·|S_{t+1}) ; store A_{t+1}
        τ ← t − n + 1                          # pair whose value we update now
        if τ ≥ 0:
            G ← Σ_{i=τ+1}^{min(τ+n,T)} γ^{i−τ−1} R_i          # n real rewards
            if τ+n < T: G ← G + γ^n · Q(S_{τ+n}, A_{τ+n})     # bootstrap (skip if terminal)
            Q(S_τ,A_τ) ← Q(S_τ,A_τ) + α [ G − Q(S_τ,A_τ) ]
    until τ = T − 1                            # flush the tail (n−1 catch-up updates)
```

`A_{τ+n}` is the action *actually selected* — the on-policy bootstrap. The
`if τ+n < T` guard is the single line that turns the tail updates into
bootstrap-free Monte-Carlo returns.

### Fitting the one-call online protocol — a circular buffer

This contest drives every agent through one
`step(reward_prev, feat, done, explore)` call (see
[`rl_core.h`](../../include/rl_core.h)); there is no explicit per-episode loop,
so the $\tau$ bookkeeping becomes a small **circular buffer** of the last $n{+}1$
entries `(state, action, reward-following)`. Each `step()`:

1. Compute $s' =$ `rl_state_of(...)` for the incoming `feat`.
2. **Attach** `reward_prev` to the newest still-pending buffered entry (that is
   the reward $R$ that *followed* the previously chosen action).
3. If `done`: **flush** — repeatedly emit the oldest buffered entry with a
   bootstrap-free n-step return until the buffer is empty (the catch-up
   updates), clear it, return `-1`.
4. Otherwise choose $a'$ on-policy ($\varepsilon$-greedy, or greedy here) for
   $s'$, **push** `(s', a', reward-TBD)` as the newest entry.
5. If the buffer now holds $n{+}1$ entries, the oldest is exactly $n$ steps back
   ($\tau = t-n+1$): **emit** its full n-step bootstrapped update (bootstrapping
   off the just-pushed $Q(s',a')$) and pop it.
6. Return $a'$.

`act_greedy()` is pure $\arg\max$, no learning, used only for evaluation.

---

## 6. Strengths and weaknesses

**Strengths**
- **Faster credit assignment** than one-step TD on long / sparse-reward tasks —
  reward reaches distant predecessors $n\times$ sooner.
- **Tunable bias/variance** via a single integer $n$; intermediate $n$ commonly
  outperforms both one-step TD and MC.
- Inherits SARSA's **on-policy safety**: the bootstrap reflects the exploratory
  policy actually run, so learned values account for exploration risk.
- Still **online and incremental** — no need to store whole episodes (unlike raw
  MC); only an $n{+}1$ entry buffer.

**Weaknesses**
- A **latency of $n-1$ steps** before the first update, plus an explicit tail
  flush at episode end (extra bookkeeping vs one-step).
- More **memory** (the buffer) and a few more multiplies per step.
- $n$ is **another hyper-parameter** to tune; too large re-introduces MC-level
  variance. Eligibility traces (SARSA(λ), [`sarsalambda.md`](sarsalambda.md))
  achieve a similar multi-step effect more smoothly by averaging over *all* $n$
  with geometric weight $\lambda$, at $O(1)$ amortised cost.

---

## 7. Doing it with pure integers (Q8.8 fixed-point)

No floating point appears anywhere (a contest disqualifier). Every fractional
quantity is **Q8.8 fixed-point**: a real value $v$ is stored as the integer
$\operatorname{round}(v\times 256)$, so `256 == 1.0`. See `rl_core.h` for the
shared `rl_fp` type and the rounded multiply `rl_mul()`.

| Quantity | Real value | Stored `rl_fp` |
|----------|-----------|----------------|
| `RL_FP_ONE` | $1.0$ | `256` |
| $\alpha$ | $\approx 0.1875$ | `48` |
| $\gamma$ | $\approx 0.992$ | `254` |
| $n$ | look-ahead | `NSTEP_N = 5` |
| optimistic $Q_0$ | $80.0$ | `RL_INT(80)` $=20480$ |
| step reward | $-1$ | `-256` |
| mine reward | $-100$ | `-25600` |

**Precomputed $\gamma^i$ powers.** The n-step return needs $\gamma^0\dots
\gamma^n$. We compute them **once** at construction with repeated `rl_mul`,
storing them in `gpow[]`, so each update is just a dot product of stored powers
with buffered rewards — no per-step exponentiation:

```c
q->gpow[0] = RL_FP_ONE;
for (int i = 1; i <= NSTEP_CAP; i++) q->gpow[i] = rl_mul(q->gpow[i-1], q->gamma);
```

The n-step return and update are then a handful of integer ops (`gpow[i]` is
$\gamma^i$ in Q8.8; `boot` is dropped on the episode tail):

```c
rl_fp G = 0;
for (int i = 0; i < nsteps; i++)
    G += rl_mul(q->gpow[i], q->br[nst__at(q, i)]);      // Σ γ^i · R_{τ+i+1}
if (boot)                                                // on-policy bootstrap
    G += rl_mul(q->gpow[nsteps], Q[bs[bj]*nact + ba[bj]]); // + γ^n · Q(S,A)
rl_fp *cell = &Q[bs[i0]*nact + ba[i0]];
*cell += rl_mul(q->alpha, G - *cell);                    // Q(S_τ,A_τ) += α·δ
```

**Why $\gamma$ is so high (`254` ≈ 0.992).** The windy maze pays only $-1$ per
step over an 18-step optimal path; with a small $\gamma$ the discounted signal
decays to nothing before it reaches the start, so the start never learns a
gradient toward the goal. A near-1 $\gamma$ keeps credit alive across the whole
path — and n-step amplifies this further by carrying $n$ real $-1$s back per
update.

**Exploration via optimistic initialisation** (Sutton & Barto §2.6), exactly as
in [`algo_sarsa.h`](../../include/algo_sarsa.h): the maze's mines merely
teleport to start (only the goal terminates), so a near-greedy on-policy agent
would loop forever and never *see* the goal — and what it never reaches, it
cannot learn. Seeding every $Q(s,a)=80.0$ (above any achievable return, all
$\le 0$ here) makes each visit drive a cell **down**, so the greedy policy
prefers the least-visited action and sweeps the state space until it finds the
goal. The behaviour policy can then be held **greedy** ($\varepsilon=0$): the
optimism *is* the exploration. Randomness and $\varepsilon$-greedy stay
integer-only via the shared `xorshift32` PRNG and an integer per-mille $\varepsilon$.

---

## 8. Validated result

Built and run exactly as the contest requires:

```sh
gcc-12 -O2 -std=c99 -Wall -Wextra -Iinclude -o /tmp/test_nstep tests/test_nstep.c && /tmp/test_nstep
```

```
[windy ] reached=1 steps=23 (opt 18) return=-23  -> PASS
[phone ] jank=8.7% (oracle 9.2%) energy=88% of perf  -> PASS

N-STEP SARSA: PASS
```

The windy maze is solved within the $2\cdot\text{opt}+5$ budget, and on the
phone DVFS task n-step SARSA **beats the oracle's jank** while spending only
**88 %** of the always-max-frequency energy — both inside the contest
thresholds, with zero warnings under `-Wall -Wextra` and not a single
`float`/`double`.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*
  (2nd ed.): n-step TD prediction §7.1 (n-step return, error-reduction
  property); **n-step SARSA §7.2** (Algorithm box, $\tau=t-n+1$ bookkeeping);
  optimistic initial values §2.6 —
  [book site](http://incompleteideas.net/book/the-book-2nd.html).
- Chapter 7 lecture notes (UT Austin CS394R, P. Stone): n-step bootstrapping
  slides — <https://www.cs.utexas.edu/~pstone/Courses/394Rspring24/resources/Ch7Slides.pdf>
- Sutton & Barto Chapter 7 summary (n-step return, bias/variance, pseudocode):
  <https://lcalem.github.io/blog/2018/11/19/sutton-chap07-nstep>
- Companion notes in this repo: [`../q-learning.md`](../q-learning.md) (MDP /
  notation, $\varepsilon$-greedy, Q8.8), [`sarsa.md`](sarsa.md) (the $n=1$
  on-policy base), [`sarsalambda.md`](sarsalambda.md) (the eligibility-trace
  alternative to a fixed $n$).
```

