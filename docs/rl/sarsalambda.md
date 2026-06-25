# SARSA(λ): On-Policy TD Control with Eligibility Traces

This note explains **SARSA(λ)** — on-policy temporal-difference control with
**eligibility traces** (Sutton & Barto, 2nd ed., §12.7) — and documents the
integer-only (Q8.8) implementation in
[`../../include/algo_sarsalambda.h`](../../include/algo_sarsalambda.h). It reuses
the notation of the [Q-learning note](../q-learning.md): an MDP
$(\mathcal S,\mathcal A,P,R,\gamma)$, the action-value $Q(s,a)$, the learning
rate $\alpha$, the discount $\gamma$, and the ε-greedy behaviour policy.

SARSA(λ) sits one step up from one-step SARSA: instead of correcting only the
single state-action pair just visited, it corrects *every* recently-visited pair
in proportion to how recently it occurred. The new knob is $\lambda \in [0,1]$,
the **trace-decay** parameter.

---

## 1. From one-step SARSA to traces

One-step SARSA (see [`../../include/algo_sarsa.h`](../../include/algo_sarsa.h))
applies the on-policy TD error

$$ \delta_t = R_{t+1} + \gamma\,Q(S_{t+1},A_{t+1}) - Q(S_t,A_t) $$

to **just** $Q(S_t,A_t)$:

$$ Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\,\delta_t. $$

The catch: information travels backward only **one** cell per visit. On a long
maze the goal reward has to diffuse hop-by-hop, taking many episodes to reach the
start. Eligibility traces fix this by remembering the trajectory and spreading
each $\delta_t$ across all of it.

Why "on-policy"? Because $\delta_t$ bootstraps off $Q(S_{t+1},A_{t+1})$, where
$A_{t+1}$ is the action the **behaviour policy actually takes next**, not the
greedy $\max_{a'}$ of Q-learning. So SARSA(λ) needs $A_{t+1}$ in hand before it
can compute $\delta_t$ — the algorithm picks $A'$ first, then learns.

---

## 2. The eligibility trace

Keep one extra scalar per state-action pair, $e(s,a)$, the **eligibility trace**.
It is a short-term, fading memory of "how much credit this pair deserves for
whatever happens next". On every step it **decays** by $\gamma\lambda$ and the
just-visited pair is **bumped**. Two bump rules exist:

- **Accumulating traces** (Sutton & Barto Eq. 12.5):
  $$ e_t(s,a) = \gamma\lambda\,e_{t-1}(s,a) + \mathbb 1[s{=}S_t,\,a{=}A_t]. $$
  A pair revisited before its trace has decayed keeps piling up ($>1$).

- **Replacing traces** (Eq. 12.10): on a visit, *set* the trace to 1 instead of
  adding:
  $$
  e_t(s,a) =
  \begin{cases}
  1 & \text{if } s{=}S_t,\ a{=}A_t,\\
  \gamma\lambda\,e_{t-1}(s,a) & \text{otherwise.}
  \end{cases}
  $$
  Replacing caps each trace at 1, which is more stable when a state is visited
  many times in one episode (common in our windy maze, where mine hits teleport
  the agent back to the start over and over).

Our default is **replacing** traces; the implementation supports either via a
`replacing` flag.

---

## 3. The SARSA(λ) update (backward view)

At each step, after computing $\delta_t$ and bumping $e(S_t,A_t)$, **sweep every
pair** and update both $Q$ and the trace:

$$ \boxed{\;\delta_t = R_{t+1} + \gamma\,Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\;} $$

$$ \text{for all } (s,a):\quad
   Q(s,a) \leftarrow Q(s,a) + \alpha\,\delta_t\,e(s,a),
   \qquad
   e(s,a) \leftarrow \gamma\lambda\,e(s,a). $$

This is the **backward view**: one TD error, distributed over the whole trace.
It is mathematically (in the offline limit) equivalent to the **forward view**,
where each state's target is the **λ-return** $G_t^\lambda$, a geometric average
of all $n$-step returns weighted by $(1-\lambda)\lambda^{n-1}$. The trace
implementation needs no look-ahead, so it runs **online**, one cheap sweep per
step.

### The role of λ

| λ value | Behaviour |
|---------|-----------|
| $\lambda = 0$ | $\gamma\lambda = 0$, traces vanish after one step ⇒ **exactly one-step SARSA**. |
| $0 < \lambda < 1$ | credit reaches back several steps, decaying geometrically; best of both worlds. |
| $\lambda = 1$ | traces decay only by $\gamma$ ⇒ on-policy **Monte-Carlo**-like, full-return credit (high variance). |

So λ smoothly interpolates between bootstrapping (low variance, biased) and
Monte-Carlo (unbiased, high variance).

---

## 4. Pseudocode (Sutton & Barto §12.7)

```
Initialize Q(s,a) = 0 and e(s,a) = 0 for all s,a
for each episode:
    s ← start;  a ← ε-greedy(Q, s)
    repeat for each step:
        take action a, observe r, s'
        a' ← ε-greedy(Q, s')                 # on-policy: need the NEXT action
        δ  ← r + γ·Q(s',a') − Q(s,a)         # (drop the γ·Q term if s' terminal)
        e(s,a) ← e(s,a) + 1                  # accumulating  (or  e(s,a) ← 1  replacing)
        for all (x,b):
            Q(x,b) ← Q(x,b) + α·δ·e(x,b)
            e(x,b) ← γ·λ·e(x,b)
        s ← s';  a ← a'
    until s is terminal
    # traces are cleared at episode end (start fresh next episode)
```

The inner full sweep is $O(|\mathcal S|\,|\mathcal A|)$ per step. For the small
tabular tasks here that is fine; production code uses a list of nonzero traces.

---

## 5. SARSA(λ) vs. its neighbours

- **vs. one-step SARSA.** Same on-policy TD target, but traces propagate reward
  backward along the whole trajectory in a single step rather than one cell per
  visit. Result: **much faster credit assignment** and, because it is still
  on-policy, the **same safety bias** — it values the policy it actually follows
  (ε-greedy), so it routes *away* from cliffs/mines rather than hugging them.

- **vs. Q(λ) / Watkins's Q(λ).** Q(λ) is the off-policy traces analogue. Because
  Q-learning bootstraps off the greedy action, an exploratory (non-greedy) action
  breaks the on-policy assumption, so Q(λ) must **cut the traces to zero whenever
  a non-greedy action is taken** — losing much of the trace's benefit during
  exploration. SARSA(λ) never has to cut traces (every action is "on-policy" by
  definition), so its traces stay intact and it typically learns more smoothly.

- **vs. TD(λ) (prediction).** TD(λ) (§12.2) is the *prediction* version (learns
  $V$); SARSA(λ) is the *control* version (learns $Q$ and improves the policy).

---

## 6. Strengths and weaknesses

**Strengths**
- Dramatically faster propagation of sparse/delayed reward (e.g. a goal reached
  after dozens of steps) than one-step methods.
- On-policy ⇒ learns a safe, robust policy under persistent exploration; avoids
  the high-penalty regions Q-learning is willing to skirt.
- Traces are never cut (unlike Q(λ)), so they keep working during exploration.

**Weaknesses**
- Converges to the value of the *ε-greedy* policy, not the optimal greedy policy
  (a bias that shrinks as ε anneals to 0).
- The naïve full sweep is $O(|\mathcal S||\mathcal A|)$ per step.
- Adds two hyperparameters' worth of sensitivity: large $\gamma\lambda$ with a
  large $\alpha$ (and accumulating traces) can over-update and oscillate.

---

## 7. Integer-only (Q8.8) notes

Everything is fixed-point Q8.8 (a real $v$ stored as $\lfloor v\cdot256\rceil$;
see the [Q-learning note §6](../q-learning.md)). Both the Q-table and the trace
table $E$ are `rl_fp` arrays.

- **Trace bump** uses `RL_INT(1)` = `256` (the Q8.8 encoding of $1.0$). Replacing
  sets the cell to `RL_INT(1)`; accumulating adds it.
- **Decay factor** $\gamma\lambda$ is precomputed once per step with
  `rl_mul(gamma, lambda)` — itself a rounded Q8.8 product — then applied with
  `rl_mul(glam, e)`.
- **The update** nests two fixed-point multiplies:
  `Q[i] += rl_mul(alpha, rl_mul(delta, e[i]))`. Doing `delta·e` first then
  `·alpha` keeps each intermediate within Q8.8 range and matches
  $\alpha\,\delta\,e(s,a)$.
- **Sweep skips zero traces** (`if (e != 0)`), which is both faster and avoids
  needless rounding churn on untouched cells.
- **Episode end** clears $E$ with `memset(..., 0, ...)` so traces never leak
  across episodes.

### Tuned constants (and why)

| Quantity | Real value | Q8.8 stored |
|----------|-----------|-------------|
| $\alpha$ | $0.25$ | `RL_FRAC(1,4)` = 64 |
| $\gamma$ | $\approx 0.996$ | `255` |
| $\lambda$ | $0.50$ | `RL_FRAC(5,10)` = 128 |
| trace bump | $1.0$ | `RL_INT(1)` = 256 |

The notably high $\gamma = 255/256$ is deliberate. The windy maze gives only
$-1$ per step and $+0$ at the goal, with the goal far from the start. A short
horizon ($\gamma\approx0.95$) lets the $-1$ gradient die out before it reaches
the start, so the greedy policy never learns which way to go and stalls in place.
With $\gamma\approx0.996$ the per-step cost propagates far enough to form a smooth
value gradient that guides the agent to the distant goal; the $\lambda=0.5$ trace
then accelerates that propagation. With these constants the integer SARSA(λ)
agent solves the windy maze in 20 greedy steps (optimal 18) and on the phone DVFS
task matches the oracle's jank while using ~90% of the performance governor's
energy.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction* (2nd
  ed.), **Chapter 12 — Eligibility Traces**: §12.2 (TD(λ)), §12.5
  (accumulating vs. replacing traces), **§12.7 (SARSA(λ))**, §12.10 (Watkins's
  Q(λ)) — [book site](http://incompleteideas.net/book/the-book-2nd.html),
  [1st-ed. Ch. 7 PDF](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%207.pdf).
- Reference code for the 2nd edition:
  <http://incompleteideas.net/book/code/code2nd.html>.
- lcalem, "Sutton & Barto summary — Chapter 12: Eligibility Traces":
  <https://lcalem.github.io/blog/2019/02/25/sutton-chap12>.
- Companion notes: [`../q-learning.md`](../q-learning.md) (notation, Q8.8
  fixed-point), [`../../include/algo_sarsa.h`](../../include/algo_sarsa.h)
  (the one-step on-policy base this extends).
