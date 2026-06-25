# Double Q-Learning: Curing the Maximization Bias (Integer-Only)

This note explains **Double Q-learning** (Sutton & Barto, *Reinforcement
Learning: An Introduction*, 2nd ed., §6.7; van Hasselt, 2010): *why* ordinary
Q-learning systematically **overestimates** action values, *how* a pair of
decoupled estimators removes that bias, and *how* we implement the whole thing
in **pure integer Q8.8 fixed-point** as an `rl_agent`
([`../../include/algo_doubleq.h`](../../include/algo_doubleq.h)). It reuses the
notation and integer machinery introduced in
[`../q-learning.md`](../q-learning.md); read that first if the Q-learning update
or the `rl_fp`/`rl_mul` conventions are unfamiliar.

---

## 1. The maximization-bias problem

Recall the Q-learning update (boxed in `q-learning.md` §3):

$$ Q(s,a) \leftarrow Q(s,a) + \alpha\Big[\, r + \gamma \max_{a'} Q(s', a') - Q(s,a) \,\Big]. $$

The bootstrap target uses $\max_{a'} Q(s',a')$. The trouble is that $Q$ is only
an **estimate** of the true value $q_\*$, and that estimate is **noisy** —
during learning every $Q(s',a')$ carries sampling error around its true value.
Taking a `max` over a set of noisy estimates is *not* an unbiased estimate of
the max of the true values. By Jensen's inequality, for noisy estimators
$X_a \approx q(s',a)$,

$$ \mathbb{E}\!\left[\max_a X_a\right] \;\ge\; \max_a \mathbb{E}[X_a] = \max_a q(s',a). $$

So the `max` is biased **upward**: even if each $Q(s',a')$ is individually
unbiased, the *maximum* over actions is, in expectation, too large. This is the
**maximization bias** (Sutton & Barto §6.7). The same set of estimates is used
both to **select** the maximising action and to **evaluate** it — and that
double use of one noisy quantity is exactly what manufactures the optimism.

The bias compounds through bootstrapping: an inflated $\max_{a'}Q(s',a')$ feeds
an inflated target back into $Q(s,a)$, which inflates *its* successors, and so
on. van Hasselt (2010) showed this can make Q-learning perform "very poorly due
to large overestimations of action values" in stochastic MDPs.

### Sutton & Barto's roulette example (§6.7)

Consider a state with many actions, all of whose true value is $q(s,a)=0$ (e.g.
each is a fair $\pm$ bet). The estimates $Q(s,a)$ fluctuate around $0$; some are
positive purely by chance. `max` happily picks one of the lucky positive ones,
so the learned value of the state climbs above $0$ and the greedy policy chases
a phantom gain. The optimal value is $0$; Q-learning reports something positive.

---

## 2. The fix: two decoupled estimators

The cure (van Hasselt, 2010; Sutton & Barto §6.7) is to **decouple action
selection from action evaluation** using **two** independent tables, $Q^A$ and
$Q^B$:

- use one table to **choose** the maximising action,
- use the *other* table to **evaluate** that action's value.

If $Q^A$ and $Q^B$ are independent noisy estimates of the same $q$, then for the
action $a^\* = \arg\max_a Q^A(s',a)$ chosen by $A$, the value $Q^B(s',a^\*)$
read from $B$ is an **unbiased** estimate of $q(s',a^\*)$ — because the noise
that made $a^\*$ look good in $A$ is *uncorrelated* with the noise in $B$. The
lucky over-estimate in $A$ does not get to evaluate itself. In fact the double
estimator can even *under*estimate the max, but it removes the systematic upward
bias that hurts Q-learning.

### Update equations (q-learning.md notation)

On each learning step, flip a fair coin. With probability $\tfrac12$ update
$Q^A$, otherwise $Q^B$:

**If updating $Q^A$:**
$$ a^\* = \arg\max_{a'} Q^A(s', a'), $$
$$ Q^A(s,a) \leftarrow Q^A(s,a) + \alpha\Big[\, r + \gamma\, Q^B\!\big(s', a^\*\big) - Q^A(s,a) \,\Big]. $$

**If updating $Q^B$ (symmetric — selection by $B$, evaluation by $A$):**
$$ b^\* = \arg\max_{a'} Q^B(s', a'), $$
$$ Q^B(s,a) \leftarrow Q^B(s,a) + \alpha\Big[\, r + \gamma\, Q^A\!\big(s', b^\*\big) - Q^B(s,a) \,\Big]. $$

At a **terminal** transition there is no successor, so the bootstrap term is
dropped exactly as in Q-learning: target $= r$.

Each update touches **one** table. Over many steps both tables see, on average,
the same data, so both converge to $q_\*$ — but neither is used to grade its own
greedy choice, so neither inherits the maximization bias.

### Behaviour and greedy policy

The behaviour policy and the evaluation (greedy) policy act on the **sum**
$Q^A + Q^B$ (equivalently their average), which combines the evidence in both
tables:

$$ \pi(s) = \arg\max_a \big[\, Q^A(s,a) + Q^B(s,a) \,\big], \qquad
A_t = \begin{cases}\text{uniform random action}, & \text{prob } \varepsilon,\\
\arg\max_a\big[Q^A(S_t,a)+Q^B(S_t,a)\big], & \text{prob } 1-\varepsilon.\end{cases} $$

---

## 3. Pseudocode (Sutton & Barto §6.7)

```
Initialize Q_A(s,a) and Q_B(s,a) for all s,a
for each episode:
    s <- start state
    repeat for each step:
        a <- epsilon-greedy action from s using (Q_A + Q_B)
        take a, observe reward r and next state s'
        with probability 0.5:
            a* <- argmax_a  Q_A(s', a)              # 0-bootstrap if s' terminal
            Q_A(s,a) <- Q_A(s,a) + alpha [ r + gamma * Q_B(s', a*) - Q_A(s,a) ]
        else:
            b* <- argmax_a  Q_B(s', a)
            Q_B(s,a) <- Q_B(s,a) + alpha [ r + gamma * Q_A(s', b*) - Q_B(s,a) ]
        s <- s'
    until s is terminal (or step budget exhausted)
```

The only differences from Q-learning are: keep **two** tables, **coin-flip**
which to update, and **cross-evaluate** (select with one, score with the other).

---

## 4. Integer-only implementation (Q8.8)

We keep everything in Q8.8 fixed point (`256 == 1.0`), reusing `rl_mul()` for
the fixed-point multiply and the `xorshift32` `rl_rand()` PRNG from
[`rl_core.h`](../../include/rl_core.h) — no floats anywhere. The update is a few
integer ops per step. The coin flip is just the low bit of the PRNG:

```c
if (rl_rand(&q->rng) & 1u) {
    /* update QA: select with QA, evaluate with QB */
    rl_fp boot = 0;
    if (!done) { int astar = argmax(QA, s); boot = QB[s*nact + astar]; }
    rl_fp target = reward_prev + rl_mul(q->gamma, boot);     /* Q8.8 */
    QA[last_s*nact + last_a] += rl_mul(q->alpha, target - QA[last_s*nact + last_a]);
} else {
    /* update QB: select with QB, evaluate with QA (symmetric) */
    rl_fp boot = 0;
    if (!done) { int bstar = argmax(QB, s); boot = QA[s*nact + bstar]; }
    rl_fp target = reward_prev + rl_mul(q->gamma, boot);
    QB[last_s*nact + last_a] += rl_mul(q->alpha, target - QB[last_s*nact + last_a]);
}
```

The agent follows the standard online one-call `rl_agent` protocol (learn from
the pending `(s,a)` using `reward_prev` and the current state, then choose the
next action — see [`rl_core.h`](../../include/rl_core.h)).

### Fixed-point notes / pitfalls

- **Rounding.** `rl_mul` rounds to nearest with explicit sign handling, so the
  TD update keeps its scale (a Q8.8 × Q8.8 product is Q16.16, shifted back by 8).
  Tiny TD errors below half an LSB ($<1/512$) round to zero — a built-in
  dead-band that actually *helps* stability near convergence.
- **Two tables, one PRNG.** A single `rl_rng` drives the coin flip, the
  $\varepsilon$-greedy draw, and the random tie/explore action. The xorshift low
  bit is balanced enough for a fair coin (verified ~50/50).
- **No overflow.** Values live in `int32_t`; with rewards of $-100$ (Q8.8
  $-25600$) and $\gamma<1$ the geometric sum stays far inside `int32_t`.
- **Optimistic initialisation.** See §6 — both tables start at a small positive
  Q8.8 value rather than zero. This is a deliberate, integer-clean choice, not a
  hack around floats.

---

## 5. Double Q-learning vs Q-learning

| | Q-learning | Double Q-learning |
|---|---|---|
| Tables | one $Q$ | two $Q^A, Q^B$ |
| Target | $r+\gamma\max_{a'}Q(s',a')$ | $r+\gamma\,Q^B(s',\arg\max_{a'}Q^A(s',a'))$ |
| Selection vs evaluation | **coupled** (same table) | **decoupled** (different tables) |
| Estimator bias | overestimates ($\ge$) | unbiased / can underestimate |
| Memory | $O(\lvert S\rvert\lvert A\rvert)$ | $2\times$ that |
| Updates/step | 1 cell | 1 cell (only one table per step) |
| Effective learning speed | full | each table sees ~half the data |
| Best when | deterministic / low-noise | stochastic, high-variance returns |

Both are **model-free, off-policy, TD-control** methods and both converge to
$q_\*$ in the tabular limit. Double Q-learning trades a constant factor of
memory (and somewhat slower per-table learning) for an **unbiased** target.

---

## 6. Strengths, weaknesses, and what this repo's tasks reveal

**Strengths.**
- Removes maximization bias → far better behaviour in **stochastic** MDPs with
  noisy/high-variance returns, where vanilla Q-learning chases phantom values.
- Same off-policy convergence guarantees, drop-in update, cheap (one extra
  table; still one cell updated per step).
- The idea generalises: **Double DQN** (van Hasselt et al., 2016) reuses exactly
  this select-with-one / evaluate-with-the-other trick to fix overestimation in
  deep RL, giving a large boost on the Atari benchmark.

**Weaknesses.**
- **Slower learning:** each table is updated only ~half the time, so it needs
  more samples to converge than single-estimator Q-learning.
- **No free optimism.** This is the subtle one and it shows up in our **windy
  maze**. Q-learning's maximization bias is a *bug for value accuracy* but a
  *feature for exploration*: the upward-biased `max` makes under-visited actions
  look attractive, producing a directed push toward the unexplored frontier.
  The maze goal is so sparse that a uniform-random walk reaches it only about
  **once per 400 000 steps**; plain Q-learning's optimism nonetheless lets it
  reach the goal *thousands* of times during training and lock onto the optimal
  18-step path. Double Q-learning, being **unbiased**, lacks that optimism, so
  pure $\varepsilon$-greedy exploration alone never finds the goal and the values
  spiral negative with no terminal anchor.
- It can **underestimate**, which in some tasks slows it the other way.

**Restoring directed exploration the principled way.** Because Double Q-learning
deliberately throws away the optimistic bias, we supply optimism *explicitly*
via **optimistic initialisation** (Sutton & Barto §2.6): both tables start at a
small positive Q8.8 value (`RL_INT(4)`, i.e. `+4.0`). Untried $(s,a)$ then look
attractive until tried, which drives a systematic sweep that does reach the
sparse goal — and the bias is *transient* (it decays as cells are visited) so it
does not corrupt the converged unbiased estimates. The init is tiny next to the
phone task's large rewards (so it is harmless there) yet meaningful for the
maze's $-1$-per-step signal. Paired with a modest learning rate
($\alpha = 24/256 \approx 0.094$) so the seed decays gradually, the agent passes
**both** contest tasks:

```
[windy ] reached=1 steps=18 (opt 18) return=-18  -> PASS
[phone ] jank=8.7% (oracle 9.2%) energy=85% of perf  -> PASS
```

On the **phone DVFS** task (a continuing, comparatively forgiving control
problem) Double Q-learning's unbiased targets are an asset: it lands at jank
**below** the oracle's tolerance while spending only 85% of the performance
governor's energy.

---

## 7. Build & run

```sh
gcc-12 -O2 -std=c99 -Wall -Wextra -Iinclude -o /tmp/test_doubleq \
    tests/test_doubleq.c && /tmp/test_doubleq
```

Zero warnings under `-Wall -Wextra`; the test prints `DOUBLEQ: PASS` and exits
`0`, so it doubles as a CI smoke test.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction* (2nd
  ed.), **§6.7 "Maximization Bias and Double Learning"** (and §2.6 optimistic
  initial values; §6.5 Q-learning) —
  [book site](http://incompleteideas.net/book/the-book-2nd.html),
  [Chapter 6 PDF](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%206.pdf)
- H. van Hasselt, **"Double Q-learning"**, *Advances in Neural Information
  Processing Systems 23* (NIPS 2010) — introduces the double estimator and the
  overestimation analysis:
  [NeurIPS paper PDF](https://proceedings.neurips.cc/paper/3964-double-q-learning.pdf),
  [abstract](https://papers.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)
- H. van Hasselt, A. Guez & D. Silver, **"Deep Reinforcement Learning with
  Double Q-learning"** (AAAI 2016) — Double DQN, the deep-RL successor:
  <https://arxiv.org/abs/1509.06461>
- *Double Q-learning* / maximization bias overview — Wikipedia, *Q-learning*:
  <https://en.wikipedia.org/wiki/Q-learning#Double_Q-learning>
- Companion note in this repo for shared notation and the integer fixed-point
  conventions: [`../q-learning.md`](../q-learning.md)
