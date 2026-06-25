# Monte-Carlo Control: Learning from Whole Episodes (Integer-Only)

This note explains **Monte-Carlo (MC) methods** for prediction and control, in
the same notation as [`../q-learning.md`](../q-learning.md), and documents the
integer-only (Q8.8) implementation used in this contest
([`../../include/algo_mc.h`](../../include/algo_mc.h)). If Q-learning and SARSA
are the *bootstrapping* corner of the RL family, MC is the opposite corner: it
waits for the **actual return of a complete episode**, with **no bootstrapping**.

---

## 1. Where Monte-Carlo sits in the RL family

The same MDP setting applies: states $\mathcal{S}$, actions $\mathcal{A}$,
transition kernel $P(s'\mid s,a)$, reward $R(s,a,s')$, discount
$\gamma\in[0,1)$. The agent maximises the expected discounted return

$$ G_t = \sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}. $$

**Monte-Carlo control** (Sutton & Barto, Chapter 5) is a **model-free,
*non*-bootstrapping, episodic** method. Piece by piece:

- **Model-free** — it learns directly from sampled experience, never needing
  $P$ or $R$ (like Q-learning and SARSA).
- **Non-bootstrapping** — this is the defining property. Where TD methods update
  $Q(s,a)$ toward an estimate that itself contains other $Q$ values
  ($r+\gamma\max_{a'}Q(s',a')$ for Q-learning, $r+\gamma Q(s',a')$ for SARSA),
  MC updates $Q(s,a)$ toward the **real, observed return** $G_t$ — the actual
  discounted sum of rewards that followed, with no $Q$ on the right-hand side.
- **Episodic** — because the target *is* the full return, MC can only learn at
  an **episode boundary**: it must reach a terminal state (or an artificial cut)
  before any update is possible.

The action-value function is *literally an expectation of a return*,
$Q^{\pi}(s,a) = \mathbb{E}_\pi[\, G_t \mid S_t=s,\,A_t=a \,]$, and MC estimates it
the most direct way imaginable: **play episodes, and average the returns you got.**

---

## 2. Monte-Carlo prediction (policy evaluation)

To evaluate a fixed policy $\pi$, run episodes under $\pi$. For each visited
state-action pair $(s,a)$ at time $t$, compute the realised return $G_t$ and
keep a running average of these returns. By the law of large numbers the average
converges to $Q^\pi(s,a)$.

There are two ways to count visits within one episode:

- **First-visit MC** — for each episode, update $(s,a)$ using only the return
  that follows its *first* occurrence in that episode. The samples are
  i.i.d., and first-visit MC is an *unbiased* estimator of $Q^\pi$ that
  converges with the standard $1/\sqrt{n}$ rate.
- **Every-visit MC** — update $(s,a)$ on *every* occurrence in the episode. The
  samples within an episode are correlated, so the estimator is biased for
  finite data, but the bias vanishes and it **also converges to $Q^\pi$**
  asymptotically. Every-visit is simpler to implement online (no per-episode
  "seen" set) and is what this implementation uses.

### The running-mean update

A sample-average can be written incrementally. With $N(s,a)$ the visit count:

$$ Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}\big[\,G_t - Q(s,a)\,\big]. $$

Replacing $1/N$ with a **constant step size $\alpha$** gives *constant-$\alpha$
MC* (the form introduced in Sutton & Barto Ch. 6 and used to contrast MC with
TD):

$$ \boxed{\; Q(s,a) \leftarrow Q(s,a) + \alpha\big[\,G_t - Q(s,a)\,\big] \;} $$

The bracket $\big[G_t - Q(s,a)\big]$ is the **MC error** — note it contains the
*full return* $G_t$, never a bootstrapped $\max_{a'}Q$. A constant $\alpha$
makes the estimate an exponentially-weighted recency average rather than a true
mean, which is exactly what we want for **non-stationary control**, where the
policy (and hence the returns) keep changing.

---

## 3. Computing the return efficiently (the backward pass)

Returns are cheapest to compute by walking the episode **backward**. With the
recursion $G_t = R_{t+1} + \gamma\,G_{t+1}$ and $G_T = 0$ at the terminal:

```
G ← 0
for t = T-1 down to 0:
    G ← r_{t+1} + γ · G
    Q(s_t, a_t) ← Q(s_t, a_t) + α · ( G − Q(s_t, a_t) )      # every-visit
```

A single backward sweep updates every step of the episode in $O(T)$ — this is
exactly `mc__flush()` in the implementation.

---

## 4. Why MC needs episodes (and what to do when there are none)

Bootstrapping is what lets TD learn from a *single* transition: $Q(s,a)$ can be
nudged using $Q(s')$ immediately, even mid-episode. MC has no such crutch — it
needs $G_t$, and $G_t$ is unknown until the episode ends. **No terminal ⇒ no
return ⇒ no update.** This is the price of being unbiased and bootstrap-free.

For a genuinely **continuing** task (the phone DVFS stream here never sets
`done`), MC would never update at all. The standard fix is to cut **artificial
episodes**: every $K$ steps, treat the buffered segment as a (truncated)
episode, compute its $K$-step return, apply the backward update, and clear the
buffer. The truncated return introduces a little bias (it omits rewards beyond
the cut), but a short $K$ keeps the agent learning online and, conveniently,
makes MC *near-myopic* — ideal for a task whose reward is essentially
per-frame.

---

## 5. Monte-Carlo control: GPI without a model

Control is **generalised policy iteration (GPI)**: alternate

1. **Evaluation** — make $Q$ a better estimate of $Q^\pi$ (the MC averaging
   above), and
2. **Improvement** — make $\pi$ greedier w.r.t. $Q$: $\pi(s)=\arg\max_a Q(s,a)$.

But pure greedy improvement never tries alternative actions, so some $(s,a)$
are never sampled and their estimates never improve. MC needs **persistent
exploration**. Two classical ways:

- **Exploring starts** — begin episodes from random $(s,a)$. Often impossible in
  practice (you can't start a phone mid-stream at an arbitrary state).
- **$\varepsilon$-soft policies** — keep $\pi$ stochastic with
  $\pi(a\mid s)\ge \varepsilon/|\mathcal{A}|$ for all $a$. The familiar
  $\varepsilon$-greedy policy is $\varepsilon$-soft:

  $$
  A_t =
  \begin{cases}
  \text{uniform random action}, & \text{prob. } \varepsilon,\\
  \arg\max_a Q(S_t,a),          & \text{prob. } 1-\varepsilon.
  \end{cases}
  $$

On-policy MC control (Sutton & Barto §5.4) evaluates and improves *this same*
$\varepsilon$-soft policy. It converges to the best policy *among the
$\varepsilon$-soft policies*; annealing $\varepsilon\to0$ (a **GLIE** schedule —
*Greedy in the Limit with Infinite Exploration*) recovers the optimal
deterministic policy.

### Pseudocode (on-policy, $\varepsilon$-soft, constant-$\alpha$)

```
Initialize Q(s,a) ← 0 for all s,a
for each episode:
    generate s0,a0,r1,s1,a1,r2,… ,sT  following ε-greedy(Q)
    G ← 0
    for t = T-1 down to 0:
        G ← r_{t+1} + γ·G
        Q(s_t,a_t) ← Q(s_t,a_t) + α·( G − Q(s_t,a_t) )   # every-visit
    (anneal ε toward a small floor)
```

---

## 6. MC vs TD / Q-learning — the bias–variance trade-off

| | **Monte-Carlo** | **TD / Q-learning, SARSA** |
|---|---|---|
| Target | full return $G_t$ | one-step bootstrap $r+\gamma Q(s',\cdot)$ |
| Bootstraps? | **no** | yes |
| Learns | only at episode end | every step (online) |
| Bias | **unbiased** (of $Q^\pi$) | biased (target uses current estimates) |
| Variance | **high** (sum of many random rewards) | low |
| Markov assumption | not required (uses real returns) | relied upon (bootstrap) |
| Sparse/delayed reward | propagates credit in one sweep | needs many sweeps to propagate |
| Continuing tasks | needs artificial episodes | natural fit |

The headline trade-off: **MC has zero bootstrap bias but high variance; TD has
low variance but bootstrap bias.** On the windy maze the high variance bites
hard — see §8.

---

## 7. Doing it with pure integers (fixed-point)

Everything is **Q8.8 fixed point** (`256 == 1.0`), exactly as in
[`../q-learning.md`](../q-learning.md). Multiplies use `rl_mul()` (Q16.16 product
shifted back by 8, with rounding). The backward return and the constant-$\alpha$
update are a handful of integer ops:

```c
rl_fp G = 0;
for (int t = len-1; t >= 0; t--) {
    G = buf[t].r + rl_mul(gamma, G);          // Q8.8 return, no bootstrap
    rl_fp *cell = &Q[buf[t].s * nact + buf[t].a];
    *cell += rl_mul(alpha, G - *cell);        // Q8.8 MC error
}
```

Fixed-point notes specific to MC:

- **No overflow on long returns.** With $\gamma=254/256\approx0.992$, a geometric
  sum of $-1$/step rewards saturates near $-1/(1-\gamma)\approx-128$ — well
  inside `int32_t` Q8.8 range. A short artificial-episode horizon (continuing
  task) bounds it further.
- **Rounding is benign.** MC already averages noisy returns; the $\pm\tfrac12$
  LSB rounding of `rl_mul` is far smaller than the return variance.
- **Determinism.** The integer `xorshift32` PRNG (`rl_rand`) makes every run
  bit-reproducible — essential for a contest scoreboard.

---

## 8. This implementation ([`algo_mc.h`](../../include/algo_mc.h))

Every-visit, constant-$\alpha$, $\varepsilon$-soft MC control, wrapped in the
uniform `rl_agent` vtable. `step()` buffers each `(s,a,reward_prev)`; on `done`
(or an artificial cut) it runs the backward-return sweep and clears the buffer.

Two practical wrinkles make a *bootstrap-free* learner solve both contest tasks:

1. **Continuing vs episodic, detected at runtime.** The agent starts assuming a
   **continuing** task and cuts artificial episodes every `MC_FLUSH_K_CONT = 16`
   steps — suiting the near-myopic phone DVFS reward. The moment it observes a
   genuine terminal `done`, it concludes the task is **episodic**, switches to
   full-episode returns, and re-initialises cleanly (the brief warmup is
   discarded). No environment-specific knowledge is hard-coded.

2. **Exploration for a sparse goal.** On the windy maze the goal is so distant
   that $\varepsilon$-greedy almost never stumbles onto it (random reaches it
   ~once per 500k steps), and — with *no* bootstrapped value gradient — MC cannot
   "feel" its way toward an unseen goal the way Q-learning can. So we fold a
   small, self-extinguishing **count-based novelty bonus** into the buffered
   reward for the first few visits of each $(s,a)$. Riding inside the MC
   *return*, it propagates backward and pulls the agent toward the frontier until
   the goal is found; the bonus then fades and $\alpha$ is annealed to 0 so the
   value surface freezes and the greedy policy stops thrashing.

Both contest tasks pass with integer-only math
([`../../tests/test_mc.c`](../../tests/test_mc.c)):

```
[windy ] reached=1 steps=24 (opt 18) return=-24  -> PASS
[phone ] jank=11.0% (oracle 9.2%) energy=88% of perf  -> PASS
MONTE-CARLO: PASS
```

The windy result is the instructive one: it took a long training budget, a
novelty bonus, and an $\alpha$-freeze to coax an *unbiased, bootstrap-free*
method through a sparse-reward maze that one-step TD solves almost trivially —
a concrete illustration of MC's **variance** and **credit-propagation** costs.

---

## Strengths & weaknesses (summary)

**Strengths**
- Conceptually simplest value estimator: *average the returns you saw.*
- **Unbiased**; converges to $Q^\pi$ regardless of bootstrap error.
- **Does not assume the Markov property** — it uses real returns, so it is
  robust to state aliasing / partial observability that breaks TD.
- One backward sweep credits an entire trajectory at once.

**Weaknesses**
- **High variance** ⇒ slow, noisy learning; needs many episodes.
- **Episodic only** — needs terminals (or artificial cuts) before it can learn.
- No online mid-episode learning; poor fit for long or continuing horizons.
- Sparse/deceptive rewards are hard: with no value gradient to follow, the agent
  must actually *reach* reward before it can learn anything about it.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
  **Chapter 5 — Monte Carlo Methods** (first-visit vs every-visit §5.1; MC
  control & GPI §5.3; on-policy control with $\varepsilon$-soft policies §5.4;
  constant-$\alpha$ MC) —
  [chapter PDF](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%205.pdf)
- Sutton & Barto, **Chapter 6** for the MC-vs-TD bias/variance contrast and the
  constant-$\alpha$ update form —
  [chapter PDF](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%206.pdf)
- lcalem, "Sutton & Barto summary — chap 05, Monte Carlo methods":
  <https://lcalem.github.io/blog/2018/10/22/sutton-chap05-montecarlo>
- J. Koh, "Key Takeaways 3 (MC Methods) from Sutton and Barto":
  <https://medium.com/mitb-for-all/key-takeaways-3-mc-methods-from-sutton-and-barto-rl-textbook-570a3a4187a6>
- Companion notes in this repo: [`../q-learning.md`](../q-learning.md) (notation,
  fixed-point background) and [`sarsa.md`](sarsa.md) (the on-policy TD sibling).
