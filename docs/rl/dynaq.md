# Dyna-Q: Integrated Planning, Acting, and Learning (Integer-Only)

This note explains **Dyna-Q** (Sutton & Barto, *Reinforcement Learning: An
Introduction*, 2nd ed., §8.2): how a single agent can **learn a model of its
environment from experience** and then use that model to do extra **planning**
in the background, dramatically improving sample efficiency over plain
Q-learning. We then show how to implement the whole thing in **pure integer
Q8.8 fixed-point** as an `rl_agent`
([`../../include/algo_dynaq.h`](../../include/algo_dynaq.h)). It reuses the
notation and integer machinery from [`../q-learning.md`](../q-learning.md);
read that first if the Q-learning update or the `rl_fp`/`rl_mul` conventions
are unfamiliar.

---

## 1. Model-free vs. model-based RL

Q-learning (and SARSA, Expected SARSA, …) are **model-free**: they learn a
value function *directly* from sampled transitions $(s,a,r,s')$ and never form
an explicit picture of the environment's dynamics $P(s'\mid s,a)$ or reward
function $R(s,a)$. Each real transition is used once to nudge $Q$, then thrown
away. This is simple and robust, but **sample-inefficient**: information from a
costly real interaction propagates only one Bellman backup at a time, and only
to the single state-action pair just visited.

**Model-based** methods instead build a **model** — a function that, given
$(s,a)$, predicts $(r,s')$ — and then *plan*: they generate simulated
experience from the model and learn from it as if it were real. Classical
planning (value iteration, policy iteration) assumes a *given* model. Dyna's
insight is to **learn** the model online from the same experience stream that
feeds direct RL, then interleave planning with acting.

Sutton & Barto (§8.1) draw the unifying picture: both *learning* and *planning*
estimate value functions by backing up update targets; the only difference is
whether the experience is **real** (from the environment) or **simulated**
(from a model). Dyna-Q runs **both** kinds of backup against the **same** value
table.

---

## 2. The Dyna architecture

Dyna integrates the three processes that an intelligent agent must perform,
all updating one shared `Q` table:

```
            real experience
   env  ─────────────────────►  direct RL  ──►  Q(s,a)
    ▲             │                                ▲
    │             ▼                                │
  acting      model learning                   planning
    │             │                                │
    └──── Q ◄─────┴──► model (r,s' | s,a) ──► simulated experience
```

1. **Acting** — pick an action from the current policy (ε-greedy on `Q`),
   apply it to the real environment.
2. **Direct RL** — update `Q` from the *real* transition with the ordinary
   Q-learning backup.
3. **Model learning** — record the observed $(s,a)\mapsto(r,s')$ so the model
   can reproduce it later.
4. **Planning** — repeat $n$ times: sample a remembered $(s,a)$, query the
   model for its $(r,s')$, and apply the *same* Q-learning backup to `Q` using
   that **simulated** transition.

Planning, acting, and learning all happen on every time step. The planning
backups are pure computation — no environment interaction — so they are
"free" in terms of real samples.

---

## 3. Update equations (q-learning.md notation)

Dyna-Q reuses the boxed Q-learning update from `q-learning.md` §3 verbatim, for
**both** the real and the simulated transitions:

$$ Q(s,a) \;\leftarrow\; Q(s,a) + \alpha\Big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \,\Big]. $$

**Direct RL** applies it to the real $(s,a,r,s')$. The agent also performs
**model learning** for a *deterministic* model — it simply stores the most
recent outcome of each pair:

$$ \mathrm{Model}(s,a) \;\leftarrow\; (r,\, s'). $$

**Planning** then performs $n$ extra applications of the *identical* update,
each on a transition *recalled* from the model:

$$
\begin{aligned}
&\textbf{repeat } n \textbf{ times:}\\
&\quad (s,a) \leftarrow \text{a random previously-visited pair}\\
&\quad (r,s') \leftarrow \mathrm{Model}(s,a)\\
&\quad Q(s,a) \leftarrow Q(s,a) + \alpha\big[\, r + \gamma \max_{a'} Q(s',a') - Q(s,a) \,\big].
\end{aligned}
$$

As in plain Q-learning, a transition into a **terminal** state drops the
bootstrap term ($\max_{a'}Q(s',a')$ is replaced by $0$, so the target is just
$r$). We store a `done` flag in the model so planned backups respect terminals
too.

---

## 4. Pseudocode (Sutton & Barto §8.2, "Tabular Dyna-Q")

```
Initialize Q(s,a) and Model(s,a) for all s,a
Loop forever:
  (a) s  ← current (nonterminal) state
  (b) a  ← ε-greedy(s, Q)
  (c) take a; observe r, s'
  (d) Q(s,a) ← Q(s,a) + α[ r + γ max_a' Q(s',a') − Q(s,a) ]   # direct RL
  (e) Model(s,a) ← (r, s')                                     # model learning
  (f) Loop n times:                                            # planning
        s̃  ← random previously visited state
        ã  ← random action previously taken in s̃
        (r̃, s̃') ← Model(s̃, ã)
        Q(s̃,ã) ← Q(s̃,ã) + α[ r̃ + γ max_a' Q(s̃',a') − Q(s̃,ã) ]
```

With $n=0$ Dyna-Q **degenerates exactly to one-step Q-learning**. Every extra
planning step is one more Bellman backup wrung from already-collected
experience.

---

## 5. Why planning improves sample efficiency

The bottleneck in model-free TD learning is **credit propagation**. A single
real backup moves value information across only one edge of the transition
graph. To carry a reward from the goal back to the start of a long corridor,
plain Q-learning must physically traverse the corridor *many times* — once per
edge per generation of propagation.

Dyna-Q breaks that coupling. Once the environment has been *experienced*, its
transitions live in the model, and the $n$ planning backups per step can
replay them in any order, propagating reward through the graph **without
further real interaction**. In Sutton & Barto's maze example (§8.2, Figure 8.2)
a Dyna-Q agent with $n=50$ planning steps finds a near-optimal path in roughly
**one-tenth** the real episodes a model-free agent ($n=0$) needs — the policy
after the *second* episode is already far better, because the first episode's
single successful trajectory gets replayed dozens of times internally.

Two forces combine:

- **Reuse**: each costly real sample is amortised over $n{+}1$ backups instead
  of one.
- **Reordering**: planning can back up states in (roughly) the order that
  propagates the newly-discovered reward fastest, rather than in the haphazard
  order the agent happened to wander through them.

This is exactly the behaviour we see on the windy maze in
[`../../tests/test_dynaq.c`](../../tests/test_dynaq.c): with $n=10$ planning
steps the greedy policy reaches the goal along the BFS-optimal path.

---

## 6. Strengths and weaknesses

**Strengths**

- **Sample efficiency.** Far fewer real interactions to reach a good policy —
  decisive when real experience is expensive (robotics, hardware-in-the-loop).
- **Anytime trade-off.** $n$ is a dial: more planning per step costs CPU, not
  samples. Tune it to the relative cost of computation vs. environment steps.
- **Simple and on top of Q-learning.** The planning loop reuses the very same
  update; correctness inherits directly from Q-learning.

**Weaknesses**

- **Model error.** Planning trusts the model. In a **stochastic** or
  **non-stationary** environment a deterministic "last-outcome" model is only
  approximate, so planned backups can chase a wrong target. Our phone DVFS task
  has frame-to-frame jitter, so the model is approximate — yet because each
  planned backup is still a valid Q-learning step toward a *recently observed*
  reward, the table converges to a good policy anyway (the test passes:
  energy < 95 % of the always-max-frequency governor with jank within the
  oracle's tolerance).
- **Stale models.** If the world *changes*, a model that only ever stores
  outcomes will keep planning over obsolete transitions. **Dyna-Q+** (§8.3)
  addresses this with an exploration bonus $\kappa\sqrt{\tau}$ that rewards
  revisiting long-untried pairs; we do not need it here.
- **Memory.** Storing $(r,s',\text{done})$ for every visited $(s,a)$ costs
  $O(|\mathcal{S}||\mathcal{A}|)$, fine for tabular tasks but not for large
  spaces.

---

## 7. Fixed-point implementation notes

The implementation in [`../../include/algo_dynaq.h`](../../include/algo_dynaq.h)
is **integer-only Q8.8**, identical in spirit to `algo_qlearn.h`:

- `Q`, `alpha`, `gamma`, and the stored `model_r` are all `rl_fp` (a value $v$
  held as $v\times256$); the backup uses `rl_mul` for the Q8.8 multiply, so
  there is **no float anywhere**.
- The model is three flat arrays indexed `s*nact + a`: `model_r` (Q8.8 reward),
  `model_s2` (next state, an `int`), and `model_done` (terminal flag). A
  `seen[]` bitmap plus a compact `seen_list[]` of visited ids lets planning
  pick a remembered pair in $O(1)$ via the integer PRNG `rl_rand`
  (`rl_rand(&rng) % nseen`) — uniform sampling without any division by a
  float.
- Because the maze rewards are sparse, both `Q` and untried pairs are seeded
  optimistically to `RL_INT(4)`, which encourages the agent to push the
  exploration frontier; the seed is negligible next to the phone task's large
  rewards.
- Hyperparameters (all integer): $\alpha = 24/256 \approx 0.094$,
  $\gamma = 243/256 \approx 0.949$, $\varepsilon = 100$ per-mille (annealed by
  the harness), and **$n = 10$ planning steps** per real step
  (`DYNAQ_PLAN_STEPS`).

### Worked Q8.8 backup

Suppose $\alpha=24$ (≈0.094), $\gamma=243$ (≈0.949), a planned transition gives
$r=\texttt{RL\_INT}(-1)=-256$, and $\max_{a'}Q(s',a')=512$ (≈2.0). Then

$$
\text{target} = r + \gamma\!\cdot\!\max = -256 + \texttt{rl\_mul}(243,512) = -256 + 486 = 230,
$$

and if $Q(s,a)=128$ the new value is
$128 + \texttt{rl\_mul}(24,\,230-128) = 128 + \texttt{rl\_mul}(24,102) = 128 + 10 = 138$
(≈0.54) — a small, rounded step toward the bootstrapped target, computed
entirely in 32/64-bit integers.

---

## 8. References

- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd
  ed., MIT Press, 2018 — **§8.1** (models and planning), **§8.2** (Dyna and
  tabular Dyna-Q, Figure 8.2), **§8.3** (Dyna-Q+ for changing environments).
- R. S. Sutton, "Integrated Architectures for Learning, Planning, and Reacting
  Based on Approximating Dynamic Programming," *Proc. 7th Int. Conf. on Machine
  Learning (ICML)*, 1990 — the original Dyna paper.
