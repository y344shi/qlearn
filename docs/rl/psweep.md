# Prioritized Sweeping: Model-Based Planning Where It Matters

This note explains **Prioritized Sweeping** — model-based reinforcement learning
that focuses its planning effort on the state-action pairs whose values are about
to change the most — from first principles, gives the exact integer-only update
equations in the same notation as [`../q-learning.md`](../q-learning.md), and
contrasts it with **Dyna-Q**, whose planning samples experience *uniformly at
random*. The companion implementation is
[`../../include/algo_psweep.h`](../../include/algo_psweep.h) and its test is
[`../../tests/test_psweep.c`](../../tests/test_psweep.c).

Reference: Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
**Chapter 8 (Planning and Learning with Tabular Methods); Prioritized Sweeping is
§8.4**. Dyna-Q is §8.2; the Dyna architecture is §8.1.

---

## 1. Model-free vs. model-based RL

Recall one-step Q-learning (q-learning.md §3). After each real transition
$(s, a, r, s')$ it forms the **TD error**

$$ \delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a) $$

and corrects a **single** cell:

$$ Q(s,a) \leftarrow Q(s,a) + \alpha\, \delta. $$

Then it throws the transition away. This is **model-free**: each sample is used
exactly once. In a long maze that is wasteful — the agent walks 18 steps to the
goal, learns about *one* of those cells, and must walk the whole way again before
the next cell can learn.

A **model-based** agent instead *remembers* the dynamics it has seen. For a
deterministic environment a tabular model is trivial: after observing
$(s, a, r, s')$ we store

$$ \widehat{R}(s,a) = r, \qquad \widehat{S}'(s,a) = s'. $$

Once we have a model we can do **planning**: run extra Q-learning backups using
remembered transitions instead of real ones. A backup that uses the model looks
identical to the real one — it just reads $r$ and $s'$ from the tables:

$$ Q(s,a) \leftarrow Q(s,a) + \alpha\big[\,\widehat{R}(s,a) + \gamma \max_{a'} Q(\widehat{S}'(s,a), a') - Q(s,a)\,\big]. $$

The question is **which** simulated transitions to back up, and **in what
order**. That is the whole game.

---

## 2. Dyna-Q: planning by uniform random replay

**Dyna-Q** (§8.2) interleaves real steps with planning. After every real step it
performs $N$ planning updates; each update picks a **previously seen** $(s,a)$
**uniformly at random**, looks up $(\widehat{R}, \widehat{S}')$, and applies the
backup above.

```
loop:
    take a real step, observe (s, a, r, s'); update the model
    Q-learning backup on (s, a)
    repeat N times:                       # the planning phase
        (s, a) <- a uniformly random previously-seen pair
        r, s'  <- model(s, a)
        Q(s,a) += alpha [ r + gamma max_a' Q(s', a') - Q(s,a) ]
```

Dyna-Q already converges in far fewer real steps than model-free Q-learning,
because each real step triggers $N$ extra backups that recirculate old
experience. But uniform sampling is **blind**: early on, almost every state still
has $Q \approx 0$ and a backup changes nothing. The agent wastes most of its
planning budget re-backing-up pairs whose values are already correct, while the
one newly-informative transition near the goal waits its turn.

---

## 3. The key idea: prioritize by TD-error magnitude

Information only flows when a backup actually *moves* a value. A backup of
$(s,a)$ moves $Q(s,a)$ by $\alpha\,|\delta|$, so the pairs worth updating are
exactly those with a **large TD error**

$$ P(s,a) \;=\; \big|\, \widehat{R}(s,a) + \gamma \max_{a'} Q(\widehat{S}'(s,a), a') - Q(s,a) \,\big|. $$

Prioritized Sweeping keeps a **priority queue** of pairs keyed by $P$. It always
backs up the pair with the **largest** pending change first. A pair is only
enqueued when its priority exceeds a small threshold $\theta$ (so trivially small
errors never consume the budget):

$$ \text{enqueue } (s,a) \quad\text{iff}\quad P(s,a) > \theta. $$

When the goal is first reached, exactly **one** pair has a large error. We back
it up. That changes $\max_{a'} Q(s', \cdot)$ for its **predecessors**, giving
*them* a large error, so we enqueue and back *them* up next, and so on. The wave
of value propagates **backward from the goal in priority order** — every backup
is a useful one. This is **backward focusing**.

---

## 4. Predecessors: who leads into a state

To propagate a value change backward we need, for a state $s$, the set of pairs
that lead into it:

$$ \mathrm{Pred}(s) = \{\, (\bar s, \bar a) : \widehat{S}'(\bar s, \bar a) = s \,\}. $$

Two ways to obtain them, both integer-only:

- **Maintain a predecessor list** per state as the model is built.
- **Scan the model** for any seen $(\bar s, \bar a)$ with
  $\widehat{S}'(\bar s, \bar a) = s$.

The tables here are small ($\le$ a few hundred states $\times$ a handful of
actions), so the implementation simply **scans the model** — it is $O(|\mathcal{S}|\,
|\mathcal{A}|)$ per popped pair, with no extra bookkeeping and no allocation in the
hot loop. (For large state spaces an explicit predecessor list is the standard
choice.)

---

## 5. The algorithm

Sutton & Barto §8.4, stated in our notation. $\theta$ is a small priority
threshold; $N$ is the planning budget per real step.

```
Initialize Q(s,a)=0, an empty model, and an empty priority queue PQueue.
for each real step:
    observe (s, a, r, s')                            # the real transition
    Model(s,a) <- (r, s')                            # learn the (det.) model
    P <- | r + gamma max_a' Q(s',a') - Q(s,a) |
    if P > theta:  PQueue.push((s,a), priority=P)

    repeat N times, while PQueue not empty:           # the focused sweep
        (s, a) <- PQueue.pop_max()
        r, s'  <- Model(s, a)
        Q(s,a) += alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]   # the backup
        for each (sbar, abar) in Pred(s):             # backward focusing
            rbar, _ <- Model(sbar, abar)
            P <- | rbar + gamma max_a' Q(s, a') - Q(sbar, abar) |
            if P > theta:  PQueue.push((sbar,abar), priority=P)
```

At a **terminal** transition the bootstrap term is dropped (target $= r$), exactly
as in q-learning.md §3; the model records the pair as terminal so its backups and
its successors' backups use $\max_{a'} Q(s', \cdot) = 0$.

### Why the priority queue allows in-place key raises

The same pair can become urgent more than once before it is popped. Rather than
storing duplicates, the queue keeps a position map `pos[sa]` and, on a repeated
push, **raises** the existing key to the larger priority. This keeps the heap
size bounded by the number of distinct pairs and guarantees the most urgent pair
is always at the root.

---

## 6. Prioritized Sweeping vs. Dyna-Q

Both learn a model and plan; they differ only in **planning order**.

| | Dyna-Q (§8.2) | Prioritized Sweeping (§8.4) |
|---|---|---|
| Pair selection | uniform random over seen pairs | highest \|TD error\| first |
| Direction | undirected | **backward** from changed values |
| Wasted backups | many (most pairs already correct) | few (skip \|δ\| ≤ θ) |
| Data structure | a list of seen pairs | a priority queue + predecessors |
| Real steps to converge | low | **lowest** |
| Per-step cost | $O(N)$ | $O(N \cdot \text{pred-scan})$ |

The trade is **compute per step for samples**: Prioritized Sweeping does more
work *per* backup (queue maintenance + predecessor scan) but needs **dramatically
fewer** backups to reach a good policy, because every backup it chooses is one
that actually changes a value. On the windy maze it discovers the optimal path
having effectively swept value backward from the goal along the model graph, not
by random replay.

---

## 7. Doing it with pure integers (fixed-point)

Everything stays in **Q8.8** (see q-learning.md §6): a value $v$ is the integer
$\operatorname{round}(v\times 256)$, and `rl_mul` multiplies two Q8.8 numbers
with rounding. No floats appear anywhere.

- **Model tables.** $\widehat{R}(s,a)$ is a `rl_fp` (the Q8.8 reward);
  $\widehat{S}'(s,a)$ is a plain `int` state index; `seen` / `done` are bytes.
- **Priority.** $P = |\delta|$ is computed with the same `rl_mul(gamma, maxQ)`
  backup math, then the sign is stripped by an integer absolute value. Because
  $P \ge 0$ it slots straight into an integer max-heap keyed on `rl_fp`.
- **Threshold $\theta$.** A small Q8.8 constant — here `RL_FRAC(1,8)` ≈ 0.125.
  Larger $\theta$ ⇒ fewer, more selective backups; too large and useful ripples
  are pruned, too small and the queue fills with noise. On the (mildly
  stochastic, jittery) phone task a *higher* $\theta$ and a *moderate* $N$ matter:
  the deterministic-model assumption is only approximate there, so over-planning
  amplifies the model's noise. The chosen $\alpha = 0.25$, $\gamma \approx 0.90$,
  $\theta \approx 0.125$, $N = 20$ pass both tasks with margin.
- **No `<math.h>`, no division by non-constants in the hot loop**, and a fixed
  `xorshift32` PRNG for $\varepsilon$-greedy — fully reproducible, embedded-friendly.

The whole backup is a handful of integer ops, identical to q-learning's, just
applied in priority order:

```c
rl_fp nmax   = done ? 0 : rl_mul(gamma, max_q(s2));   // Q8.8
rl_fp target = model_r + nmax;                        // Q8.8
rl_fp delta  = target - Q[s][a];                      // Q8.8 (priority = |delta|)
Q[s][a]     += rl_mul(alpha, delta);                  // Q8.8
```

---

## 8. Strengths and weaknesses

**Strengths**
- **Sample efficiency.** Among tabular methods it typically reaches a good policy
  in the *fewest real environment steps* — ideal when real interaction is the
  expensive resource.
- **Directed credit assignment.** Value changes propagate backward exactly along
  the paths that produced them, so a single discovery (e.g. the goal) is exploited
  immediately and globally rather than seeping back one hop per episode.
- **Self-pruning.** The $\theta$ gate means a converged region costs almost
  nothing: with no large errors the queue empties and the sweep stops early.

**Weaknesses**
- **Needs a model.** It stores and trusts $\widehat{R}, \widehat{S}'$. In
  **stochastic** environments a single-sample deterministic model is biased; the
  textbook fix is to learn expected/empirical transition probabilities and back up
  an expectation (more memory and arithmetic).
- **Per-step compute.** Queue maintenance plus predecessor discovery cost more per
  backup than Dyna-Q's random draw. With a model scan for predecessors the cost
  grows with the table size.
- **Tuning sensitivity.** $\theta$ and $N$ interact: too much planning on an
  imperfect model (as on the jittery phone task) can *hurt*, chasing noise faster
  than the policy stabilises.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
  **Chapter 8** — the Dyna architecture (§8.1), Dyna-Q (§8.2), and **Prioritized
  Sweeping (§8.4)**, including the priority-queue formulation and the
  predecessor/backward-focusing idea —
  [chapter PDF](http://incompleteideas.net/book/RLbook2020.pdf) (Ch. 8).
- A. W. Moore & C. G. Atkeson, *Prioritized Sweeping: Reinforcement Learning with
  Less Data and Less Time*, Machine Learning 13 (1993) — the original method.
- J. Peng & R. J. Williams, *Efficient Learning and Planning Within the Dyna
  Framework* (1993) — the closely related "queue-Dyna".
- Q notation / fixed-point background for embedded targets:
  [Embedded.com, "Fixed-point math"](https://www.embedded.com/fixed-point-math/)
