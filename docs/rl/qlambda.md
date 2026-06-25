# Watkins's Q(λ): Off-Policy Control with Eligibility Traces

This note explains **Watkins's Q(λ)** — Q-learning augmented with **eligibility
traces** — from first principles, gives the exact integer-only update equations
in the same notation as [`../q-learning.md`](../q-learning.md), and explains the
one subtlety that makes the *off-policy* version tricky: the **trace cutoff on
exploratory actions**. The companion implementation is
[`../../include/algo_qlambda.h`](../../include/algo_qlambda.h) and its test is
[`../../tests/test_qlambda.c`](../../tests/test_qlambda.c).

Reference: Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
**Chapter 12 (Eligibility Traces); Watkins's Q(λ) is §12.10**.

---

## 1. The problem with one-step Q-learning

Recall the one-step Q-learning update (q-learning.md §3). After a transition
$(s, a, r, s')$ it forms the **TD error**

$$ \delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a) $$

and corrects a **single** cell:

$$ Q(s,a) \leftarrow Q(s,a) + \alpha\, \delta. $$

The trouble is *credit assignment over time*. Suppose the agent walks 18 steps
through the windy maze and only at the goal discovers a reward. One-step
Q-learning moves that information back **one cell per episode**: the
second-to-last state learns on episode 1, the third-to-last on episode 2, and so
on. Information seeps backward one hop at a time, so deep tasks need many
episodes to converge.

We would like a single TD error to update *all the states that led here*, not
just the last one. That is exactly what eligibility traces do.

---

## 2. Eligibility traces: a short-term memory of "what led here"

We keep a second table $e(x,b)$, the **eligibility trace**, the same shape as
$Q$. Intuitively $e(x,b)$ measures *how responsible* pair $(x,b)$ is for whatever
happens right now — large if we visited it recently and often, fading toward 0
otherwise.

Each step we (i) mark the pair we just took as fully eligible and (ii) fade all
traces by a factor $\gamma\lambda$:

$$
e(s,a) \leftarrow 1 \qquad\text{(set / "replacing" trace)} \\
e(x,b) \leftarrow \gamma\,\lambda\,e(x,b)\quad \text{for all } (x,b).
$$

- $\gamma$ is the usual discount.
- $\lambda \in [0,1]$ is the **trace-decay** parameter. $\lambda=0$ collapses the
  whole method back to one-step Q-learning (only the latest pair is eligible);
  $\lambda=1$ makes credit reach far back, toward a Monte-Carlo return.

Then **every** TD error is broadcast across the whole trace at once:

$$
\boxed{\;Q(x,b) \leftarrow Q(x,b) + \alpha\,\delta\,e(x,b)\quad\text{for all }(x,b)\;}
$$

A pair that was visited $n$ steps ago still has $e \approx (\gamma\lambda)^n$, so
it gets a fraction $(\gamma\lambda)^n$ of the new error. One reward at the goal
now nudges *the entire trajectory* in a single sweep.

### Accumulating vs replacing traces

Two conventions for the "mark eligible" step:

- **Accumulating:** $e(s,a) \leftarrow e(s,a) + 1$. Revisiting a pair stacks its
  eligibility above 1.
- **Replacing:** $e(s,a) \leftarrow 1$. Revisiting just refreshes it to 1.

Replacing traces are more stable (no unbounded build-up in loopy gridworlds), so
this implementation uses **replacing** traces. With Q8.8 fixed point, "1" is the
constant `RL_INT(1)` = 256.

---

## 3. The λ-return view (why this is principled)

Eligibility traces are the *backward, online* implementation of a *forward*
idea: the **λ-return**. Define the $n$-step return

$$ G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \dots + \gamma^{n-1} r_{t+n}
              + \gamma^{n}\max_{a'} Q(s_{t+n}, a'). $$

The λ-return is a geometric blend of *all* $n$-step returns:

$$ G_t^{\lambda} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{\,n-1}\,G_t^{(n)}. $$

$\lambda$ slides smoothly between the one-step TD target ($\lambda=0$) and the
full Monte-Carlo return ($\lambda=1$). Updating $Q$ toward $G_t^{\lambda}$ would
require waiting for the whole episode (the *forward view*). The eligibility-trace
recursion above produces (for off-line updates) **exactly the same** total
change while running **online, one step at a time** — this is the
forward/backward equivalence of TD(λ). The trace is just an efficient bookkeeping
of "how much each past pair should share in the current error."

---

## 4. Watkins's cutoff: the off-policy correction

Here is the catch that makes *Q*-learning's traces different from SARSA(λ).

The $n$-step / λ-return above bootstraps off $\max_{a'} Q$ — the **greedy**
(optimal-policy) value. That is only a valid estimate of the optimal return **as
long as the behaviour policy actually took greedy actions**. The moment the
ε-greedy behaviour policy takes an **exploratory (non-greedy)** action, the
trajectory from there on no longer reflects the greedy policy, so any multi-step
return that runs *through* that action is biased.

**Watkins's Q(λ) solution** (Sutton & Barto §12.10): decay the traces normally
**while greedy actions are chosen**, but **zero out the entire trace the moment a
non-greedy action is taken** — after using the current step's error. Equivalently,
the trace "looks only as far ahead as the next exploratory action."

```
choose a' (ε-greedy)
apply TD error δ across the trace
if a' was greedy:    e(x,b) ← γλ·e(x,b)   for all (x,b)   # keep the trail
if a' was exploratory: e(x,b) ← 0          for all (x,b)   # cut the trail
```

This is the off-policy price: in the worst case (every action exploratory) the
trace is cut every step and Q(λ) degenerates to one-step Q(0) — no faster than
plain Q-learning. In practice, with a small ε and a mostly-greedy policy, long
greedy runs persist and the speed-up is large.

On a **terminal** transition the bootstrap term is dropped ($\delta = r - Q(s,a)$,
just like one-step Q-learning) and **all traces are cleared** so nothing leaks
across episode boundaries.

---

## 5. Pseudocode (Sutton & Barto §12.10, replacing traces)

```
Initialize Q(s,a) arbitrarily (zeros) and e(s,a) = 0 for all s,a
for each episode:
    e(s,a) ← 0 for all s,a
    s ← start;  a ← ε-greedy(Q, s)
    repeat for each step:
        take a, observe r, s'
        a' ← ε-greedy(Q, s')            # behaviour action at s'
        a* ← argmax_b Q(s', b)          # greedy action at s'  (0 if s' terminal)
        δ  ← r + γ·Q(s', a*) − Q(s, a)  # off-policy TD error (drop γ-term if done)
        e(s, a) ← 1                      # replacing trace
        for all (x, b):
            Q(x, b) ← Q(x, b) + α·δ·e(x, b)
            if a' == a*:  e(x, b) ← γλ·e(x, b)   # greedy → decay
            else:         e(x, b) ← 0            # exploratory → CUT
        s ← s';  a ← a'
    until s terminal
```

In the contest's **one-call online** protocol (`rl_agent.step`), the same logic
is reorganised: each call learns the *previous* pending $(s,a)$ using
`reward_prev` and the freshly observed state, then chooses the next action. The
"is the next action greedy?" test drives the decay-vs-cut decision. See
`qlam_step` in `algo_qlambda.h`.

> Implementation note on "exploratory": we treat an action as exploratory only
> when the random draw fires **and** the sampled action differs from the greedy
> one. If ε-greedy happens to re-pick the greedy action, the policy was *de facto*
> greedy, so the trace is kept — a small but legitimate optimisation.

---

## 6. Fixed-point (integer-only) notes

Everything stays in **Q8.8**: a real value $v$ is the integer
$\operatorname{round}(v\times 256)$, and `rl_mul(a,b)` multiplies two Q8.8 numbers
with rounding (see rl_core.h). The eligibility table `E[]` is `rl_fp` just like
`Q[]`.

| Quantity | Real value | Q8.8 integer |
|----------|-----------|--------------|
| $\alpha$ | $0.2$  | `RL_FRAC(1,5)` = 51 |
| $\gamma$ | $\approx 0.95$ | `243` |
| $\lambda$ | $0.7$ | `RL_FRAC(7,10)` = 179 |
| $\gamma\lambda$ | $\approx 0.665$ | `rl_mul(243,179)` ≈ 170 |
| full eligibility "1" | $1.0$ | `RL_INT(1)` = 256 |

Key integer details:

- **Precompute $\gamma\lambda$ once** as a Q8.8 constant (`glam`) rather than two
  multiplies per cell per step. The trace decay is then a single `rl_mul`.
- **The per-cell update is two nested fixed-point multiplies:**
  `Q += rl_mul(alpha, rl_mul(delta, e))`. Doing the inner `rl_mul(delta, e)`
  first keeps both operands in range and rounds once at each shift.
- **Traces decay to integer 0.** Because Q8.8 has finite resolution, a trace
  below ≈ `1/256` rounds to 0 and stops costing work; we skip cells where
  `E[i] == 0`. This makes the full $|\mathcal{S}|\times|\mathcal{A}|$ sweep cheap
  in practice even though it is written as a full loop.
- **Full sweeps are fine here.** The maze ($96\times4$) and phone
  ($96\times7$) tables are tiny, so a complete sweep over `nstates*nact` each
  step is negligible. (Production code would keep a list of currently-nonzero
  traces.)

---

## 7. Q(λ) vs one-step Q-learning

| | one-step Q-learning | Watkins's Q(λ) |
|---|---|---|
| credit per TD error | one cell $(s,a)$ | the whole eligible trail |
| info propagation | one hop / episode | many hops / episode |
| extra memory | — | second table $e$ (same size as $Q$) |
| extra compute | $O(1)$ per step | $O(\|\mathcal S\|\|\mathcal A\|)$ sweep per step* |
| off-policy handling | trivial (max in target) | needs the **trace cutoff** |
| hyper-parameters | $\alpha,\gamma,\varepsilon$ | + $\lambda$ |

\* with a sparse trace list, $O(\text{active traces})$.

**Strengths**
- **Faster credit assignment / sample efficiency:** one goal reward updates the
  whole path at once — exactly what helps in long, sparse-reward tasks like the
  maze. In our test the greedy policy reaches the goal in the *optimal* 18 steps.
- Still **off-policy**: learns the greedy/optimal policy while exploring.
- Smoothly tunable bias/variance via $\lambda$.

**Weaknesses**
- **The cutoff wastes traces:** frequent exploration repeatedly zeroes the trail,
  so under high ε the method collapses toward Q(0) (Sutton & Barto note it is
  then "no more efficient than Q(0)"). The binary keep/cut rule is crude — it
  ignores that exploratory actions come from a *distribution*.
- Higher $\lambda$ (and accumulating traces) can **increase variance** and, in
  fixed point, accumulate rounding; $\lambda=0.7$ with replacing traces is a
  stable middle ground.
- Extra memory and per-step compute.

---

## 8. Validated result

Built and run exactly as the contest requires:

```sh
gcc-12 -O2 -std=c99 -Wall -Wextra -Iinclude -o /tmp/test_qlambda \
    tests/test_qlambda.c && /tmp/test_qlambda
```

```
[windy ] reached=1 steps=18 (opt 18) return=-18  -> PASS
[phone ] jank=10.8% (oracle 9.2%) energy=92% of perf  -> PASS

Q(LAMBDA): PASS
```

Zero warnings under `-Wall -Wextra`; the binary prints `PASS` and exits `0`. The
agent finds the **optimal-length** mine-free maze path and, on the DVFS task,
keeps jank within the oracle tolerance while spending ~92% of the
performance-governor energy — both purely from integer arithmetic.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction* (2nd
  ed.), **Chapter 12 — Eligibility Traces**; **§12.10 Watkins's Q(λ)** (also
  §12.1 the λ-return, §12.7 replacing traces) —
  [book PDF](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- C. J. C. H. Watkins, *Learning from Delayed Rewards* (PhD thesis, 1989) —
  original Q(λ).
- Sutton & Barto Ch. 12 summary (eligibility traces, Watkins's cutoff) —
  [lcalem notes](https://lcalem.github.io/blog/2019/02/25/sutton-chap12)
- J. Koh, "Eligibility Traces — Sutton and Barto" (λ-return, trace decay) —
  [Medium](https://medium.com/mitb-for-all/key-takeaways-5-eligibility-traces-sutton-and-barto-rl-textbook-6def3eb4bfd5)
- Companion one-step Q-learning note: [`../q-learning.md`](../q-learning.md).
