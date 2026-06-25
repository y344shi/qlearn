# One-Step Actor-Critic: Policy-Based TD Control (Integer-Only)

This note explains **one-step Actor-Critic** (Sutton & Barto, 2nd ed., §13.5) in
the same notation as [`../q-learning.md`](../q-learning.md), then shows the
integer-only (Q8.8 fixed-point) implementation used in this contest
([`../../include/algo_actorcritic.h`](../../include/algo_actorcritic.h)).

Where Q-learning and SARSA are **value-based** — they learn an action-value
table and *derive* a policy from it (argmax) — Actor-Critic is **policy-based**:
it learns a *parameterised policy directly*, with a learned value function acting
only as a training aid. It is the natural bridge between the value-based TD
methods of chapter 6 and the policy-gradient methods of chapter 13.

---

## 1. Policy-based vs value-based RL

The same MDP setting applies: states $\mathcal{S}$, actions $\mathcal{A}$,
transition kernel $P(s'\mid s,a)$, reward $R$, discount $\gamma\in[0,1)$; the
agent maximises the expected discounted return
$G_t = \sum_{k\ge 0}\gamma^{k}R_{t+k+1}$.

| | **Value-based** (Q-learning, SARSA) | **Policy-based** (Actor-Critic) |
|---|---|---|
| what is learned | $Q(s,a)$, an action-value table | $\pi_\theta(a\mid s)$, a policy, *plus* $V(s)$ |
| how it acts | $\arg\max_a Q(s,a)$ (+ $\varepsilon$-greedy) | sample from / argmax of $\pi$ |
| policy form | implicit, deterministic-greedy | explicit, **stochastic** |
| update | TD on $Q$ | policy-gradient on $\theta$, TD on $V$ |

The policy-based view has real advantages: it can represent **stochastic optimal
policies** (sometimes the best policy *is* randomised), it changes the action
distribution smoothly (no sudden argmax flips), and it extends cleanly to huge or
continuous action spaces. Its weakness is higher variance and sensitivity to step
sizes — which is exactly what the *critic* is there to fix.

---

## 2. The actor (policy) + critic (value) split

Actor-Critic keeps **two** learnable structures:

- **Actor** — action *preferences* $H(s,a)$ defining the policy by a **softmax**:

$$ \pi(a\mid s) \;=\; \frac{e^{H(s,a)}}{\sum_b e^{H(s,b)}}. $$

  The actor is *what acts*. It is improved by policy-gradient ascent.

- **Critic** — a **state-value** estimate $V(s)\approx V^{\pi}(s)$. The critic
  never picks actions; it *criticises* the actions the actor took, by telling it
  whether a transition turned out better or worse than expected.

The name is literal: the actor proposes, the critic evaluates, and the
evaluation is fed straight back to the actor as its learning signal.

---

## 3. The TD error: the critic's signal

After a transition $(s,a,r,s')$ the critic forms the one-step **TD error**

$$ \boxed{\; \delta \;=\; r + \gamma\,V(s') \;-\; V(s) \;}
\qquad(\text{drop } \gamma V(s') \text{ if } s' \text{ is terminal}). $$

This $\delta$ is the same quantity that drives TD(0), SARSA and Q-learning, but
here it plays a second role: it is a one-sample estimate of the **advantage**
$A(s,a)=Q(s,a)-V(s)$ — *how much better than average* action $a$ was in state
$s$. $\delta>0$ means the action did better than the critic expected; $\delta<0$
means worse. The critic itself learns by semi-gradient TD(0):

$$ V(s) \;\leftarrow\; V(s) + \alpha_v\,\delta. $$

---

## 4. The policy-gradient / actor update

The actor performs gradient **ascent** on expected return. The policy-gradient
theorem says the gradient is proportional to $\delta\,\nabla_\theta \ln\pi(a\mid
s)$ — the advantage times the **score function** of the chosen action. For a
tabular softmax over preferences $H$, that score has a famously clean form:

$$
\frac{\partial \ln\pi(a\mid s)}{\partial H(s,b)}
= \mathbb{1}[b=a] - \pi(b\mid s).
$$

So the one-step actor update is

$$
\boxed{
\begin{aligned}
H(s,a) &\;\leftarrow\; H(s,a) + \alpha_h\,\delta\,\bigl(1 - \pi(a\mid s)\bigr) \\
H(s,b) &\;\leftarrow\; H(s,b) - \alpha_h\,\delta\,\pi(b\mid s) \qquad (b\neq a)
\end{aligned}}
$$

Read it intuitively: when $\delta>0$, push the **taken** action's preference up
and pull **every other** action down, each weighted by how probable it was. When
$\delta<0$, do the reverse. Over many visits this sculpts $\pi$ toward actions
the critic keeps being pleasantly surprised by. (Sutton & Barto write this with a
$\gamma^t$ discounting term $I$ on the actor step; for our episodic/continuing
tasks with constant step sizes we fold that into $\alpha_h$, the standard
practical simplification.)

The critic's job is variance reduction: subtracting the baseline $V(s)$ inside
$\delta$ leaves only the *advantage*, so the actor is nudged by how an action
compares to the state's average, not by the raw (high-variance) return.

---

## 5. Pseudocode (Sutton & Barto §13.5, one-step actor-critic)

```
Initialize V(s)           (critic)         -- here: 0
Initialize H(s,a)         (actor prefs)    -- here: optimistic (see §7)
for each episode:
    s <- start state
    repeat for each step:
        a  ~  pi(.|s) = softmax(H(s,.))        # actor chooses (sample, or argmax)
        take a, observe r, s'
        delta <- r + gamma * V(s') - V(s)      # critic's TD error (V(s')=0 if terminal)
        V(s)  <- V(s) + alpha_v * delta        # critic update
        for each action b:                     # actor (policy-gradient) update
            grad   <- (1 if b==a else 0) - pi(b|s)
            H(s,b) <- H(s,b) + alpha_h * delta * grad
        s <- s'
    until s terminal
```

### Fitting the one-call online protocol

The contest drives every agent through a single `step(reward_prev, feat, done,
explore)` call (see [`rl_core.h`](../../include/rl_core.h)). Actor-Critic maps on
cleanly because, unlike SARSA, it does **not** need the next action to compute
its target ($V(s')$ suffices). Inside `step()`:

1. $s' = $ `rl_state_of(...)`.
2. If a previous $(s,a)$ is pending, learn from it: form $\delta = r + \gamma
   V(s')-V(s)$ (no bootstrap when `done`), update $V(s)$, then sweep the actor
   update over all actions using $\pi(\cdot\mid s)$.
3. If `done`: clear the pending flag, return $-1$.
4. Otherwise choose an action for $s'$, remember $(s',a')$, return it.

`act_greedy()` is $\arg\max_a H(s,a)$ with no learning, used for evaluation.

---

## 6. Doing the softmax with pure integers (the crux)

A softmax needs an **exponential**, and floats are a contest disqualifier — this
is the hardest part of the algorithm. Everything is **Q8.8 fixed-point**: a real
$v$ is the integer $\operatorname{round}(v\times256)$, `256 == 1.0`. See
`rl_core.h` for `rl_fp` and the rounded multiply `rl_mul()`.

**(a) Keep preferences bounded.** $H$ is clipped to $[-8,+8]$ (`AC_H_CLIP`) so
the exponentials never blow up and the policy can never become *exactly*
deterministic.

**(b) Subtract the max for stability.** As with any softmax we compute weights
from $H(s,a)-\max_b H(s,b)$, so every exponent is $\le 0$ and $e^{\cdot}\in(0,1]$
— no overflow, ever.

**(c) An integer `exp` over $[-8,0]$.** We precompute a 65-entry Q8.8 lookup
table `exptab[i] ≈ exp(-8·i/64)` once per agent, built **without floats** using
the identity $e^{x}=(e^{x/M})^M$: take a tiny step $h$, approximate
$e^{-h}\approx 1-h+h^2/2$ by its truncated power series, and multiply it in
repeatedly with `rl_mul`. At run time `aca__exp_neg(x)` linearly interpolates
between table entries (still integer: a Q8.8 fraction blended with `rl_mul`).
Inputs below $-8$ saturate to the floor weight.

```c
int32_t w_a = aca__exp_neg(exptab, H[s][a] - maxH);   /* in (W_MIN .. 256] */
```

**(d) Probabilities and sampling are integer too.** $\pi(a\mid s)$ in Q8.8 is
just `w_a * 256 / sum`. To **sample** we draw `rl_rand() % sum` and walk the
cumulative weights (integer roulette); to act **greedily** we take
$\arg\max_a H(s,a)$. Both appear in the header (`aca__sample`, `aca__argmax`).

So the entire policy — exp, normalisation, sampling — is pure `int32_t`/`int64_t`
arithmetic with a 65-word table.

---

## 7. Two integer stabilisers that make it *learn*

Faithful one-step Actor-Critic is correct but, in fixed point on these two very
different tasks, needs two pragmatic guards. Both are standard RL techniques, and
both stay strictly integer-only.

**(1) Advantage clipping (reward-scale invariance).** The two tasks live on
wildly different reward scales: the windy maze pays $-1$/step and $-100$/mine,
while the phone DVFS task pays rewards up to $\sim\pm3700$. A raw actor step
$\alpha_h\delta$ that is gentle on the maze would, on the phone, slam every
preference into the $\pm8$ clip in a single update and destroy all learning. We
therefore feed the actor a **clipped advantage**, $\operatorname{clip}(\delta,
\pm10)$ (`AC_ADV_CLIP`). The band sits *above* the maze's advantage range (so the
maze signal is untouched) but far *below* the phone's, taming the phone's huge
rewards into a bounded gradient. The **critic** still learns from the full,
unclipped $\delta$ — only the actor's gradient is clipped, exactly as gradient
clipping is used in deep policy-gradient methods.

**(2) Optimistic preferences = directed exploration.** A purely stochastic
softmax walk is hopeless on the *sparse, mine-guarded* maze: a uniform-random
policy reaches the goal only ~2 times in 400 000 steps (mines teleport you back
to the start, so a diffusing walk almost never threads the needle). The fix is
the classic **optimistic initial values** (Sutton & Barto §2.6): we seed every
preference $H(s,a)=+1.0$ — above where the bounded gradient will settle it — and
behave by **argmax** of $H$ during training. Because every advantage is bounded,
each visit nudges the *taken* action's preference *down*, so the greedy policy
keeps switching to the **least-tried** action in each state: a deterministic
least-tried-first sweep that systematically covers the maze until it finds the
goal, after which the policy-gradient shapes the optimal route. As preferences
settle the optimism washes out and `act_greedy` is pure $\arg\max H$.

This argmax-of-optimistic-preferences behaviour is the **escort / zero-
temperature limit** of the softmax-with-exploration-bonus policy — an
integer-only Boltzmann/escort approximation, which the contest explicitly allows
"as long as it is integer-only and learns". The full stochastic `aca__sample`
sampler is kept in the header for environments where genuine stochastic
exploration is preferable.

| Quantity | Real value | Stored `rl_fp` |
|----------|-----------|----------------|
| `RL_FP_ONE` | $1.0$ | `256` |
| $\alpha_v$ (critic) | $0.125$ | `RL_FRAC(1,8)` $=32$ |
| $\alpha_h$ (actor) | $\approx 0.021$ | `RL_FRAC(1,48)` |
| $\gamma$ | $\approx 0.95$ | `243` |
| optimistic $H_0$ | $1.0$ | `RL_INT(1)` $=256$ |
| pref clip | $\pm 8.0$ | `RL_INT(8)` $=2048$ |
| advantage clip | $\pm 10.0$ | `RL_INT(10)` |

The critic deliberately learns faster than the actor ($\alpha_v>\alpha_h$): the
advantage signal should be reasonably accurate *before* the policy commits to
chasing it, or the actor amplifies the critic's early noise.

---

## 8. Strengths & weaknesses (what distinguishes Actor-Critic)

**Strengths.**
- Learns an explicit **stochastic policy** $\pi(a\mid s)$ directly, rather than
  reading one off a value table — it can represent randomised optimal policies
  and adjusts action probabilities *smoothly*.
- The critic's baseline gives **lower-variance** updates than pure
  policy-gradient (REINFORCE), and the one-step bootstrap makes it **fully
  online and incremental** — no waiting for episode returns.
- Scales naturally to large/continuous action spaces (just parameterise $\pi$),
  where an argmax over $Q$ would be intractable.

**Weaknesses.**
- Two interacting learners and two step sizes → **more sensitive to tuning** than
  a single $Q$-table; a too-fast actor chases critic noise.
- Bootstrapping adds **bias**; the policy can converge to a local optimum.
- Sparse-reward exploration is genuinely hard for a stochastic policy (hence the
  optimistic-preference directed exploration in §7).

---

## 9. Validated result

Built and run exactly as the contest requires:

```sh
gcc-12 -O2 -std=c99 -Wall -Wextra -Iinclude -o /tmp/test_actorcritic \
    tests/test_actorcritic.c && /tmp/test_actorcritic
```

```
[windy ] reached=1 steps=18 (opt 18) return=-18  -> PASS
[phone ] jank=9.0% (oracle 9.2%) energy=83% of perf  -> PASS

Actor-Critic: PASS
```

The windy maze is solved at the BFS-optimal **18 steps**; on the phone DVFS task
the learned policy matches the oracle's jank (9.0 % vs 9.2 %) while spending
**83 %** of always-max-frequency energy — both inside the contest thresholds,
with zero warnings under `-Wall -Wextra` and not a single `float`/`double`.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*
  (2nd ed.): **one-step Actor-Critic §13.5** (Figure 13.3 pseudocode), the
  policy-gradient theorem §13.2, softmax-in-preferences §13.1, eligibility-trace
  actor-critic §13.6; baselines/advantage §13.4; optimistic initial values §2.6 —
  [book site](http://incompleteideas.net/book/the-book-2nd.html),
  [Chapter 13 PDF](http://incompleteideas.net/book/RLbook2020.pdf)
- L. Calem, *Sutton & Barto summary — Chapter 13, Policy Gradient Methods*
  (one-step actor-critic update equations and pseudocode):
  <https://lcalem.github.io/blog/2019/03/21/sutton-chap13>
- D. Karunakaran, *The Actor-Critic Reinforcement Learning algorithm* (TD-error
  as the critic signal, actor/critic split):
  <https://medium.com/intro-to-artificial-intelligence/the-actor-critic-reinforcement-learning-algorithm-c8095a655c14>
- Companion note in this repo: [`../q-learning.md`](../q-learning.md) (shared
  MDP/notation, $\varepsilon$-greedy, Q8.8 fixed-point background) and
  [`sarsa.md`](sarsa.md) (optimistic initialisation on this same maze).
```
