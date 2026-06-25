# SARSA: On-Policy TD Control (Integer-Only)

This note explains **SARSA**, the on-policy sibling of Q-learning, in the same
notation as [`../q-learning.md`](../q-learning.md), then shows the integer-only
(Q8.8 fixed-point) implementation used in this contest
([`../../include/algo_sarsa.h`](../../include/algo_sarsa.h)). If you have read
the Q-learning note, SARSA is a *one-line change to the update target* — but
that one line changes the *kind* of policy the agent learns.

---

## 1. Where SARSA sits in the RL family

The same MDP setting as Q-learning applies: states $\mathcal{S}$, actions
$\mathcal{A}$, transition kernel $P(s'\mid s,a)$, reward $R(s,a,s')$, and
discount $\gamma\in[0,1)$. The agent maximises the expected discounted return
$G_t = \sum_{k\ge 0}\gamma^{k}R_{t+k+1}$.

SARSA (Rummery & Niranjan 1994; named and popularised in Sutton & Barto §6.4)
is a **model-free, temporal-difference (TD), on-policy control** method. Reading
that label piece by piece:

- **Model-free** — it learns directly from sampled transitions, never needing
  $P$ or $R$ (exactly like Q-learning and Monte-Carlo control).
- **Temporal-difference** — it bootstraps: it updates a value estimate toward
  another current estimate after a *single* step, rather than waiting for the
  full episode return (the Monte-Carlo approach).
- **On-policy** — and this is the defining property — it evaluates and improves
  the *same* policy it is using to behave, including that policy's exploration.

In the standard 2×2 taxonomy of one-step TD control:

|                | bootstraps on the **greedy** action | bootstraps on the **behaviour** action |
|----------------|-------------------------------------|----------------------------------------|
| **off-policy** | **Q-learning** (target $\max_{a'}Q$) |                                         |
| **on-policy**  | Expected SARSA (greedy ⇒ equals Q-learning) | **SARSA** (target $Q(s',a')$, $a'$ actually taken) |

SARSA and Q-learning are the two canonical entry points to TD control; n-step
SARSA, SARSA(λ) with eligibility traces, and Expected SARSA all generalise it.

---

## 2. The action-value function it estimates

Like Q-learning, SARSA stores an **action-value** table $Q(s,a)$. But where
Q-learning chases the *optimal* value $Q^\*$, SARSA estimates $Q^{\pi}$ for the
**behaviour policy $\pi$ it is currently following** (e.g. an $\varepsilon$-greedy
policy):

$$ Q^{\pi}(s,a) = \mathbb{E}_\pi\!\left[\, G_t \mid S_t=s,\, A_t=a \,\right]. $$

As exploration is annealed ($\varepsilon \to 0$), $\pi$ approaches greedy and, in
the limit, $Q^{\pi}\to Q^\*$ — so SARSA *also* converges to optimal control,
provided every state-action pair keeps being visited and $\varepsilon$ decays
appropriately (this is the GLIE condition: Greedy in the Limit with Infinite
Exploration).

---

## 3. The SARSA update rule

After observing the quintuple $(s, a, r, s', a')$ — **S**tate, **A**ction,
**R**eward, next **S**tate, next **A**ction — SARSA nudges $Q(s,a)$ toward a
one-step bootstrap target:

$$ \boxed{\; Q(s,a) \leftarrow Q(s,a) + \alpha \Big[\, \underbrace{r + \gamma\, Q(s', a')}_{\text{TD target}} - Q(s,a) \,\Big] \;} $$

The bracket is the **TD error** $\delta_t = r + \gamma Q(s',a') - Q(s,a)$. The
name of the algorithm *is* the data the update consumes: $(s,a,r,s',a')$.

Contrast the **only** structural difference from Q-learning:

$$
\begin{aligned}
\text{Q-learning:}\quad & \text{target} = r + \gamma \, \max_{a'} Q(s', a') \\
\text{SARSA:}\quad      & \text{target} = r + \gamma \, Q(s', a') \quad\text{where } a' \sim \pi(\cdot\mid s')
\end{aligned}
$$

Q-learning bootstraps off the **best** next action (a `max`, regardless of what
it will actually do). SARSA bootstraps off the **action it actually chose next**
under its exploratory behaviour policy. That is the whole difference — and it is
why SARSA is on-policy and Q-learning is off-policy.

- **Learning rate $\alpha\in(0,1]$** — same semantics as Q-learning: $\alpha=0$
  learns nothing, larger $\alpha$ overwrites the old estimate faster. We use
  $\alpha=0.25$.
- **Terminal states** — at a terminal $s'$ there is no successor action, so the
  bootstrap term is dropped: target $= r$. (In this codebase the maze goal is the
  terminal; a mine merely teleports you to start, it is *not* terminal.)
- **Exploration** still uses $\varepsilon$-greedy (§4 of the Q-learning note),
  but here the explored action $a'$ feeds *back into the target*, so the cost of
  exploration is baked into the learned values.

---

## 4. Pseudocode (Sutton & Barto §6.4)

```
Initialize Q(s,a) for all s,a   (optimistically — see §6)
for each episode:
    s  ← start state
    a  ← action chosen from s using ε-greedy(Q)        # choose FIRST action
    repeat for each step of the episode:
        take action a, observe reward r and next state s'
        a' ← action chosen from s' using ε-greedy(Q)   # choose NEXT action
        Q(s,a) ← Q(s,a) + α [ r + γ·Q(s',a') − Q(s,a) ]  # γ·Q term = 0 if s' terminal
        s ← s' ;  a ← a'                               # carry a' forward
    until s is terminal
```

The crucial ordering: the next action $a'$ must be **selected before** the
update, because the update *uses* $Q(s',a')$. The action you commit to next is
the action you learn from — there is no `max`.

### Fitting the one-call online protocol

This contest drives every agent through a single `step(reward_prev, feat, done,
explore)` call (see [`rl_core.h`](../../include/rl_core.h)). SARSA maps onto it
naturally once you respect the ordering above. When a previous $(s_{\text{last}},
a_{\text{last}})$ is pending we need $a'$ for the current state $s'$ before we can
learn, so inside `step()`:

1. Compute the current discrete state $s' = $ `rl_state_of(...)`.
2. If `done`: flush the pending cell with target $= r$ only (no bootstrap),
   clear the pending flag, return `-1`.
3. Otherwise **choose $a'$ for $s'$** with the $\varepsilon$-greedy behaviour
   policy *first*.
4. **Then** apply the SARSA update to $Q(s_{\text{last}}, a_{\text{last}})$ using
   $Q(s', a')$.
5. Remember $(s', a')$ as the new pending pair and return $a'$.

`act_greedy()` is pure $\arg\max$ with no learning and no exploration, used only
for evaluation.

---

## 5. On-policy vs off-policy: why it matters

Because Q-learning's target uses $\max_{a'}Q(s',a')$, it learns the value of the
*greedy* policy even while it behaves randomly — it learns the **optimal** path
and simply ignores the risk that exploration might step off it. SARSA's target
uses the action it *actually* takes, so the value of a state reflects the danger
that the very next exploratory step could be a bad one.

The textbook illustration is **Cliff Walking** (Sutton & Barto Example 6.6) — and
this repo's *windy/slippery cliff* arena is a direct descendant:

- **Q-learning** learns the **optimal** route hugging the cliff edge (shortest,
  highest return *if executed perfectly*). But while it is still
  $\varepsilon$-exploring, that greedy-but-risky policy occasionally takes a
  random action and falls off the cliff, so its *online* return during learning
  is worse and noisier.
- **SARSA** learns a **safer**, slightly longer route that stays away from the
  edge, because its bootstrap accounts for the $\varepsilon$ chance of an
  exploratory plunge. Its *online* return during training is **higher**, even
  though its final greedy policy is sub-optimal by a step or two — unless you
  anneal $\varepsilon \to 0$, at which point SARSA's safe route converges to the
  optimal one.

The slogan: **Q-learning optimises the policy you *intend* to run; SARSA
optimises the policy you *are actually running*, exploration and all.** If acting
badly during learning is expensive (a real robot, a live phone, a cliff), SARSA's
risk-aware online behaviour is the safer choice.

---

## 6. Exploration on this maze: optimistic initialisation

There is a sharp practical consequence of being on-policy in this codebase's
**windy maze**. The maze gives $-1$ per step and $-100$ for a mine, where a mine
*teleports you back to start* (it is not terminal); only the goal terminates.
Under a near-greedy SARSA agent the on-policy values are nearly uniform (every
non-terminating loop is worth $-1/(1-\gamma)\approx-20$), so the greedy policy
gets no goal-ward gradient and tends to **loop forever, almost never reaching the
goal** — and what it never reaches, it cannot learn. (Empirically: vanilla SARSA
reached the goal only ~4 times in 400 000 steps; Q-learning, pulled goal-ward by
its `max`-bootstrap, reached it ~12 800 times.) This is exactly the on-policy
weakness §5 predicts, amplified by sparse termination.

The clean fix — and a classic technique in its own right — is **optimistic
initial values** (Sutton & Barto §2.6): seed every $Q(s,a)$ **above any
achievable return**. Here every real return is $\le 0$, so we initialise
$Q(s,a) = 100.0$. Now each visit drives a cell's value *down*, making the greedy
policy prefer the **least-visited** action in every state — a built-in,
deterministic exploration drive that sweeps the state space breadth-first until
it discovers the goal, after which on-policy bootstrapping shapes a clean
shortest path. With this, SARSA reaches the goal tens of thousands of times and
learns the optimal 18-step route with the behaviour policy held *greedy*
($\varepsilon = 0$) — the optimism *is* the exploration.

---

## 7. Doing it with pure integers (Q8.8 fixed-point)

No floating point appears anywhere (a contest disqualifier). Every fractional
quantity is **Q8.8 fixed-point**: a real value $v$ is stored as the integer
$\operatorname{round}(v\times 256)$, so `256 == 1.0`. See `rl_core.h` for the
shared `rl_fp` type and `rl_mul()` rounded multiply.

| Quantity | Real value | Stored `rl_fp` |
|----------|-----------|----------------|
| `RL_FP_ONE` | $1.0$ | `256` |
| $\alpha$ | $0.25$ | `RL_FRAC(1,4)` $=64$ |
| $\gamma$ | $\approx 0.95$ | `243` |
| optimistic $Q_0$ | $100.0$ | `RL_INT(100)` $=25600$ |
| step reward | $-1$ | `-256` |
| mine reward | $-100$ | `-25600` |

The SARSA update is a handful of integer operations — note `qnext` is read at
the **chosen** action `act`, not a `max`:

```c
int act = sar__choose(q, s, explore);             // pick a' FIRST (ε-greedy)
if (q->have) {                                     // learn pending (last_s,last_a)
    rl_fp *cell  = &q->Q[last_s*nact + last_a];
    rl_fp  qnext = q->Q[s*nact + act];             // Q(s', a')  — on-policy, no max
    rl_fp  target = reward_prev + rl_mul(q->gamma, qnext);  // Q8.8
    *cell += rl_mul(q->alpha, target - *cell);     // Q(s,a) += α·δ
}
q->last_s = s; q->last_a = act; q->have = 1;       // carry (s',a') forward
return act;
```

On a terminal transition the bootstrap is skipped entirely:

```c
if (done) { *cell += rl_mul(q->alpha, reward_prev - *cell); /* target = r */ }
```

Two more pieces stay integer-only, shared with the reference Q-learner:

- **Randomness** — the `xorshift32` PRNG (`rl_rand`); a fixed seed gives
  reproducible runs.
- **$\varepsilon$-greedy** — $\varepsilon$ is an integer probability out of 1000;
  explore when `rl_rand(...) % 1000 < eps`, annealed by integer division in the
  harness.

---

## 8. Validated result

Built and run exactly as the contest requires:

```sh
gcc-12 -O2 -std=c99 -Wall -Wextra -Iinclude -o /tmp/test_sarsa tests/test_sarsa.c && /tmp/test_sarsa
```

```
[windy ] reached=1 steps=18 (opt 18) return=-18  -> PASS
[phone ] jank=9.0% (oracle 9.2%) energy=88% of perf  -> PASS

SARSA: PASS
```

The windy maze is solved at the BFS-optimal **18 steps**, and on the phone DVFS
task SARSA matches the oracle's jank while spending **88 %** of the
always-max-frequency energy — both within the contest thresholds, with zero
warnings under `-Wall -Wextra` and not a single `float`/`double`.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*
  (2nd ed.): SARSA §6.4; Cliff Walking Example 6.6 (SARSA-vs-Q-learning);
  Expected SARSA §6.6; optimistic initial values §2.6 —
  [book site](http://incompleteideas.net/book/the-book-2nd.html),
  [Chapter 6 PDF](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%206.pdf)
- G. A. Rummery & M. Niranjan, *On-line Q-learning using connectionist systems*,
  CUED Technical Report (1994) — the original "modified connectionist Q-learning"
  that Sutton later renamed SARSA.
- *State–action–reward–state–action* — Wikipedia (update rule, on-policy
  framing): <https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action>
- Companion note in this repo: [`../q-learning.md`](../q-learning.md) (shared
  MDP/notation, $\varepsilon$-greedy, Q8.8 fixed-point background).
