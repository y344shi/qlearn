# Expected SARSA: Averaging Out the Next Action

This note explains **Expected SARSA** from first principles, gives the exact
update in the same notation as [`../q-learning.md`](../q-learning.md), shows the
pseudocode, contrasts it with SARSA and Q-learning, and documents the
**integer-only (Q8.8) fixed-point** implementation in
[`../../include/algo_expsarsa.h`](../../include/algo_expsarsa.h). The companion
test is [`../../tests/test_expsarsa.c`](../../tests/test_expsarsa.c).

Expected SARSA is Sutton & Barto, *Reinforcement Learning: An Introduction*
(2nd ed.), **section 6.6**.

---

## 1. The setting (same MDP as Q-learning)

Like Q-learning and SARSA, Expected SARSA is a **model-free, temporal-difference
(TD)** control method for a Markov Decision Process
$(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. It learns an action-value table
$Q(s,a)$ from sampled transitions $(s, a, r, s')$ without ever knowing $P$ or
$R$, and acts with an **ε-greedy** behaviour policy (see
[`../q-learning.md`](../q-learning.md) §1–4 for the MDP, return $G_t$, the
Bellman optimality equation, and ε-greedy exploration — all shared here).

The interesting question for any TD method is: **what next-state value do we
bootstrap from?** That single choice is what separates the three siblings.

---

## 2. The TD-method family: what do you put in the target?

After a transition $(s,a,r,s')$ every one-step TD method nudges $Q(s,a)$ toward a
**target** of the form $r + \gamma \cdot \big(\text{value of } s'\big)$:

$$ Q(s,a) \leftarrow Q(s,a) + \alpha\big[\, \underbrace{r + \gamma\,V_{\text{boot}}(s')}_{\text{TD target}} - Q(s,a)\,\big]. $$

The bracket is the **TD error** $\delta$. The three classic choices for
$V_{\text{boot}}(s')$ are:

| Method | $V_{\text{boot}}(s')$ | Uses the realised $a'$? | Policy |
|--------|-----------------------|-------------------------|--------|
| **SARSA** | $Q(s', a')$ — the action actually taken next | yes | on-policy |
| **Q-learning** | $\max_{a'} Q(s', a')$ — the greedy action | no | off-policy |
| **Expected SARSA** | $\sum_{a'} \pi(a'\mid s')\,Q(s',a')$ — the *expectation* | no | on- **or** off-policy |

Expected SARSA sits **exactly between** the other two: instead of sampling one
next action (SARSA) or taking the single best one (Q-learning), it averages over
*all* next actions, weighted by the policy probabilities $\pi(a'\mid s')$.

---

## 3. The Expected SARSA update

$$ \boxed{\; Q(s,a) \leftarrow Q(s,a) + \alpha\Big[\, r + \gamma \sum_{a'} \pi(a'\mid s')\,Q(s',a') - Q(s,a) \,\Big] \;} $$

At a **terminal** state there is no successor, so the bootstrap term is dropped:
target $= r$ (identical convention to Q-learning).

### Closed form under an ε-greedy target policy

The sum looks expensive, but for an **ε-greedy** target policy with $n$ actions
it collapses to a closed form. Every action receives probability $\varepsilon/n$;
the greedy (argmax) action receives an additional $(1-\varepsilon)$:

$$
\pi(a'\mid s') =
\begin{cases}
\dfrac{\varepsilon}{n} + (1-\varepsilon), & a' = \arg\max_b Q(s',b),\\[8pt]
\dfrac{\varepsilon}{n}, & \text{otherwise.}
\end{cases}
$$

Substituting and regrouping, the expectation becomes a simple blend of the
**mean** and the **max** of the next-state row:

$$ \sum_{a'} \pi(a'\mid s')\,Q(s',a') \;=\; \varepsilon\cdot\underbrace{\frac{1}{n}\sum_{a'} Q(s',a')}_{\text{mean}_{a'}\,Q(s',a')} \;+\; (1-\varepsilon)\cdot \max_{a'} Q(s',a'). $$

This is the form implemented here. Two sanity checks fall right out:

- $\varepsilon = 0$ ⇒ target $= \max_{a'} Q(s',a')$ — **identical to Q-learning**.
- $\varepsilon = 1$ ⇒ target $= \text{mean}_{a'} Q(s',a')$ — the value of a uniform-random
  next move.

So Expected SARSA with a **greedy target** *is* Q-learning, and Expected SARSA
whose **target equals its behaviour policy** is the on-policy variant. This is
why the table above lists it as "on- **or** off-policy": the target policy
$\pi$ in the expectation need not be the behaviour policy.

---

## 4. Pseudocode (same style as `q-learning.md` §5)

```
Initialize Q(s,a) for all s,a   (e.g. zeros)
for each episode:
    s ← start state
    repeat for each step of the episode:
        a ← action chosen from s using ε-greedy(Q)        # behaviour policy
        take action a, observe reward r and next state s'
        # expectation over the TARGET policy's next action:
        mx ← max over a' of Q(s', a')                     # 0 if s' is terminal
        mn ← mean over a' of Q(s', a')                    # 0 if s' is terminal
        E  ← ε_target·mn + (1 − ε_target)·mx
        Q(s,a) ← Q(s,a) + α [ r + γ·E − Q(s,a) ]
        s ← s'
    until s is terminal (or step budget exhausted)
```

Unlike SARSA, the loop **does not need the next action $a'$** before it can
learn — the expectation is computed entirely from $Q(s',\cdot)$. The agent can
therefore update *before* choosing how it will actually move.

---

## 5. Why average? Variance vs. SARSA

SARSA's target $Q(s',a')$ depends on which action the ε-greedy policy *happened*
to sample. When exploring, that can be a poor action, injecting noise into the
target even when $Q$ is already correct. Expected SARSA replaces that random
sample with its **expected value**, eliminating the variance due to the random
selection of $A'$ (Sutton & Barto §6.6).

Consequences:

- **Lower variance updates.** The only remaining randomness in the target is the
  environment's own stochasticity in $(r, s')$ — the policy's coin-flip is gone.
- **Larger, more stable learning rate.** Because the target is steadier,
  Expected SARSA tolerates a bigger $\alpha$ than SARSA; in deterministic tasks
  it is often run with $\alpha = 1$. Sutton & Barto report it dominating SARSA
  across $\alpha$ on Cliff Walking.
- **Slightly more computation per step** — an $O(n)$ sweep of the action row to
  form both the mean and the max, versus SARSA's single lookup. For small action
  sets (here $n=4$ and $n=7$) this is negligible.

### vs. Q-learning

Q-learning bootstraps off $\max_{a'}$, which is **optimistically biased**: the
max of noisy estimates over-estimates the true value (the "maximization bias",
Sutton & Barto §6.7). Expected SARSA's blend pulls the target away from the pure
max toward the mean, softening that bias when $\varepsilon > 0$. With a greedy
target ($\varepsilon_{\text{target}}=0$) it recovers Q-learning exactly but keeps
the lower-variance machinery.

### Strengths / weaknesses summary

| | Expected SARSA |
|---|---|
| **Strengths** | Lowest update variance of the three; stable at large $\alpha$; generalises both SARSA and Q-learning; can be on- or off-policy by choosing the target $\pi$; reduces (does not invite) maximization bias |
| **Weaknesses** | $O(n)$ per step instead of $O(1)$; on-policy variant learns the value of the *exploratory* policy, which in cliff/mine worlds can flood the table with the penalty mean (see §6) |

---

## 6. Integer fixed-point implementation (Q8.8)

Everything is done in **Q8.8** (a real $v$ stored as $\operatorname{round}(v\times
256)$; `256 == 1.0`), using only `rl_fp` and `rl_mul` from
[`../../include/rl_core.h`](../../include/rl_core.h) — **no float or double
anywhere**, as the contest requires. The expectation maps to a handful of integer
ops:

```c
/* one pass over the action row builds both max and sum (Q8.8) */
rl_fp bv = r[0]; int64_t sum = r[0];
for (int k = 1; k < nact; k++){ if (r[k] > bv) bv = r[k]; sum += r[k]; }

rl_fp mean     = (rl_fp)(sum / nact);                       /* mean_a Q(s',a)   */
rl_fp eps_frac = (rl_fp)((teps * RL_FP_ONE + 500) / 1000);  /* ε in Q8.8, rounded */
rl_fp greedy   = RL_FP_ONE - eps_frac;                      /* (1-ε) in Q8.8    */
rl_fp expected = rl_mul(eps_frac, mean) + rl_mul(greedy, bv);
```

then the standard TD step (shared with Q-learning):

```c
rl_fp nexp   = done ? 0 : expected;                 /* drop bootstrap at terminal */
rl_fp target = reward + rl_mul(gamma, nexp);        /* Q8.8 */
*cell       += rl_mul(alpha, target - *cell);       /* Q8.8 */
```

**Fixed-point notes / pitfalls handled:**

- **`eps_frac = ε/1000` in Q8.8.** Computed as
  `(ε·256 + 500)/1000` so the integer division **rounds to nearest** instead of
  truncating toward zero — important because $\varepsilon$ is given in per-mille
  and truncation would systematically bias the blend toward the max term.
- **64-bit accumulator for the mean.** The action-value sum is accumulated in
  `int64_t` before dividing by `nact`, so summing several large-magnitude Q8.8
  cells (e.g. mine states near $-100\cdot256$) cannot overflow.
- **`rl_mul` rounds** both products (`mean` and `max` contributions) symmetrically
  about zero, and the two weights `eps_frac + greedy_frac = RL_FP_ONE` sum to
  exactly 1.0 in Q8.8 — so the blend introduces no scale drift.

### A practical knob: a near-greedy *target* epsilon

On the windy **mine** maze, the pure on-policy variant (target $\varepsilon$ =
behaviour $\varepsilon$) struggles: with a high early exploration rate the
`mean` term mixes the $-100$ mine-hitting actions into *every* state's target,
flooding the grid to the discounted wall-bumping floor ($-1/(1-\gamma) \approx
-20$) before the goal's value can propagate, and the greedy policy gets stuck in
a self-loop. The implementation therefore caps the **target-policy** epsilon used
inside the expectation (`target_eps`, default 50‰) independently of the annealed
**behaviour** epsilon. A near-greedy target is precisely the *off-policy* form of
Expected SARSA (§3) — lower-variance, less pessimistic, and it lets the goal
value propagate cleanly. Behaviour exploration still anneals as the harness
dictates.

---

## 7. Validated result

Built and run exactly as the contest specifies:

```sh
gcc-12 -O2 -std=c99 -Wall -Wextra -Iinclude -o /tmp/test_expsarsa \
    tests/test_expsarsa.c && /tmp/test_expsarsa
```

```
[windy ] reached=1 steps=18 (opt 18) return=-18  -> PASS
[phone ] jank=8.7% (oracle 9.2%) energy=90% of perf  -> PASS

EXPECTED SARSA: PASS
```

The agent finds the **optimal 18-step** mine-free route on the windy maze, and on
the phone DVFS task it beats the oracle jank threshold while spending only ~90% of
the performance governor's energy — all with integer arithmetic only.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction*
  (2nd ed.), **§6.6 "Expected Sarsa"** (the update and its lower-variance
  argument), §6.4 (Sarsa), §6.5 (Q-learning), §6.7 (maximization bias),
  Example 6.6 (Cliff Walking) —
  [chapter PDF](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%206.pdf)
- H. van Seijen, H. van Hasselt, S. Whiteson & M. Wiering, *"A Theoretical and
  Empirical Analysis of Expected Sarsa"* (IEEE ADPRL 2009) — formal variance
  reduction and convergence analysis.
- *SARSA / Expected SARSA* — Wikipedia (closed-form ε-greedy expectation):
  <https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action>
- Q notation / fixed-point background for embedded targets:
  [Embedded.com, "Fixed-point math"](https://www.embedded.com/fixed-point-math/)
