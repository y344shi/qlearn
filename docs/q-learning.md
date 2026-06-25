# Q-Learning: Control Loop, Formulas, and an Integer-Only Implementation

This note explains Q-learning from first principles, gives the exact update
formulas and control loop, then shows how to run the whole algorithm using
**only integer arithmetic** (no floating point anywhere) on a combined
*Cliff-with-Wind* gridworld. The companion program is
[`../src/cliff_wind_qlearn.c`](../src/cliff_wind_qlearn.c).

---

## 1. The reinforcement-learning setting

Q-learning solves a **Markov Decision Process (MDP)**, the tuple
$(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

| Symbol | Meaning |
|--------|---------|
| $\mathcal{S}$ | set of states $s$ |
| $\mathcal{A}$ | set of actions $a$ |
| $P(s' \mid s,a)$ | probability of landing in $s'$ after taking $a$ in $s$ |
| $R(s,a,s')$ | reward received on that transition |
| $\gamma \in [0,1)$ | discount factor for future reward |

The agent interacts in discrete steps producing the trajectory

$$ S_0 \xrightarrow{A_0} R_1, S_1 \xrightarrow{A_1} R_2, S_2 \xrightarrow{A_2} \dots $$

A **policy** $\pi(a\mid s)$ chooses actions. The objective is to maximize the
expected **discounted return**

$$ G_t = \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1}. $$

The discount $\gamma$ trades off immediate vs. future reward: $\gamma=0$ is
"myopic" (only the next reward matters); $\gamma \to 1$ makes the agent
far-sighted. Values $\ge 1$ can diverge.

---

## 2. The action-value function and the Bellman optimality equation

The **action-value** (or "Q") function under a policy is the expected return of
taking $a$ in $s$ and following $\pi$ thereafter:

$$ Q^{\pi}(s,a) = \mathbb{E}_\pi\!\left[\, G_t \mid S_t=s, A_t=a \,\right]. $$

The **optimal** action-value $Q^\*(s,a) = \max_\pi Q^\pi(s,a)$ satisfies the
**Bellman optimality equation**:

$$ Q^\*(s,a) = \mathbb{E}\!\left[\, R_{t+1} + \gamma \max_{a'} Q^\*(S_{t+1}, a') \;\middle|\; S_t=s,\, A_t=a \,\right]. $$

Once $Q^\*$ is known, the optimal policy is simply **greedy** with respect to it:
$\pi^\*(s) = \arg\max_a Q^\*(s,a)$.

---

## 3. The Q-learning update rule

Q-learning (Watkins, 1989) is a **model-free, off-policy, temporal-difference
(TD)** control method: it learns $Q^\*$ directly from sampled transitions
$(s, a, r, s')$ without ever knowing $P$ or $R$. After each transition it nudges
$Q(s,a)$ toward a one-step bootstrap target:

$$ \boxed{\; Q(s,a) \leftarrow Q(s,a) + \alpha \Big[\, \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{TD target}} - Q(s,a) \,\Big] \;} $$

The bracketed quantity is the **TD error** $\delta$. An equivalent
"interpolation" form makes the role of the learning rate obvious:

$$ Q(s,a) \leftarrow (1-\alpha)\, Q(s,a) + \alpha\,\big[\, r + \gamma \max_{a'} Q(s',a') \,\big]. $$

- **Learning rate $\alpha \in (0,1]$** — how strongly new information overrides
  the old estimate. $\alpha=0$ learns nothing; $\alpha=1$ keeps only the latest
  sample. In *deterministic* environments $\alpha=1$ is valid; in stochastic
  ones a small constant (e.g. $0.1$) or a decaying schedule is used.
- The update is **off-policy**: the target uses $\max_{a'} Q(s',a')$ (the greedy
  action) regardless of which action the behaviour policy actually took next.
  This is why Q-learning learns the *optimal* edge-hugging path on Cliff Walking
  even while exploring.
- At a **terminal** state there is no successor, so the bootstrap term is
  dropped: target $= r$.

---

## 4. Exploration: ε-greedy

To guarantee every state-action pair keeps getting sampled, actions are chosen
with an **ε-greedy** behaviour policy:

$$
A_t =
\begin{cases}
\text{a uniformly random action}, & \text{with probability } \varepsilon,\\[2pt]
\arg\max_a Q(S_t, a), & \text{with probability } 1-\varepsilon.
\end{cases}
$$

$\varepsilon$ is usually **annealed** from a high value toward a small floor so
the agent explores early and exploits later.

---

## 5. The control loop (off-policy TD control)

Sutton & Barto's pseudocode, stated plainly:

```
Initialize Q(s,a) for all s,a   (e.g. zeros)
for each episode:
    s ← start state
    repeat for each step of the episode:
        a ← action chosen from s using ε-greedy(Q)
        take action a, observe reward r and next state s'
        a*max ← max over a' of Q(s', a')          # 0 if s' is terminal
        Q(s,a) ← Q(s,a) + α [ r + γ·a*max − Q(s,a) ]
        s ← s'
    until s is terminal (or step budget exhausted)
```

The loop is just: **act → observe → bootstrap → correct**, repeated to
convergence.

### Convergence
For a finite MDP, Q-learning converges to $Q^\*$ with probability 1 provided
(i) every state-action pair is visited infinitely often, and (ii) the learning
rate satisfies the Robbins-Monro conditions $\sum_t \alpha_t = \infty$ and
$\sum_t \alpha_t^2 < \infty$. In practice a small constant $\alpha$ with decaying
$\varepsilon$ works well for small tasks like this one.

---

## 6. Doing it with pure integers (fixed-point)

Floating point is unnecessary — and on many embedded / DSP targets undesirable.
We represent every fractional quantity in **Q8.8 fixed point**: a real value
$v$ is stored as the integer $\operatorname{round}(v \times 256)$. So
`256 == 1.0`, `128 == 0.5`, `-256 == -1.0`.

| Quantity | Real value | Stored integer |
|----------|-----------|----------------|
| `FP_ONE` | $1.0$ | `256` |
| $\alpha$ | $0.25$ | `ALPHA_FP = 64` |
| $\gamma$ | $\approx 0.95$ | `GAMMA_FP = 243` |
| step reward | $-1$ | `-256` |
| cliff reward | $-100$ | `-25600` |

**Fixed-point multiply** keeps the scale correct (multiplying two Q8.8 numbers
gives a Q16.16 product, so we shift right by 8 to get back to Q8.8), with
rounding and sign handled explicitly:

```c
static long fx_mul(long a, long b) {     // (a*b) in Q8.8, rounded
    long p = a * b;
    if (p >= 0) return  ((p + 128) >> 8);
    else        return -(((-p) + 128) >> 8);
}
```

The update becomes a handful of integer ops:

```c
long next_max = done ? 0 : max_q(next);              // Q8.8
long target   = reward + fx_mul(GAMMA_FP, next_max); // Q8.8
long td_error = target - Q[s][a];                    // Q8.8
Q[s][a]      += fx_mul(ALPHA_FP, td_error);          // Q8.8
```

Two more pieces avoid floats entirely:

- **Randomness** — a deterministic 32-bit `xorshift32` integer PRNG (fixed seed
  ⇒ reproducible runs).
- **ε-greedy** — `ε` is an integer probability out of 1000; we explore when
  `xrand() % 1000 < eps`, and anneal `eps` with integer division.

---

## 7. The Cliff-with-Wind environment

A 4×6 gridworld combining two Sutton & Barto classics:

- **Cliff Walking (Example 6.6):** the bottom-row cells between Start and Goal
  are a cliff. Stepping onto one gives reward **−100** and teleports the agent
  back to Start (the episode continues). Every other step costs **−1**; reaching
  the Goal ends the episode.
- **Windy Gridworld (Example 6.5):** certain columns have an upward "wind" that
  shifts the agent toward row 0 *after* it moves.

```
 row 0   .   .   .   .   .   .
 row 1   .   .   .   .   .   .
 row 2   .   .   .   .   .   .
 row 3   S   X   X   X   X   G     X = cliff
         col0 ... ............ col5
 wind:    0   0   1   1   0   0    (cells pushed up per column)
```

Actions: `0=up, 1=right, 2=down, 3=left`. The transition is deterministic:
move (clamped to the grid) → apply the wind of the landing column → resolve
cliff/goal. The wind in columns 2–3 makes naively hugging the cliff edge
unreliable, so the learned policy must route up and over.

---

## 8. Validated result

The program (and an exact integer-equivalent reference mirror) was run for 3000
episodes. The greedy policy converges by ~episode 500 and stays optimal:

```
Learning curve (greedy rollout):
  episode  steps  reward  reached
  0        100    -10000  no        <- untrained: never reaches goal
  500      9      -9      yes
  1000     9      -9      yes
  ...
  2999     9      -9      yes        <- optimal: 9 steps, return -9

Greedy policy:                  State values V(s)=max_a Q(s,a):
   >  >  >  >  >  v                 -7   -6   -5   -5   -4   -3
   >  >  >  >  >  v                 -7   -6   -5   -4   -3   -2
   >  >  ^  ^  >  v                 -7   -6    0    0   -2   -1
   S  X  X  X  X  G                 -7    0    0    0    0    0

--- Validation ---
reached_goal=yes  steps=9  return=-9
RESULT: PASS
```

### Convergence plots (rendered in-terminal, no plotting library)

The program also records two per-episode metrics during training and draws them
as dependency-free ASCII charts — appropriate for an embedded target with no
graphics stack.

**Reward vs episode** (per-episode return, y-axis clipped at −120 because early
greedy episodes fall off the cliff many times, reaching ≈ −10000):

```
       0 |                                                                   *
         |                                               ** ** *** ********** 
         |                         *        ****  * **  *  *  *   *           
         |              *  **   *   **   ***    ** *  **                      
         |          * ** **   *  **   ***                                     
         |        *  *       * *                                              
         |  * ** * *                                                          
     -65 | *    *                                                             
         |   *                                                                
    -120 |*                                                                   
  reward +--------------------------------------------------------------------
         0                                                          3000 episode
```

**Bellman delta vs episode** — the mean \|TD error\| $|\delta|$ per episode. It
spikes early then collapses toward 0 as $Q$ approaches the Bellman fixed point:

```
       6 |*
         | *******************************************************************
       0 |
  |delta| +--------------------------------------------------------------------
         0                                                          3000 episode
```

**Best path found** (the greedy trajectory the trained agent actually follows;
arrows are the move taken in each visited cell):

```
   .  .  .  >  >  v
   .  .  >  .  .  v
   >  >  .  .  .  v
   S  X  X  X  X  G
  moves: ^>>>>>vvv   (9 steps, return -9, reached_goal=yes)
```

The agent learns to climb away from the cliff, cross along the top, and drop
into the goal down the wind-free rightmost column — exactly the safe optimal
route, discovered purely from −1/−100 reward signals with integer math only. The
reward curve rises from the clipped floor toward −9 and the Bellman delta decays
to ≈ 0: the two classic signatures of a converged Q-learner.

---

## 9. Build & run

```sh
./build.sh           # builds ./cliff   (or: cc -O2 -std=c99 -o cliff src/cliff_wind_qlearn.c)
./cliff
```

Exit status is `0` on PASS, `1` on FAIL, so it doubles as a CI smoke test.

---

## 10. Running it on the GPU (optional)

`src/cliff_wind_qlearn_gpu.cu` scales the *same* integer algorithm to a GPU by
training a whole **ensemble of independent agents in parallel** — one CUDA thread
per agent, each with its own private Q-table and its own xorshift seed. This both
exercises the GPU and gives a far stronger validation: the policy must converge to
the optimum across *thousands of different random seeds*, not just one.

The per-agent maths (`fx_mul`, `env_step`, ε-greedy) is shared with the CPU
version via `__host__ __device__` functions — bit-for-bit the same integer
operations, just run by the thousands concurrently.

```sh
./build.sh gpu       # builds ./cliff_gpu
./cliff_gpu          # default 8192 agents;  ./cliff_gpu 200000  for more
```

Measured on an NVIDIA RTX 5070 Ti (Blackwell, compute 12.0):

```
Trained 200000 agents on the GPU in 847 ms (236152 agents/sec)
  reached goal : 200000 / 200000  (100%)
  optimal (9 steps, return -9): 200000 / 200000  (100%)
  greedy return  median=-9  best=-9
RESULT: PASS
```

**Toolchain note.** This machine is WSL2 with the CUDA 12.0 toolkit (`nvcc`) and
`gcc-12`/`g++-12`. The RTX 5070 Ti is Blackwell (`sm_120`), newer than CUDA 12.0
knows about, so we compile device code to **PTX for the `compute_90` virtual
architecture** and let the (CUDA 13.2) driver JIT-compile it to `sm_120` when the
module loads. PTX is forward-compatible, so this "compile-old, run-new" pattern is
the supported way to target a GPU newer than your toolkit. nvcc needs a GNU host
compiler it recognises (gcc/g++ ≤ 12 for CUDA 12.0), hence `-ccbin g++-12`.

---

## Sources

- R. S. Sutton & A. G. Barto, *Reinforcement Learning: An Introduction* (2nd ed.),
  Chapter 6 (TD learning; Q-learning §6.5; Cliff Walking Example 6.6; Windy
  Gridworld Example 6.5) —
  [chapter PDF](https://people.cs.umass.edu/~barto/courses/cs687/Chapter%206.pdf)
- *Q-learning* — Wikipedia (update rule, α/γ semantics, convergence):
  <https://en.wikipedia.org/wiki/Q-learning>
- Cliff Walking environment specification (4×12 grid, −1/−100 rewards):
  [reinforcelearn docs](https://rdrr.io/cran/reinforcelearn/man/CliffWalking.html)
- Sutton & Barto figures (Windy Gridworld Fig. 6.10, Cliff Walking Fig. 6.13):
  <http://www.incompleteideas.net/book/first/figures/figures.html>
- Q notation / fixed-point background for embedded targets:
  [Embedded.com, "Fixed-point math"](https://www.embedded.com/fixed-point-math/)
