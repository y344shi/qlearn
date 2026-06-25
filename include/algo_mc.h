/*
 * algo_mc.h - Monte-Carlo control (epsilon-soft, constant-alpha) as an rl_agent.
 * =============================================================================
 * Model-free, on-policy Monte-Carlo CONTROL wrapped in the uniform rl_agent
 * vtable (see rl_core.h), using INTEGER-ONLY Q8.8 math. Structure mirrors
 * algo_qlearn.h; the substantive difference is that MC does NOT bootstrap: it
 * has no  gamma * max_a' Q(s',a')  term. Instead it buffers a whole episode's
 * (s,a,r) transitions and, at the episode's end, walks them BACKWARD computing
 * the actual sampled return G and regresses Q(s,a) toward it (Sutton & Barto
 * S5.4, "constant-alpha Monte Carlo"):
 *
 *     for t = T-1 .. 0:
 *         G <- r_{t+1} + gamma * G
 *         Q(s_t,a_t) <- Q(s_t,a_t) + alpha * ( G - Q(s_t,a_t) )      (every-visit)
 *
 * Because the target is the realised return G (a complete sample, not a
 * one-step estimate), MC is UNBIASED w.r.t. the policy's value but
 * HIGH-VARIANCE, and it needs an episode boundary before it can learn anything.
 *
 * One-call protocol (rl_core.h):
 *   step(a, reward_prev, feat, done, explore):
 *     1. if a previous (s,a) is pending, RECORD reward_prev as that transition's
 *        reward into the episode buffer (MC buffers; it never bootstraps here).
 *     2. if done==1: run the backward-return MC update over the buffer, clear
 *        it, and return -1.
 *     3. else: epsilon-greedy (explore==1) / greedy choose an action, remember
 *        (s,a) as pending, and return it.
 *
 * CONTINUING TASKS: the phone DVFS task never sets done. MC needs episodes, so
 * we cut ARTIFICIAL episodes: every MC_FLUSH_K_CONT transitions we flush a
 * truncated-return MC update of that segment and clear it. This keeps MC
 * learning online on a non-terminating stream; the short horizon bounds both
 * the variance of the truncated return and the memory, and makes MC near-myopic
 * which suits the (essentially per-frame) DVFS tradeoff. The agent starts in
 * this continuing mode and only switches to full-episode returns once it
 * actually observes a terminal `done` (which proves the task is EPISODIC) -- so
 * no environment-specific knowledge is hard-coded into the agent.
 *
 * SPARSE-REWARD EXPLORATION (the windy maze): epsilon-greedy alone almost never
 * stumbles onto the distant goal (random policy reaches it ~once per 500k
 * steps), and -- crucially -- MC has no bootstrapped value gradient to follow.
 * So we add a count-based NOVELTY BONUS to the buffered reward for the first
 * `bvis` visits of each (s,a). Because the bonus is folded into the MC return
 * and propagates backward, it builds a value gradient that pulls the agent
 * toward unexplored frontier states until it discovers the goal. The bonus is
 * self-extinguishing (it stops after bvis visits per pair, and globally after
 * MC_BONUS_OFF agent-steps), and alpha is annealed toward 0 over the training
 * tail (MC_ANNEAL_LO .. MC_ANNEAL_HI) so the final value surface FREEZES and
 * the greedy policy stops thrashing -- this is what makes the learned policy
 * stable across training budgets rather than a lucky snapshot. On the
 * dense-reward CONTINUING phone task these schedule thresholds (millions of
 * steps) are never reached, and every (s,a) saturates its per-pair visit cap
 * almost immediately, so the bonus and anneal vanish and it reduces to plain
 * short-horizon constant-alpha MC -- exactly textbook every-visit MC.
 *
 * NOTE: the windy maze spends a brief warmup (~30k steps, until its first goal)
 * in continuing mode; on that first terminal the agent re-initialises and
 * restarts cleanly in episodic mode, so the warmup never biases the result.
 *
 *   gamma=254 (~0.992) gives a long horizon so the -1/step return gradient
 *   reaches the distant maze goal; alpha=0.2; eps is set by the harness.
 */
#ifndef ALGO_MC_H
#define ALGO_MC_H

#include <stdlib.h>
#include <string.h>
#include "rl_core.h"

/* --- buffer cap (also the EPISODIC full-episode flush safety-net length) --- */
#define MC_FLUSH_K     512
/* --- artificial-episode length for CONTINUING tasks. A short truncated-return
 *     horizon makes MC near-myopic on the dense-reward DVFS stream (each frame's
 *     jank/power tradeoff is essentially local). The agent starts in this mode
 *     and only leaves it once it observes a genuine terminal `done` (which marks
 *     the task EPISODIC -- e.g. the windy maze reaches its first goal in ~30k
 *     steps, after which full-episode returns are used).                       */
#define MC_FLUSH_K_CONT 16
/* --- novelty-bonus schedule (sparse-reward exploration) --- */
#define MC_BONUS       RL_INT(40)   /* bonus magnitude (Q8.8), folded into return */
#define MC_BVIS        8            /* bonus applies to first MC_BVIS visits/pair  */
#define MC_BONUS_OFF   3200000      /* global: bonus disabled after this many steps */
/* --- alpha anneal-to-zero tail: constant for [0,LO), linear LO..HI down to 0 --- */
#define MC_ANNEAL_LO   4000000
#define MC_ANNEAL_HI   8000000

typedef struct { int s, a; rl_fp r; } mc_step_t;

typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *Q;                       /* [nstates * nact] Q8.8 action values   */
    int   *cnt;                     /* [nstates * nact] visit counts (bonus)  */
    rl_fp alpha0, alpha, gamma; int eps;  /* alpha,gamma Q8.8; eps per-mille */
    rl_rng rng;
    mc_step_t *buf; int cap, len;   /* episode transition buffer              */
    int last_s, last_a, have;       /* pending (s,a) awaiting its reward      */
    long t;                         /* internal training-step counter         */
    int  episodic;                  /* 0 until a terminal `done` is observed   */
} mc_t;

/* greedy argmax; ties broken randomly during exploration (rndtie=1) for better
   coverage, deterministically (lowest index) for evaluation (rndtie=0). */
static inline int mc__argmax(mc_t *q, int s, int rndtie){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0]; int ties = 1;
    for (int k = 1; k < q->nact; k++){
        if (r[k] > bv){ bv = r[k]; b = k; ties = 1; }
        else if (r[k] == bv){ ties++; if (rndtie && (int)(rl_rand(&q->rng) % ties) == 0) b = k; }
    }
    return b;
}
/* epsilon-greedy action choice for state s (explore==0 => pure greedy) */
static inline int mc__choose(mc_t *q, int s, int explore){
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps)
        return (int)(rl_rand(&q->rng) % q->nact);
    return mc__argmax(q, s, 1);
}

/* Walk the buffered episode BACKWARD, accumulating the discounted return G and
   nudging each visited Q(s,a) toward it by constant-alpha. Every-visit MC: each
   buffered (s,a) is updated with the return that follows it. Then clear. */
static inline void mc__flush(mc_t *q){
    rl_fp G = 0;
    for (int t = q->len - 1; t >= 0; t--){
        G = q->buf[t].r + rl_mul(q->gamma, G);
        rl_fp *cell = &q->Q[(size_t)q->buf[t].s*q->nact + q->buf[t].a];
        *cell += rl_mul(q->alpha, G - *cell);
    }
    q->len = 0;
}

/* Append the pending (last_s,last_a) with its just-observed reward (plus the
   self-extinguishing novelty bonus) to the buffer, flushing a truncated episode
   first when the active horizon is reached so MC keeps learning online without
   ever overflowing the buffer. */
static inline void mc__record(mc_t *q, rl_fp reward_prev){
    /* Cut an artificial episode. CONTINUING tasks (no `done` yet seen) flush at
       the short MC_FLUSH_K_CONT horizon so the DVFS stream learns near-immediate
       returns. EPISODIC tasks flush at `done`; the buffer cap is only a safety
       net for very long goal-free maze wandering. */
    int k = q->episodic ? q->cap : MC_FLUSH_K_CONT;
    if (q->len >= k) mc__flush(q);
    rl_fp r = reward_prev;
    int i = q->last_s*q->nact + q->last_a;
    if (q->t < MC_BONUS_OFF && q->cnt[i] < MC_BVIS) r += MC_BONUS;   /* novelty bonus */
    q->cnt[i]++;
    q->buf[q->len].s = q->last_s;
    q->buf[q->len].a = q->last_a;
    q->buf[q->len].r = r;
    q->len++;
}

/* update the constant-alpha learning rate per the anneal-to-zero tail schedule */
static inline void mc__sched(mc_t *q){
    if (q->t < MC_ANNEAL_LO) q->alpha = q->alpha0;
    else if (q->t >= MC_ANNEAL_HI) q->alpha = 0;
    else {
        long num = MC_ANNEAL_HI - q->t, den = MC_ANNEAL_HI - MC_ANNEAL_LO;
        q->alpha = (rl_fp)((long)q->alpha0 * num / den);
    }
}

static inline int mc_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    mc_t *q = (mc_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);
    q->t++; mc__sched(q);

    /* 1. record the reward for the previously chosen (s,a) -- buffer, no bootstrap */
    if (q->have) mc__record(q, reward_prev);

    /* 2. episode end: backward-return MC update over the whole buffer, clear.
          The FIRST genuine terminal proves the task is EPISODIC: from here on we
          use full-episode returns (flush at done). We also wipe the value table,
          visit counts and step clock so the agent restarts cleanly in episodic
          mode -- the brief continuing-mode warmup (which used the wrong,
          truncated horizon) is discarded rather than left to bias learning. */
    if (done){
        if (!q->episodic){
            q->episodic = 1;
            memset(q->Q,   0, (size_t)q->nstates*q->nact*sizeof(rl_fp));
            memset(q->cnt, 0, (size_t)q->nstates*q->nact*sizeof(int));
            q->t = 0; q->len = 0; q->alpha = q->alpha0;
            q->have = 0;
            return -1;
        }
        mc__flush(q);
        q->have = 0;
        return -1;
    }

    /* 3. choose an action for the current state, remember it as pending */
    int act = mc__choose(q, s, explore);
    q->last_s = s; q->last_a = act; q->have = 1;
    return act;
}

static inline int  mc_greedy(rl_agent *a, const rl_fp *feat){
    mc_t *q = (mc_t*)a->ctx; return mc__argmax(q, rl_state_of(q->spec, q->nfeat, feat), 0);
}
static inline void mc_seteps(rl_agent *a, int m){ ((mc_t*)a->ctx)->eps = m; }
static inline void mc_destroy(rl_agent *a){ mc_t *q = (mc_t*)a->ctx; free(q->Q); free(q->cnt); free(q->buf); free(q); }

static inline rl_agent mc_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    mc_t *q = malloc(sizeof(mc_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    q->Q   = calloc((size_t)q->nstates*nact, sizeof(rl_fp));
    q->cnt = calloc((size_t)q->nstates*nact, sizeof(int));
    /* Buffer holds one artificial episode of MC_FLUSH_K transitions; on a long
       maze episode it flushes truncated segments at the cap. */
    q->cap = MC_FLUSH_K;
    q->buf = malloc(sizeof(mc_step_t) * (size_t)q->cap);
    q->len = 0;
    q->alpha0 = RL_FRAC(2,10);  /* 0.2 constant-alpha; tames MC's high variance  */
    q->alpha  = q->alpha0;
    q->gamma  = 254;            /* ~0.992 in Q8.8: long horizon for the maze goal */
    q->eps    = 100;            /* epsilon-soft behaviour policy (per-mille)      */
    q->have   = 0;
    q->t      = 0;
    q->episodic = 0;            /* assume continuing until a terminal is seen   */
    rl_seed(&q->rng, seed);
    rl_agent a; a.name = "montecarlo"; a.ctx = q;
    a.step = mc_step; a.act_greedy = mc_greedy;
    a.set_epsilon = mc_seteps; a.destroy = mc_destroy;
    return a;
}

#endif /* ALGO_MC_H */
