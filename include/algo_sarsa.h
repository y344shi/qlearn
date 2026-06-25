/*
 * algo_sarsa.h - SARSA: on-policy TD control as an rl_agent.
 * =========================================================
 * On-policy one-step temporal-difference control (Sutton & Barto, S6.4),
 * wrapped in the uniform rl_agent vtable (see rl_core.h) using INTEGER-ONLY
 * Q8.8 math. Structure mirrors algo_qlearn.h; the ONLY substantive change is
 * the TD target: instead of bootstrapping off max_a' Q(s',a') (the greedy
 * action, off-policy), SARSA bootstraps off Q(s',a') where a' is the action
 * the behaviour policy ACTUALLY chooses next.
 *
 * Update:  Q(s,a) += alpha * ( r + gamma * Q(s',a') - Q(s,a) )
 *
 * Name SARSA = the quintuple it uses: State, Action, Reward, State', Action'.
 *
 * EXPLORATION via OPTIMISTIC INITIALISATION. Because SARSA is on-policy, its
 * value targets only reflect states the behaviour policy actually visits -- it
 * does not get Q-learning's free goal-ward gradient from the max-bootstrap. In
 * sparse-reward / trap-heavy tasks (a maze that teleports you back to start on
 * a mine) a near-greedy SARSA agent therefore tends to loop and almost never
 * stumble onto the goal, so it never learns. The classic remedy (Sutton &
 * Barto S2.6) is OPTIMISTIC INITIAL VALUES: seed every Q(s,a) ABOVE any
 * achievable return. Each visit then drives a cell's value DOWN, so the greedy
 * policy is automatically pulled toward the least-visited (s,a) -- a built-in,
 * systematic exploration drive that performs a breadth-first-like sweep until
 * it discovers the goal, after which the on-policy values shape a clean path.
 * We seed Q to RL_INT(100) (= 100.0 in Q8.8); every real return here is <= 0.
 *
 * One-call protocol note: when a previous (last_s,last_a) is pending we need a'
 * BEFORE we can learn. So in step() we first epsilon-greedily pick a' for the
 * current state s', THEN apply the SARSA update to Q(last_s,last_a) using that
 * Q(s',a'), THEN remember (s',a') and return a'. On `done` the bootstrap term
 * is dropped (target = reward_prev only).
 */
#ifndef ALGO_SARSA_H
#define ALGO_SARSA_H

#include <stdlib.h>
#include "rl_core.h"

typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *Q;                       /* [nstates * nact] Q8.8 */
    rl_fp alpha, gamma; int eps;    /* eps in per-mille */
    rl_rng rng;
    int last_s, last_a, have;
} sar_t;

static inline int sar__argmax(sar_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv){ bv = r[k]; b = k; }
    return b;
}
/* epsilon-greedy action choice for state s (explore==0 => pure greedy) */
static inline int sar__choose(sar_t *q, int s, int explore){
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps)
        return (int)(rl_rand(&q->rng) % q->nact);
    return sar__argmax(q, s);
}

static inline int sar_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    sar_t *q = (sar_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);

    /* terminal: flush the pending transition with no bootstrap, end episode */
    if (done){
        if (q->have){
            rl_fp *cell = &q->Q[(size_t)q->last_s*q->nact + q->last_a];
            *cell += rl_mul(q->alpha, reward_prev - *cell);     /* target = r */
        }
        q->have = 0;
        return -1;
    }

    /* choose a' for the CURRENT state s' under the behaviour policy first ... */
    int act = sar__choose(q, s, explore);

    /* ... then learn the pending (last_s,last_a) on-policy toward Q(s',a') */
    if (q->have){
        rl_fp *cell = &q->Q[(size_t)q->last_s*q->nact + q->last_a];
        rl_fp qnext = q->Q[(size_t)s*q->nact + act];            /* Q(s',a') */
        rl_fp target = reward_prev + rl_mul(q->gamma, qnext);
        *cell += rl_mul(q->alpha, target - *cell);
    }

    q->last_s = s; q->last_a = act; q->have = 1;
    return act;
}
static inline int  sar_greedy(rl_agent *a, const rl_fp *feat){
    sar_t *q = (sar_t*)a->ctx; return sar__argmax(q, rl_state_of(q->spec, q->nfeat, feat));
}
static inline void sar_seteps(rl_agent *a, int m){ ((sar_t*)a->ctx)->eps = m; }
static inline void sar_destroy(rl_agent *a){ sar_t *q = (sar_t*)a->ctx; free(q->Q); free(q); }

static inline rl_agent sarsa_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    sar_t *q = malloc(sizeof(sar_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    q->Q = malloc((size_t)q->nstates*nact*sizeof(rl_fp));
    /* OPTIMISTIC initial values: above any achievable return (all <= 0 here),
       which gives on-policy SARSA a built-in exploration drive (see header). */
    {
        rl_fp q0 = RL_INT(100);
        for (size_t i = 0; i < (size_t)q->nstates*nact; i++) q->Q[i] = q0;
    }
    q->alpha = RL_FRAC(1,4);    /* 0.25 */
    q->gamma = 243;             /* ~0.95 in Q8.8 */
    q->eps = 100; q->have = 0;
    rl_seed(&q->rng, seed);
    rl_agent a; a.name = "sarsa"; a.ctx = q;
    a.step = sar_step; a.act_greedy = sar_greedy;
    a.set_epsilon = sar_seteps; a.destroy = sar_destroy;
    return a;
}

#endif /* ALGO_SARSA_H */
