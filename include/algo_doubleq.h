/*
 * algo_doubleq.h - Double Q-learning as an rl_agent (Sutton & Barto 6.7).
 * =======================================================================
 * Tabular Double Q-learning wrapped in the uniform rl_agent vtable (rl_core.h),
 * using INTEGER-ONLY Q8.8 math. Structure copied from algo_qlearn.h.
 *
 * Motivation: ordinary Q-learning bootstraps from max_a' Q(s',a'), where the
 * SAME estimates are used both to SELECT the maximising action and to EVALUATE
 * it. Because the estimates are noisy, E[max] >= max E, so this systematically
 * OVERESTIMATES action values (maximisation bias). Double Q-learning removes
 * the bias by keeping two independent estimators QA, QB and decoupling
 * selection from evaluation: one table picks the greedy action, the OTHER
 * scores it.
 *
 * Update (Hasselt 2010). On each learning step flip a fair coin:
 *   with prob 1/2 update QA:
 *     a* = argmax_a QA(s',a)
 *     QA(s,a) += alpha * ( r + gamma * QB(s', a*) - QA(s,a) )
 *   else update QB (symmetric: a* from QB, evaluate with QA).
 * Terminal (done): target = r (no bootstrap).
 *
 * Behaviour policy is epsilon-greedy on QA+QB; act_greedy = argmax (QA+QB).
 */
#ifndef ALGO_DOUBLEQ_H
#define ALGO_DOUBLEQ_H

#include <stdlib.h>
#include "rl_core.h"

typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *QA, *QB;                 /* each [nstates * nact] Q8.8 */
    rl_fp alpha, gamma; int eps;    /* eps in per-mille */
    rl_rng rng;
    int last_s, last_a, have;
} dq_t;

/* argmax over a single table's row */
static inline int dq__argmax_tbl(dq_t *q, const rl_fp *T, int s){
    const rl_fp *r = &T[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv){ bv = r[k]; b = k; }
    return b;
}
/* argmax over the summed estimator QA+QB (behaviour / greedy policy) */
static inline int dq__argmax_sum(dq_t *q, int s){
    const rl_fp *ra = &q->QA[(size_t)s*q->nact];
    const rl_fp *rb = &q->QB[(size_t)s*q->nact];
    int b = 0; rl_fp bv = ra[0] + rb[0];
    for (int k = 1; k < q->nact; k++){ rl_fp v = ra[k] + rb[k]; if (v > bv){ bv = v; b = k; } }
    return b;
}

static inline int dq_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    dq_t *q = (dq_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);
    if (q->have){                                   /* learn from previous (s,a) */
        /* fair coin from the integer PRNG: pick which estimator to update */
        if (rl_rand(&q->rng) & 1u){
            /* update QA: select with QA, evaluate with QB */
            rl_fp *cell = &q->QA[(size_t)q->last_s*q->nact + q->last_a];
            rl_fp boot = 0;
            if (!done){
                int astar = dq__argmax_tbl(q, q->QA, s);
                boot = q->QB[(size_t)s*q->nact + astar];
            }
            rl_fp target = reward_prev + rl_mul(q->gamma, boot);
            *cell += rl_mul(q->alpha, target - *cell);
        } else {
            /* update QB: select with QB, evaluate with QA */
            rl_fp *cell = &q->QB[(size_t)q->last_s*q->nact + q->last_a];
            rl_fp boot = 0;
            if (!done){
                int astar = dq__argmax_tbl(q, q->QB, s);
                boot = q->QA[(size_t)s*q->nact + astar];
            }
            rl_fp target = reward_prev + rl_mul(q->gamma, boot);
            *cell += rl_mul(q->alpha, target - *cell);
        }
    }
    if (done){ q->have = 0; return -1; }
    int act;
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps) act = (int)(rl_rand(&q->rng) % q->nact);
    else act = dq__argmax_sum(q, s);
    q->last_s = s; q->last_a = act; q->have = 1;
    return act;
}
static inline int  dq_greedy(rl_agent *a, const rl_fp *feat){
    dq_t *q = (dq_t*)a->ctx; return dq__argmax_sum(q, rl_state_of(q->spec, q->nfeat, feat));
}
static inline void dq_seteps(rl_agent *a, int m){ ((dq_t*)a->ctx)->eps = m; }
static inline void dq_destroy(rl_agent *a){ dq_t *q = (dq_t*)a->ctx; free(q->QA); free(q->QB); free(q); }

/* Optimistic initial value (Q8.8). Double Q-learning is an UNBIASED estimator,
 * so it lacks the optimistic "maximisation bias" that makes plain Q-learning
 * explore toward unseen frontier. We restore that directed exploration the
 * principled way: initialise both tables to a small positive value (+4) so
 * untried (s,a) look attractive and get visited. It is tiny next to the phone
 * task's large rewards (so harmless there) yet meaningful for the sparse maze. */
#define DOUBLEQ_OPT_INIT RL_INT(4)

static inline rl_agent doubleq_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    dq_t *q = malloc(sizeof(dq_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    size_t n = (size_t)q->nstates * nact;
    q->QA = malloc(n * sizeof(rl_fp));
    q->QB = malloc(n * sizeof(rl_fp));
    for (size_t i = 0; i < n; i++){ q->QA[i] = DOUBLEQ_OPT_INIT; q->QB[i] = DOUBLEQ_OPT_INIT; }
    /* alpha = 24/256 ~= 0.094: small enough that the optimistic seed decays
     * gradually (sustaining the sweep), large enough to track the DVFS task. */
    q->alpha = 24; q->gamma = 243; q->eps = 100; q->have = 0;
    rl_seed(&q->rng, seed);
    rl_agent a; a.name = "doubleq"; a.ctx = q;
    a.step = dq_step; a.act_greedy = dq_greedy;
    a.set_epsilon = dq_seteps; a.destroy = dq_destroy;
    return a;
}

#endif /* ALGO_DOUBLEQ_H */
