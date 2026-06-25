/*
 * algo_qlearn.h - REFERENCE algorithm: tabular Q-learning as an rl_agent.
 * ======================================================================
 * This is the template every contest algorithm follows. It wraps one-step
 * off-policy Q-learning in the uniform rl_agent vtable (see rl_core.h), using
 * INTEGER-ONLY Q8.8 math. Copy this structure for SARSA, Dyna-Q, etc.
 *
 * Update:  Q(s,a) += alpha * ( r + gamma * max_a' Q(s',a') - Q(s,a) )
 */
#ifndef ALGO_QLEARN_H
#define ALGO_QLEARN_H

#include <stdlib.h>
#include "rl_core.h"

typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *Q;                       /* [nstates * nact] Q8.8 */
    rl_fp alpha, gamma; int eps;    /* eps in per-mille */
    rl_rng rng;
    int last_s, last_a, have;
} qla_t;

static inline int qla__argmax(qla_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv){ bv = r[k]; b = k; }
    return b;
}
static inline rl_fp qla__maxq(qla_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv) bv = r[k];
    return bv;
}
static inline int qla_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    qla_t *q = (qla_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);
    if (q->have){                                   /* learn from previous (s,a) */
        rl_fp *cell = &q->Q[(size_t)q->last_s*q->nact + q->last_a];
        rl_fp nmax = done ? 0 : qla__maxq(q, s);
        rl_fp target = reward_prev + rl_mul(q->gamma, nmax);
        *cell += rl_mul(q->alpha, target - *cell);
    }
    if (done){ q->have = 0; return -1; }
    int act;
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps) act = (int)(rl_rand(&q->rng) % q->nact);
    else act = qla__argmax(q, s);
    q->last_s = s; q->last_a = act; q->have = 1;
    return act;
}
static inline int  qla_greedy(rl_agent *a, const rl_fp *feat){
    qla_t *q = (qla_t*)a->ctx; return qla__argmax(q, rl_state_of(q->spec, q->nfeat, feat));
}
static inline void qla_seteps(rl_agent *a, int m){ ((qla_t*)a->ctx)->eps = m; }
/* optional: override learning rate / discount (eps<0 leaves epsilon unchanged) */
static inline void qlearn_agent_set(rl_agent *a, rl_fp alpha, rl_fp gamma, int eps){
    qla_t *q = (qla_t*)a->ctx; q->alpha = alpha; q->gamma = gamma;
    if (eps >= 0) q->eps = eps;
}
static inline void qla_destroy(rl_agent *a){ qla_t *q = (qla_t*)a->ctx; free(q->Q); free(q); }

static inline rl_agent qlearn_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    qla_t *q = malloc(sizeof(qla_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    q->Q = calloc((size_t)q->nstates*nact, sizeof(rl_fp));
    q->alpha = RL_FRAC(1,4); q->gamma = 243; q->eps = 100; q->have = 0;
    rl_seed(&q->rng, seed);
    rl_agent a; a.name = "qlearn"; a.ctx = q;
    a.step = qla_step; a.act_greedy = qla_greedy;
    a.set_epsilon = qla_seteps; a.destroy = qla_destroy;
    return a;
}

#endif /* ALGO_QLEARN_H */
