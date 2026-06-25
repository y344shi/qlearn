/*
 * algo_qlambda.h - Watkins's Q(lambda): off-policy TD control with eligibility
 * ===========================================================================
 * traces, wrapped in the uniform rl_agent vtable (see rl_core.h) using
 * INTEGER-ONLY Q8.8 math. Structure mirrors algo_qlearn.h.
 *
 * One-step Q-learning credits ONLY the most recent (s,a) for a TD error. Watkins
 * Q(lambda) (Sutton & Barto, 2nd ed., S12.10) keeps a short-term memory -- an
 * ELIGIBILITY TRACE e(x,b) for every state-action pair -- so a single TD error
 * is propagated back to ALL recently visited pairs, weighted by how recently and
 * how often they occurred. This blends one-step bootstrapping with Monte-Carlo
 * style multi-step credit assignment (the lambda-return view) and usually learns
 * a great deal faster.
 *
 * Per step, with current (s,a) -> reward r -> next state s':
 *     delta = r + gamma * max_a' Q(s',a') - Q(s,a)        (0 bootstrap if done)
 *     e(s,a) = RL_INT(1)                                  (replacing trace)
 *     for ALL (x,b):  Q(x,b) += alpha * delta * e(x,b)
 *                     e(x,b) *= gamma * lambda            (decay)
 *
 * WATKINS'S CUTOFF (the off-policy correction): the trace is only valid while the
 * behaviour policy is greedy. The instant an EXPLORATORY (non-greedy) action is
 * taken, the multi-step return is no longer an estimate of the greedy/optimal
 * return, so ALL traces are ZEROED after the update. On a greedy action the
 * traces keep decaying. On `done`: bootstrap term = 0 and clear all traces.
 *
 * Update:  Q(x,b) += alpha * delta * e(x,b);  e *= gamma*lambda  (cut on explore)
 */
#ifndef ALGO_QLAMBDA_H
#define ALGO_QLAMBDA_H

#include <stdlib.h>
#include "rl_core.h"

typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *Q;                       /* [nstates * nact] Q8.8 action values */
    rl_fp *E;                       /* [nstates * nact] Q8.8 eligibility traces */
    rl_fp alpha, gamma, lambda;     /* Q8.8 */
    rl_fp glam;                     /* gamma*lambda, precomputed Q8.8 */
    int eps;                        /* exploration rate, per-mille */
    rl_rng rng;
    int last_s, last_a, have;
} qlam_t;

static inline int qlam__argmax(qlam_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv){ bv = r[k]; b = k; }
    return b;
}
static inline rl_fp qlam__maxq(qlam_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv) bv = r[k];
    return bv;
}
static inline void qlam__clear_traces(qlam_t *q){
    size_t n = (size_t)q->nstates * q->nact;
    for (size_t i = 0; i < n; i++) q->E[i] = 0;
}

/* Apply TD error `delta` over the whole trace, then decay (or cut) the traces. */
static inline void qlam__update(qlam_t *q, rl_fp delta, int cut){
    size_t n = (size_t)q->nstates * q->nact;
    if (cut){
        /* exploratory next action: spend the error, then zero every trace */
        for (size_t i = 0; i < n; i++){
            if (q->E[i]) q->Q[i] += rl_mul(q->alpha, rl_mul(delta, q->E[i]));
            q->E[i] = 0;
        }
    } else {
        for (size_t i = 0; i < n; i++){
            if (q->E[i]){
                q->Q[i] += rl_mul(q->alpha, rl_mul(delta, q->E[i]));
                q->E[i]  = rl_mul(q->glam, q->E[i]);     /* decay by gamma*lambda */
            }
        }
    }
}

static inline int qlam_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    qlam_t *q = (qlam_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);

    /* terminal: flush pending transition with no bootstrap, clear traces, end. */
    if (done){
        if (q->have){
            rl_fp cell = q->Q[(size_t)q->last_s*q->nact + q->last_a];
            rl_fp delta = reward_prev - cell;            /* target = r, no bootstrap */
            q->E[(size_t)q->last_s*q->nact + q->last_a] = RL_INT(1);
            qlam__update(q, delta, 1);                   /* episode over -> cut traces */
        }
        qlam__clear_traces(q);
        q->have = 0;
        return -1;
    }

    /* choose the action for s' under the behaviour policy (greedy vs explore) */
    int greedy_a = qlam__argmax(q, s);
    int act = greedy_a, exploratory = 0;
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps){
        act = (int)(rl_rand(&q->rng) % q->nact);
        if (act != greedy_a) exploratory = 1;            /* a true non-greedy pick */
    }

    /* learn the pending (last_s,last_a) off-policy via max_a' Q(s',a') */
    if (q->have){
        rl_fp cell = q->Q[(size_t)q->last_s*q->nact + q->last_a];
        rl_fp nmax = qlam__maxq(q, s);                   /* bootstrap off greedy s' */
        rl_fp delta = reward_prev + rl_mul(q->gamma, nmax) - cell;
        q->E[(size_t)q->last_s*q->nact + q->last_a] = RL_INT(1);  /* replacing trace */
        qlam__update(q, delta, exploratory);             /* Watkins cutoff on explore */
    }

    q->last_s = s; q->last_a = act; q->have = 1;
    return act;
}

static inline int qlam_greedy(rl_agent *a, const rl_fp *feat){
    qlam_t *q = (qlam_t*)a->ctx;
    return qlam__argmax(q, rl_state_of(q->spec, q->nfeat, feat));
}
static inline void qlam_seteps(rl_agent *a, int m){ ((qlam_t*)a->ctx)->eps = m; }
static inline void qlam_destroy(rl_agent *a){
    qlam_t *q = (qlam_t*)a->ctx; free(q->E); free(q->Q); free(q);
}

static inline rl_agent qlambda_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    qlam_t *q = malloc(sizeof(qlam_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    q->Q = calloc((size_t)q->nstates*nact, sizeof(rl_fp));
    q->E = calloc((size_t)q->nstates*nact, sizeof(rl_fp));
    q->alpha  = RL_FRAC(1,5);       /* 0.2  */
    q->gamma  = 243;                /* ~0.95 in Q8.8 */
    q->lambda = RL_FRAC(7,10);      /* 0.7  */
    q->glam   = rl_mul(q->gamma, q->lambda);
    q->eps = 100; q->have = 0;
    rl_seed(&q->rng, seed);
    rl_agent a; a.name = "qlambda"; a.ctx = q;
    a.step = qlam_step; a.act_greedy = qlam_greedy;
    a.set_epsilon = qlam_seteps; a.destroy = qlam_destroy;
    return a;
}

#endif /* ALGO_QLAMBDA_H */
