/*
 * algo_sarsalambda.h - SARSA(lambda): on-policy TD control with eligibility
 * traces, as an rl_agent.
 * =========================================================================
 * On-policy temporal-difference control with ELIGIBILITY TRACES (Sutton &
 * Barto, S12.7), wrapped in the uniform rl_agent vtable (see rl_core.h) using
 * INTEGER-ONLY Q8.8 math. Structure mirrors algo_sarsa.h; the substantive
 * additions are (1) a trace table e[s*nact+a] in Q8.8 and (2) a full sweep
 * over every (state,action) on each step that applies one TD error to ALL
 * states in proportion to how "eligible" (recently visited) each one is.
 *
 * One-step SARSA only credits the single just-taken (s,a). SARSA(lambda)
 * instead keeps a fading memory of every (s,a) it has visited this episode and
 * lets a single TD error update them all at once. lambda controls how far the
 * credit reaches back: lambda=0 reduces to one-step SARSA, lambda=1 is the
 * (on-policy) Monte-Carlo end. This propagates reward backward many steps per
 * update, so the agent converges much faster on long-horizon tasks.
 *
 * TD error (on-policy: bootstrap off Q(s',a') with a' the ACTUAL next action):
 *     delta = r + gamma * Q(s',a') - Q(s,a)              (r only when terminal)
 * Trace bump for the just-acted (s,a) (accumulating shown; replacing = set 1):
 *     e(s,a) += 1
 * Sweep over EVERY (x,b):
 *     Q(x,b) += alpha * delta * e(x,b)
 *     e(x,b) *= gamma * lambda
 *
 * One-call protocol note: because the method is on-policy we need a' BEFORE we
 * can compute delta. So in step() we first epsilon-greedily pick a' for the
 * current state s', THEN form delta against Q(s',a'), bump+sweep the traces,
 * THEN remember (s',a') and return a'. On `done` the bootstrap term is dropped
 * (target = reward_prev only) and the traces are cleared for the next episode.
 */
#ifndef ALGO_SARSALAMBDA_H
#define ALGO_SARSALAMBDA_H

#include <stdlib.h>
#include <string.h>
#include "rl_core.h"

typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *Q;                       /* [nstates * nact] Q8.8 */
    rl_fp *E;                       /* [nstates * nact] Q8.8 eligibility traces */
    rl_fp alpha, gamma, lambda;     /* all Q8.8 */
    int eps;                        /* eps in per-mille */
    int replacing;                  /* 1 = replacing traces, 0 = accumulating */
    rl_rng rng;
    int last_s, last_a, have;
} sarl_t;

static inline int sarl__argmax(sarl_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv){ bv = r[k]; b = k; }
    return b;
}
/* epsilon-greedy action choice for state s (explore==0 => pure greedy) */
static inline int sarl__choose(sarl_t *q, int s, int explore){
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps)
        return (int)(rl_rand(&q->rng) % q->nact);
    return sarl__argmax(q, s);
}

/* Apply one TD error delta to ALL (x,b) weighted by their trace, then decay
   every trace by gamma*lambda. This is the heart of SARSA(lambda). */
static inline void sarl__update_all(sarl_t *q, rl_fp delta){
    rl_fp glam = rl_mul(q->gamma, q->lambda);   /* gamma*lambda, Q8.8 */
    size_t n = (size_t)q->nstates * q->nact;
    for (size_t i = 0; i < n; i++){
        rl_fp e = q->E[i];
        if (e != 0){
            q->Q[i] += rl_mul(q->alpha, rl_mul(delta, e));
            q->E[i] = rl_mul(glam, e);
        }
    }
}

static inline int sarl_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    sarl_t *q = (sarl_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);

    /* terminal: flush the pending transition with no bootstrap, clear traces */
    if (done){
        if (q->have){
            rl_fp qsa = q->Q[(size_t)q->last_s*q->nact + q->last_a];
            rl_fp delta = reward_prev - qsa;            /* target = r only */
            rl_fp *e = &q->E[(size_t)q->last_s*q->nact + q->last_a];
            if (q->replacing) *e = RL_INT(1); else *e += RL_INT(1);
            sarl__update_all(q, delta);
        }
        memset(q->E, 0, (size_t)q->nstates*q->nact*sizeof(rl_fp));
        q->have = 0;
        return -1;
    }

    /* choose a' for the CURRENT state s' under the behaviour policy first ... */
    int act = sarl__choose(q, s, explore);

    /* ... then learn the pending (last_s,last_a) on-policy toward Q(s',a') */
    if (q->have){
        rl_fp qsa  = q->Q[(size_t)q->last_s*q->nact + q->last_a];
        rl_fp qnext = q->Q[(size_t)s*q->nact + act];    /* Q(s',a') */
        rl_fp delta = reward_prev + rl_mul(q->gamma, qnext) - qsa;
        rl_fp *e = &q->E[(size_t)q->last_s*q->nact + q->last_a];
        if (q->replacing) *e = RL_INT(1); else *e += RL_INT(1);
        sarl__update_all(q, delta);
    }

    q->last_s = s; q->last_a = act; q->have = 1;
    return act;
}
static inline int  sarl_greedy(rl_agent *a, const rl_fp *feat){
    sarl_t *q = (sarl_t*)a->ctx; return sarl__argmax(q, rl_state_of(q->spec, q->nfeat, feat));
}
static inline void sarl_seteps(rl_agent *a, int m){ ((sarl_t*)a->ctx)->eps = m; }
static inline void sarl_destroy(rl_agent *a){ sarl_t *q = (sarl_t*)a->ctx; free(q->Q); free(q->E); free(q); }

static inline rl_agent sarsalambda_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    sarl_t *q = malloc(sizeof(sarl_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    q->Q = calloc((size_t)q->nstates*nact, sizeof(rl_fp));
    q->E = calloc((size_t)q->nstates*nact, sizeof(rl_fp));
    q->alpha  = RL_FRAC(1,4);   /* 0.25 learning rate */
    q->gamma  = 255;            /* ~0.996 in Q8.8: a long horizon so the -1/step
                                   gradient reaches far enough to guide the agent
                                   to the distant goal on the windy maze */
    q->lambda = RL_FRAC(5,10);  /* 0.50 trace decay */
    q->eps = 100; q->replacing = 1; q->have = 0;
    rl_seed(&q->rng, seed);
    rl_agent a; a.name = "sarsa(lambda)"; a.ctx = q;
    a.step = sarl_step; a.act_greedy = sarl_greedy;
    a.set_epsilon = sarl_seteps; a.destroy = sarl_destroy;
    return a;
}

#endif /* ALGO_SARSALAMBDA_H */
