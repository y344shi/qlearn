/*
 * algo_dynaq.h - Dyna-Q: integrated planning, acting and learning as an
 * rl_agent (Sutton & Barto, 2nd ed., §8.2). INTEGER-ONLY Q8.8 math.
 * =====================================================================
 * Structure copied from algo_qlearn.h. Dyna-Q augments one-step tabular
 * Q-learning with a learned, deterministic ENVIRONMENT MODEL and a fixed
 * number of background PLANNING updates per real step.
 *
 * Real step  (direct RL): on a sampled transition (s,a,r,s') apply the usual
 *   off-policy Q-learning update
 *       Q(s,a) += alpha * ( r + gamma * max_a' Q(s',a') - Q(s,a) )
 *   ...AND record it in the model:  model_r[s*nact+a]=r, model_s2[...]=s',
 *   marking (s,a) visited (deterministic-model assumption).
 *
 * Planning (model-based RL): repeat N times -- sample a previously visited
 *   (s,a) UNIFORMLY from the seen-list, look up the simulated (r,s') from the
 *   model, and apply the SAME Q-learning update. These cheap, purely internal
 *   updates propagate reward information across the table far faster than waiting
 *   for the agent to physically revisit those transitions, which is what gives
 *   Dyna-Q its sample efficiency.
 *
 * Terminal handling: a transition into a terminal state stores done=1 in the
 * model so both real and planned updates drop the bootstrap term (target = r).
 *
 * Behaviour policy is epsilon-greedy on Q; act_greedy = argmax Q.
 *
 * Fixed point: identical conventions to algo_qlearn.h -- alpha/gamma are Q8.8,
 * rl_mul does the Q8.8 multiply, model_r holds the reward already in Q8.8.
 */
#ifndef ALGO_DYNAQ_H
#define ALGO_DYNAQ_H

#include <stdlib.h>
#include "rl_core.h"

#ifndef DYNAQ_PLAN_STEPS
#define DYNAQ_PLAN_STEPS 10        /* N background planning updates per real step */
#endif

/* Optimistic initial value (Q8.8): makes untried (s,a) look attractive so the
 * sparse maze frontier gets explored; tiny next to the phone task's rewards. */
#ifndef DYNAQ_OPT_INIT
#define DYNAQ_OPT_INIT RL_INT(4)
#endif

typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *Q;                       /* [nstates * nact] Q8.8 */
    /* learned deterministic model, indexed [s*nact + a] */
    rl_fp *model_r;                 /* reward (Q8.8) */
    int   *model_s2;                /* next state */
    int   *model_done;              /* 1 if (s,a) led to terminal */
    int   *seen;                    /* 1 if (s,a) has ever been taken */
    int   *seen_list;               /* flat (s*nact+a) ids of visited pairs */
    int    nseen;                   /* number of distinct visited pairs */
    int    nplan;                   /* planning steps N */
    rl_fp  alpha, gamma; int eps;   /* eps in per-mille */
    rl_rng rng;
    int last_s, last_a, have;
} dyq_t;

static inline int dyq__argmax(dyq_t *q, int s){
    const rl_fp *r = &q->Q[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv){ bv = r[k]; b = k; }
    return b;
}
static inline rl_fp dyq__maxq(dyq_t *q, int s){
    const rl_fp *r = &q->Q[(size_t)s*q->nact]; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv) bv = r[k];
    return bv;
}

/* one Q-learning update of Q(s,a) toward r + gamma*max_a' Q(s2,a') (done=no boot) */
static inline void dyq__update(dyq_t *q, int s, int a, rl_fp r, int s2, int done){
    rl_fp *cell = &q->Q[(size_t)s*q->nact + a];
    rl_fp nmax = done ? 0 : dyq__maxq(q, s2);
    rl_fp target = r + rl_mul(q->gamma, nmax);
    *cell += rl_mul(q->alpha, target - *cell);
}

/* record a real transition into the deterministic model + seen-list */
static inline void dyq__model_learn(dyq_t *q, int s, int a, rl_fp r, int s2, int done){
    size_t i = (size_t)s*q->nact + a;
    q->model_r[i] = r; q->model_s2[i] = s2; q->model_done[i] = done;
    if (!q->seen[i]){ q->seen[i] = 1; q->seen_list[q->nseen++] = (int)i; }
}

/* N planning sweeps: replay random visited (s,a) from the model */
static inline void dyq__plan(dyq_t *q){
    if (q->nseen == 0) return;
    for (int n = 0; n < q->nplan; n++){
        int j = (int)(rl_rand(&q->rng) % (uint32_t)q->nseen);
        int id = q->seen_list[j];
        int s = id / q->nact, a = id % q->nact;
        dyq__update(q, s, a, q->model_r[id], q->model_s2[id], q->model_done[id]);
    }
}

static inline int dyq_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    dyq_t *q = (dyq_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);
    if (q->have){                                   /* learn from previous (s,a) */
        /* (1) direct RL on the real transition */
        dyq__update(q, q->last_s, q->last_a, reward_prev, s, done);
        /* (2) model learning */
        dyq__model_learn(q, q->last_s, q->last_a, reward_prev, s, done);
        /* (3) planning: N simulated Q-learning updates */
        dyq__plan(q);
    }
    if (done){ q->have = 0; return -1; }
    int act;
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps) act = (int)(rl_rand(&q->rng) % q->nact);
    else act = dyq__argmax(q, s);
    q->last_s = s; q->last_a = act; q->have = 1;
    return act;
}
static inline int  dyq_greedy(rl_agent *a, const rl_fp *feat){
    dyq_t *q = (dyq_t*)a->ctx; return dyq__argmax(q, rl_state_of(q->spec, q->nfeat, feat));
}
static inline void dyq_seteps(rl_agent *a, int m){ ((dyq_t*)a->ctx)->eps = m; }
static inline void dyq_destroy(rl_agent *a){
    dyq_t *q = (dyq_t*)a->ctx;
    free(q->Q); free(q->model_r); free(q->model_s2); free(q->model_done);
    free(q->seen); free(q->seen_list); free(q);
}

static inline rl_agent dynaq_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    dyq_t *q = malloc(sizeof(dyq_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    size_t n = (size_t)q->nstates * nact;
    q->Q        = malloc(n * sizeof(rl_fp));
    q->model_r  = calloc(n, sizeof(rl_fp));
    q->model_s2 = calloc(n, sizeof(int));
    q->model_done = calloc(n, sizeof(int));
    q->seen      = calloc(n, sizeof(int));
    q->seen_list = malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) q->Q[i] = DYNAQ_OPT_INIT;
    q->nseen = 0; q->nplan = DYNAQ_PLAN_STEPS;
    /* alpha = 24/256 ~= 0.094: gentle enough that the optimistic seed and the
     * stochastic phone task stay stable while planning compounds the updates. */
    q->alpha = 24; q->gamma = 243; q->eps = 100; q->have = 0;
    rl_seed(&q->rng, seed);
    rl_agent a; a.name = "dynaq"; a.ctx = q;
    a.step = dyq_step; a.act_greedy = dyq_greedy;
    a.set_epsilon = dyq_seteps; a.destroy = dyq_destroy;
    return a;
}

#endif /* ALGO_DYNAQ_H */
