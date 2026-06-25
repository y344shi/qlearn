/*
 * algo_expsarsa.h - Expected SARSA as an rl_agent (integer-only, Q8.8).
 * =====================================================================
 * Expected SARSA (Sutton & Barto, 2nd ed., section 6.6). Like SARSA it is a
 * temporal-difference control method, but instead of bootstrapping off the
 * single realised next action a' it bootstraps off the EXPECTED next action
 * value under the current policy pi:
 *
 *   Q(s,a) <- Q(s,a) + alpha * [ r + gamma * SUM_a' pi(a'|s') Q(s',a') - Q(s,a) ]
 *
 * Under an epsilon-greedy target policy each action gets probability eps/nact,
 * and the greedy (argmax) action gets an extra (1-eps). Hence the expectation
 * collapses to a closed form we can compute in fixed point:
 *
 *   E[Q(s',.)] = eps_frac * mean_a Q(s',a) + (1 - eps_frac) * max_a Q(s',a)
 *
 * with eps_frac = eps/1000 stored in Q8.8 and mean = sum/nact. Averaging out
 * the next-action choice removes SARSA's sampling variance, so a larger alpha
 * is safe. The target policy used here is the SAME epsilon-greedy behaviour
 * policy (the on-policy variant); setting the target eps to 0 would recover
 * off-policy Q-learning, which is why Expected SARSA generalises both.
 *
 * INTEGER-ONLY: no float/double anywhere; all math via rl_fp / rl_mul.
 */
#ifndef ALGO_EXPSARSA_H
#define ALGO_EXPSARSA_H

#include <stdlib.h>
#include "rl_core.h"

typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *Q;                       /* [nstates * nact] Q8.8 */
    rl_fp alpha, gamma; int eps;    /* behaviour eps in per-mille (0..1000) */
    int target_eps;                 /* eps of the TARGET policy in the bootstrap
                                       expectation, per-mille. Kept small and
                                       fixed so heavy early behaviour-exploration
                                       does not flood every target with the mean
                                       (which mixes in -100 mine actions). A near
                                       greedy target makes this the lower-variance
                                       off-policy form of Expected SARSA. */
    rl_rng rng;
    int last_s, last_a, have;
} esa_t;

static inline int esa__argmax(esa_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv){ bv = r[k]; b = k; }
    return b;
}

/* Expected next-state value E[Q(s',.)] under the epsilon-greedy target policy,
 * computed entirely in Q8.8 integer fixed point. */
static inline rl_fp esa__expected(esa_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact];
    rl_fp bv = r[0]; int64_t sum = r[0];
    for (int k = 1; k < q->nact; k++){
        if (r[k] > bv) bv = r[k];
        sum += r[k];
    }
    /* mean_a Q(s',a) = sum / nact, kept in Q8.8 */
    rl_fp mean = (rl_fp)(sum / q->nact);
    /* The bootstrap uses the TARGET policy's epsilon, capped so a high
     * behaviour-exploration rate cannot make every target pessimistic. */
    int teps = q->target_eps < q->eps ? q->target_eps : q->eps;
    /* eps_frac in Q8.8 = (teps/1000) * 256 = teps*256/1000, rounded */
    rl_fp eps_frac = (rl_fp)(((int64_t)teps * RL_FP_ONE + 500) / 1000);
    /* greedy mass on the argmax action */
    rl_fp greedy_frac = RL_FP_ONE - eps_frac;
    return rl_mul(eps_frac, mean) + rl_mul(greedy_frac, bv);
}

static inline int esa_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    esa_t *q = (esa_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);
    if (q->have){                                   /* learn from previous (s,a) */
        rl_fp *cell = &q->Q[(size_t)q->last_s*q->nact + q->last_a];
        rl_fp nexp = done ? 0 : esa__expected(q, s);
        rl_fp target = reward_prev + rl_mul(q->gamma, nexp);
        *cell += rl_mul(q->alpha, target - *cell);
    }
    if (done){ q->have = 0; return -1; }
    int act;
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps) act = (int)(rl_rand(&q->rng) % q->nact);
    else act = esa__argmax(q, s);
    q->last_s = s; q->last_a = act; q->have = 1;
    return act;
}
static inline int  esa_greedy(rl_agent *a, const rl_fp *feat){
    esa_t *q = (esa_t*)a->ctx; return esa__argmax(q, rl_state_of(q->spec, q->nfeat, feat));
}
static inline void esa_seteps(rl_agent *a, int m){ ((esa_t*)a->ctx)->eps = m; }
static inline void esa_destroy(rl_agent *a){ esa_t *q = (esa_t*)a->ctx; free(q->Q); free(q); }

static inline rl_agent expsarsa_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    esa_t *q = malloc(sizeof(esa_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    q->Q = calloc((size_t)q->nstates*nact, sizeof(rl_fp));
    /* Lower variance than SARSA => a larger learning rate is stable. */
    q->alpha = RL_FRAC(1,4); q->gamma = 243; q->eps = 100;
    q->target_eps = 50; q->have = 0;
    rl_seed(&q->rng, seed);
    rl_agent a; a.name = "expsarsa"; a.ctx = q;
    a.step = esa_step; a.act_greedy = esa_greedy;
    a.set_epsilon = esa_seteps; a.destroy = esa_destroy;
    return a;
}

#endif /* ALGO_EXPSARSA_H */
