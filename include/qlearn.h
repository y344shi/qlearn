/*
 * qlearn.h - general-purpose, dependency-free, PURE-INTEGER Q-learning agent.
 * ===========================================================================
 *
 * A tabular Q-learning agent that is completely decoupled from any particular
 * environment. You describe your state as a vector of integer FEATURES; the
 * agent quantises each feature into bins and indexes a Q-table. It supports any
 * number of features (any dimensionality) and any number of actions.
 *
 * Everything is integer arithmetic (Q8.8 fixed point) and there is no libc
 * dependency beyond <stdint.h> -- suitable for embedded / kernel use. The
 * caller owns the Q-table memory (no malloc inside), so it works with static
 * buffers too.
 *
 * --- Header-only ---------------------------------------------------------
 * Just `#include "qlearn.h"`. All functions are `static inline`; include it in
 * as many translation units as you like.
 *
 * --- Two ways to drive it -------------------------------------------------
 *  (1) Explicit, for episodic / offline training (you manage s,a,r,s'):
 *        int s  = qlearn_state(&ag, features);
 *        int a  = qlearn_select(&ag, features);     // epsilon-greedy
 *        ... apply a, observe reward and next features ...
 *        qlearn_update(&ag, features, a, reward_fp, next_features, done);
 *
 *  (2) Online one-call loop, for a live controller like a CPU-freq tuner:
 *        int a = qlearn_step(&ag, reward_for_previous_action, features, done);
 *        ... apply action a (e.g. set frequency) ...
 *      Call it once per control tick. It learns from the previous action and
 *      returns the next action. Pass reward 0 on the very first call.
 *
 * Rewards and Q-values are Q8.8 fixed point: use QL_INT(n) for an integer
 * reward n, or QL_FRAC(num,den) for a fraction.
 */
#ifndef QLEARN_H
#define QLEARN_H

#include <stdint.h>
#include <stddef.h>

/* ----------------------------- fixed point ------------------------------ */
typedef int32_t ql_fp;                 /* a real value v stored as v * 256   */
#define QL_FP_SHIFT 8
#define QL_FP_ONE   256
#define QL_INT(n)        ((ql_fp)((n) * QL_FP_ONE))            /* n.0        */
#define QL_FRAC(num,den) ((ql_fp)(((num) * QL_FP_ONE) / (den)))/* num/den    */

static inline ql_fp ql__mul(ql_fp a, ql_fp b) {   /* (a*b) in Q8.8, rounded */
    int64_t p = (int64_t)a * b;
    return (ql_fp)(p >= 0 ? ((p + (QL_FP_ONE / 2)) >> QL_FP_SHIFT)
                          : -(((-p) + (QL_FP_ONE / 2)) >> QL_FP_SHIFT));
}

/* --------------------------- feature description ------------------------ */
/* One per input feature: values in [lo, hi] are split into `bins` buckets.  */
typedef struct {
    ql_fp lo, hi;      /* expected value range of this feature (raw integers) */
    int   bins;        /* number of discrete buckets (>= 1)                   */
} ql_feature;

#define QL_MAX_FEATURES 16

/* ------------------------------ the agent ------------------------------- */
typedef struct {
    int               n_features;
    int               n_actions;
    int               n_states;            /* product of all bins            */
    const ql_feature *spec;                /* caller-owned, must outlive agent*/
    ql_fp            *Q;                    /* [n_states * n_actions], caller-owned */

    ql_fp    alpha;                        /* learning rate  (Q8.8)          */
    ql_fp    gamma;                        /* discount       (Q8.8)          */
    int      epsilon_milli;                /* explore prob out of 1000       */
    uint32_t rng;                          /* xorshift32 state               */

    int last_state, last_action, have_last;/* for the online qlearn_step API */
} qlearn_t;

/* ------------------------------ internals ------------------------------- */
static inline uint32_t ql__rand(qlearn_t *a) {
    uint32_t x = a->rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    a->rng = x ? x : 0x9e3779b9u;
    return a->rng;
}
static inline int ql__bin(ql_fp x, ql_fp lo, ql_fp hi, int bins) {
    if (bins <= 1 || hi <= lo) return 0;
    if (x <= lo) return 0;
    if (x >= hi) return bins - 1;
    long b = (long)(x - lo) * bins / (hi - lo);
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    return (int)b;
}

/* number of states implied by a feature spec (product of bins) */
static inline int qlearn_state_count(const ql_feature *spec, int n_features) {
    long n = 1;
    for (int i = 0; i < n_features; i++)
        n *= (spec[i].bins > 0 ? spec[i].bins : 1);
    return (int)n;
}
/* Q-table element count = states * actions (use to size the caller buffer). */
static inline size_t qlearn_qcount(const ql_feature *spec, int n_features, int n_actions) {
    return (size_t)qlearn_state_count(spec, n_features) * (size_t)n_actions;
}

/* ------------------------------- set-up --------------------------------- */
/* Q must point to qlearn_qcount(...) ql_fp elements (zero-initialised here). */
static inline void qlearn_init(qlearn_t *a, const ql_feature *spec,
                               int n_features, int n_actions,
                               ql_fp *Q, uint32_t seed) {
    a->n_features = n_features;
    a->n_actions  = n_actions;
    a->spec       = spec;
    a->n_states   = qlearn_state_count(spec, n_features);
    a->Q          = Q;
    a->alpha      = QL_FRAC(1, 4);     /* 0.25 */
    a->gamma      = 243;               /* ~0.95 */
    a->epsilon_milli = 100;            /* 0.10 */
    a->rng        = seed ? seed : 0x2545F4914F6CDD1Du;
    a->have_last  = 0;
    size_t n = (size_t)a->n_states * a->n_actions;
    for (size_t i = 0; i < n; i++) Q[i] = 0;
}
static inline void qlearn_set_params(qlearn_t *a, ql_fp alpha, ql_fp gamma,
                                     int epsilon_milli) {
    a->alpha = alpha; a->gamma = gamma; a->epsilon_milli = epsilon_milli;
}
static inline void qlearn_set_epsilon(qlearn_t *a, int epsilon_milli) {
    a->epsilon_milli = epsilon_milli;
}
/* call between episodes (offline training) so no update bridges the boundary */
static inline void qlearn_reset_episode(qlearn_t *a) { a->have_last = 0; }

/* --------------------- feature vector  ->  state index ------------------ */
static inline int qlearn_state(const qlearn_t *a, const ql_fp *features) {
    int idx = 0;
    for (int i = 0; i < a->n_features; i++)
        idx = idx * a->spec[i].bins
            + ql__bin(features[i], a->spec[i].lo, a->spec[i].hi, a->spec[i].bins);
    return idx;
}

/* -------------------- index-based core (hot path) ----------------------- */
static inline ql_fp qlearn_maxq_idx(const qlearn_t *a, int s) {
    const ql_fp *row = &a->Q[(size_t)s * a->n_actions];
    ql_fp best = row[0];
    for (int k = 1; k < a->n_actions; k++) if (row[k] > best) best = row[k];
    return best;
}
static inline int qlearn_argmax_idx(const qlearn_t *a, int s) {
    const ql_fp *row = &a->Q[(size_t)s * a->n_actions];
    int best = 0; ql_fp bv = row[0];
    for (int k = 1; k < a->n_actions; k++) if (row[k] > bv) { bv = row[k]; best = k; }
    return best;
}
static inline int qlearn_select_idx(qlearn_t *a, int s) {     /* epsilon-greedy */
    if ((int)(ql__rand(a) % 1000) < a->epsilon_milli)
        return (int)(ql__rand(a) % a->n_actions);
    return qlearn_argmax_idx(a, s);
}
/* tabular TD(0) update for an explicit transition (s, action) -> (s_next).
   Returns the TD error (Bellman delta) used, handy for monitoring convergence.*/
static inline ql_fp qlearn_update_idx(qlearn_t *a, int s, int action,
                                      ql_fp reward, int s_next, int done) {
    ql_fp *q = &a->Q[(size_t)s * a->n_actions + action];
    ql_fp next_max = done ? 0 : qlearn_maxq_idx(a, s_next);
    ql_fp target   = reward + ql__mul(a->gamma, next_max);
    ql_fp td       = target - *q;
    *q += ql__mul(a->alpha, td);
    return td;
}

/* ----------------------- feature-vector wrappers ------------------------ */
static inline int qlearn_select(qlearn_t *a, const ql_fp *features) {
    return qlearn_select_idx(a, qlearn_state(a, features));
}
static inline int qlearn_greedy(const qlearn_t *a, const ql_fp *features) {
    return qlearn_argmax_idx(a, qlearn_state(a, features));
}
static inline ql_fp qlearn_value(const qlearn_t *a, const ql_fp *features) {
    return qlearn_maxq_idx(a, qlearn_state(a, features));
}
static inline ql_fp qlearn_update(qlearn_t *a, const ql_fp *features, int action,
                                  ql_fp reward, const ql_fp *next_features, int done) {
    return qlearn_update_idx(a, qlearn_state(a, features), action, reward,
                             qlearn_state(a, next_features), done);
}

/* ----------------------- online one-call controller --------------------- */
/* Learn from the previous action's reward, then choose+return the next action
   for the current `features`. Use this for a live control loop. On the very
   first call `reward_prev` is ignored (there is no previous action yet).      */
static inline int qlearn_step(qlearn_t *a, ql_fp reward_prev,
                              const ql_fp *features, int done) {
    int s = qlearn_state(a, features);
    if (a->have_last)
        qlearn_update_idx(a, a->last_state, a->last_action, reward_prev, s, done);
    int action = qlearn_select_idx(a, s);
    a->last_state = s; a->last_action = action; a->have_last = 1;
    return action;
}

#endif /* QLEARN_H */
