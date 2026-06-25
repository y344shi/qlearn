/*
 * rl_core.h - shared core for the integer RL algorithm zoo.
 * =========================================================
 * Common fixed-point (Q8.8) math, feature-vector quantisation, an integer PRNG,
 * and the generic ENVIRONMENT interface that every algorithm plugs into. This
 * lets all the algorithms (qlearn.h and its siblings) be tested on the exact
 * same windy-maze and phone-frequency tasks, for a fair comparison.
 *
 * Pure integer, no libc beyond <stdint.h>/<stdlib.h>. Include "rl_env.h" to get
 * this plus both concrete environments.
 */
#ifndef RL_CORE_H
#define RL_CORE_H

#include <stdint.h>

/* ----------------------------- fixed point ------------------------------ */
typedef int32_t rl_fp;                 /* value v stored as v * 256          */
#define RL_FP_SHIFT 8
#define RL_FP_ONE   256
#define RL_INT(n)        ((rl_fp)((n) * RL_FP_ONE))
#define RL_FRAC(num,den) ((rl_fp)(((num) * RL_FP_ONE) / (den)))

static inline rl_fp rl_mul(rl_fp a, rl_fp b) {     /* (a*b) in Q8.8, rounded */
    int64_t p = (int64_t)a * b;
    return (rl_fp)(p >= 0 ? ((p + (RL_FP_ONE/2)) >> RL_FP_SHIFT)
                          : -(((-p) + (RL_FP_ONE/2)) >> RL_FP_SHIFT));
}

/* ------------------------------- PRNG ----------------------------------- */
typedef struct { uint32_t s; } rl_rng;
static inline void     rl_seed(rl_rng *r, uint32_t s){ r->s = s ? s : 0x2545F491u; }
static inline uint32_t rl_rand(rl_rng *r){
    uint32_t x = r->s; x ^= x<<13; x ^= x>>17; x ^= x<<5;
    r->s = x ? x : 0x9e3779b9u; return r->s;
}

/* --------------------- feature spec & quantisation ---------------------- */
typedef struct { rl_fp lo, hi; int bins; } rl_feature;

static inline int rl_bin(rl_fp x, rl_fp lo, rl_fp hi, int bins){
    if (bins <= 1 || hi <= lo) return 0;
    if (x <= lo) return 0;
    if (x >= hi) return bins - 1;
    long b = (long)(x - lo) * bins / (hi - lo);
    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;
    return (int)b;
}
static inline int rl_state_count(const rl_feature *spec, int nfeat){
    long n = 1;
    for (int i = 0; i < nfeat; i++) n *= (spec[i].bins > 0 ? spec[i].bins : 1);
    return (int)n;
}
/* map a feature vector to a flat discrete state index (mixed radix) */
static inline int rl_state_of(const rl_feature *spec, int nfeat, const rl_fp *f){
    int idx = 0;
    for (int i = 0; i < nfeat; i++)
        idx = idx * spec[i].bins + rl_bin(f[i], spec[i].lo, spec[i].hi, spec[i].bins);
    return idx;
}

/* ------------------------- environment interface ------------------------ */
/* Generic episodic/continuing environment. An algorithm only ever calls
   reset() and step(); it never needs to know which task it is solving.
     reset(env, feat_out)          -> writes the start feature vector
     step(env, action, feat_out, done) -> applies action, writes next feature
                                          vector, sets *done, returns reward(Q8.8)
   `episodic`   : 1 = reset at episode end (maze), 0 = continuing stream (phone)
   `max_steps`  : per-episode step budget (maze) / convergence horizon hint
   `spec`,`n_features`,`n_actions` describe the state/action space.            */
typedef struct rl_env rl_env;
struct rl_env {
    int n_actions;
    int n_features;
    const rl_feature *spec;
    int episodic;
    int max_steps;
    void *ctx;
    void  (*reset)(rl_env *env, rl_fp *feat_out);
    rl_fp (*step)(rl_env *env, int action, rl_fp *feat_out, int *done);
};

/* --------------------------- AGENT interface ---------------------------- */
/* EVERY algorithm in the contest implements this uniform vtable, so one
   harness/scoreboard can drive all of them. The online one-call `step` makes
   on-policy (SARSA), off-policy (Q-learning), n-step, trace, planning (Dyna)
   and Monte-Carlo methods all expressible:

     step(agent, reward_prev, feat, done, explore):
        1. if a previous (s,a) is pending, LEARN from it using reward_prev and
           the current state `feat` (bootstrap is skipped when done==1).
        2. if done==1: end the episode (flush traces / MC returns), return -1.
        3. else: choose an action for `feat` (epsilon-greedy when explore==1,
           greedy when explore==0), remember (s,a), and return it.

   `act_greedy` is pure exploitation with NO learning (used for evaluation).
   `set_epsilon` sets the exploration rate in per-mille (0..1000).

   IMPORTANT (contest rule): implementations must be INTEGER-ONLY. No float or
   double anywhere -- use rl_fp (Q8.8) and rl_mul. */
typedef struct rl_agent rl_agent;
struct rl_agent {
    const char *name;
    void *ctx;
    int  (*step)(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore);
    int  (*act_greedy)(rl_agent *a, const rl_fp *feat);
    void (*set_epsilon)(rl_agent *a, int milli);
    void (*destroy)(rl_agent *a);
};

#endif /* RL_CORE_H */
