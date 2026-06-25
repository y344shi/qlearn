/*
 * env_arena.h - the CONTEST ARENA: a slippery cliff-walk.
 * =======================================================
 * This environment is deliberately designed to DISTINGUISH RL algorithms, not
 * just to be solved. It is Sutton & Barto's Cliff Walking with one twist:
 * actions are STOCHASTIC ("slip") -- with probability `slip` the intended move
 * is replaced by a random one.
 *
 *   . . . . . . . .      row 0 (top)
 *   . . . . . . . .
 *   . . . . . . . .      row R-2  <- the "risky shortcut" hugs the cliff edge
 *   S C C C C C C G      row R-1  (C = cliff: -100 and back to start)
 *
 * Step reward -1; falling off the cliff -100 (and reset to start, episode
 * continues); reaching G ends the episode.
 *
 * Why it discriminates:
 *   - OFF-policy (Q-learning, Double-Q): learn the OPTIMAL but risky edge path.
 *     Under slip they fall a lot -> poor ONLINE return, many training falls.
 *   - ON-policy (SARSA, Expected-SARSA, SARSA(lambda)): learn a SAFE path away
 *     from the edge -> higher online/eval return when slip is present.
 *   - Planning (Dyna-Q) and traces / n-step: converge in FEWER episodes
 *     (sample efficiency).
 *   - Monte-Carlo: high variance, slower.
 * The scoreboard measures: greedy eval return (robustness), cliff-falls during
 * training (safety), online return and episodes-to-threshold (efficiency).
 *
 * Pure integer.
 */
#ifndef ENV_ARENA_H
#define ENV_ARENA_H

#include <stdlib.h>
#include "rl_core.h"

#define ARENA_STEP_R  RL_INT(-1)
#define ARENA_CLIFF_R RL_INT(-100)

typedef struct {
    int R, C, NS, start, goal_r, goal_c, goal, cur;
    int slip;                 /* slip probability, per-mille */
    long long falls;          /* cumulative cliff falls (for the safety metric) */
    rl_rng rng;
    rl_feature spec[2];
    int max_steps;
} arena_ctx;

static inline int arena__is_cliff(arena_ctx *w, int r, int c){
    return r == w->R-1 && c >= 1 && c <= w->C-2;
}
static inline void arena__feat(arena_ctx *w, int s, rl_fp *f){ f[0] = s/w->C; f[1] = s%w->C; }

static inline void arena_reset(rl_env *e, rl_fp *f){
    arena_ctx *w = (arena_ctx*)e->ctx; w->cur = w->start; arena__feat(w, w->cur, f);
}
static inline rl_fp arena_step(rl_env *e, int a, rl_fp *f, int *done){
    arena_ctx *w = (arena_ctx*)e->ctx;
    if ((int)(rl_rand(&w->rng) % 1000) < w->slip) a = (int)(rl_rand(&w->rng) % 4);
    int r = w->cur/w->C, c = w->cur%w->C, nr = r, nc = c;
    switch (a){ case 0: nr--; break; case 1: nc++; break;
                case 2: nr++; break; case 3: nc--; break; }
    if (nr < 0) nr = 0;
    if (nr > w->R-1) nr = w->R-1;
    if (nc < 0) nc = 0;
    if (nc > w->C-1) nc = w->C-1;
    rl_fp rew; int ns;
    if (arena__is_cliff(w, nr, nc)){ rew = ARENA_CLIFF_R; w->falls++; ns = w->start; *done = 0; }
    else if (nr == w->goal_r && nc == w->goal_c){ rew = ARENA_STEP_R; ns = w->goal; *done = 1; }
    else { rew = ARENA_STEP_R; ns = nr*w->C+nc; *done = 0; }
    w->cur = ns; arena__feat(w, ns, f);
    return rew;
}
static inline void arena_make(rl_env *e, int rows, int cols, int slip_milli, uint32_t seed){
    arena_ctx *w = malloc(sizeof(arena_ctx));
    if (rows < 3) rows = 3;
    if (cols < 3) cols = 3;
    w->R = rows; w->C = cols; w->NS = rows*cols;
    w->start = (rows-1)*cols; w->goal_r = rows-1; w->goal_c = cols-1; w->goal = w->goal_r*cols+w->goal_c;
    w->slip = slip_milli; w->falls = 0;
    rl_seed(&w->rng, seed);
    w->spec[0].lo = 0; w->spec[0].hi = rows-1; w->spec[0].bins = rows;
    w->spec[1].lo = 0; w->spec[1].hi = cols-1; w->spec[1].bins = cols;
    w->max_steps = 8*(rows+cols);
    e->n_actions = 4; e->n_features = 2; e->spec = w->spec;
    e->episodic = 1; e->max_steps = w->max_steps;
    e->ctx = w; e->reset = arena_reset; e->step = arena_step;
}
static inline long long arena_falls(rl_env *e){ return ((arena_ctx*)e->ctx)->falls; }
static inline void      arena_free(rl_env *e){ free(e->ctx); }

/* Greedy evaluation: average return + success% + falls over K episodes (still
   stochastic, so this rewards robust policies that stay away from the edge).  */
typedef struct { int avg_return; int success_pct; long long falls; } arena_result;
static inline arena_result arena_eval(rl_env *e, rl_agent *ag, int episodes){
    arena_ctx *w = (arena_ctx*)e->ctx;
    long long retsum = 0, f0 = w->falls; int succ = 0;
    for (int ep = 0; ep < episodes; ep++){
        rl_fp feat[2]; e->reset(e, feat); int steps = 0, done = 0; long long ret = 0;
        while (steps < e->max_steps && !done){
            int a = ag->act_greedy(ag, feat);
            ret += e->step(e, a, feat, &done);
            steps++;
        }
        retsum += ret; if (done) succ++;
    }
    arena_result R;
    R.avg_return = (int)(retsum / episodes / RL_FP_ONE);
    R.success_pct = succ * 100 / episodes;
    R.falls = w->falls - f0;
    return R;
}

#endif /* ENV_ARENA_H */
