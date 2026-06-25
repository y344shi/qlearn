/*
 * env_windy.h - the "windy maze" task as a reusable rl_env.
 * =========================================================
 * Arbitrary R x C gridworld with random mines and per-column upward wind
 * (generalised Cliff-Walking + Windy-Gridworld). Deterministic dynamics.
 * State features = (row, col); actions = up/right/down/left. Episodic: stepping
 * on a mine costs -100 and resets to start; reaching the goal (+ -1/step) ends
 * the episode. Pure integer.
 */
#ifndef ENV_WINDY_H
#define ENV_WINDY_H

#include <stdlib.h>
#include "rl_core.h"

#define WINDY_STEP_R  RL_INT(-1)
#define WINDY_MINE_R  RL_INT(-100)

typedef struct {
    int R, C, NS, start, goal_r, goal_c, goal, cur;
    int *wind, *mine;
    int optimal_len;            /* BFS shortest mine-free path (steps) */
    rl_feature spec[2];
    rl_rng rng;
} windy_ctx;

static inline int windy__is_mine(windy_ctx *w, int r, int c){ return w->mine[r*w->C+c]; }

/* deterministic transition from a flat state under action a */
static inline int windy__next(windy_ctx *w, int s, int a, rl_fp *reward, int *done){
    int r = s / w->C, c = s % w->C, nr = r, nc = c;
    switch (a){ case 0: nr--; break; case 1: nc++; break;
                case 2: nr++; break; case 3: nc--; break; }
    if (nr < 0) nr = 0;
    if (nr > w->R-1) nr = w->R-1;
    if (nc < 0) nc = 0;
    if (nc > w->C-1) nc = w->C-1;
    nr -= w->wind[nc];
    if (nr < 0) nr = 0;
    if (windy__is_mine(w, nr, nc)){ *reward = WINDY_MINE_R; *done = 0; return w->start; }
    if (nr == w->goal_r && nc == w->goal_c){ *reward = WINDY_STEP_R; *done = 1; return nr*w->C+nc; }
    *reward = WINDY_STEP_R; *done = 0; return nr*w->C+nc;
}

/* BFS shortest path length start->goal over the deterministic graph */
static inline int windy__bfs(windy_ctx *w){
    int *dist = malloc(sizeof(int)*w->NS), *q = malloc(sizeof(int)*w->NS);
    for (int i = 0; i < w->NS; i++) dist[i] = -1;
    int head = 0, tail = 0; dist[w->start] = 0; q[tail++] = w->start;
    int ans = -1;
    while (head < tail){
        int s = q[head++];
        if (s == w->goal){ ans = dist[s]; break; }
        for (int a = 0; a < 4; a++){
            rl_fp rr; int dd; int s2 = windy__next(w, s, a, &rr, &dd);
            if (s2 == s) continue;
            if (dist[s2] < 0){ dist[s2] = dist[s]+1; q[tail++] = s2; }
        }
    }
    free(dist); free(q);
    return ans;
}

static inline void windy__feat(windy_ctx *w, int s, rl_fp *f){ f[0] = s / w->C; f[1] = s % w->C; }

static inline void windy_reset(rl_env *e, rl_fp *f){
    windy_ctx *w = (windy_ctx*)e->ctx; w->cur = w->start; windy__feat(w, w->cur, f);
}
static inline rl_fp windy_step(rl_env *e, int a, rl_fp *f, int *done){
    windy_ctx *w = (windy_ctx*)e->ctx;
    rl_fp r; int s2 = windy__next(w, w->cur, a, &r, done); w->cur = s2; windy__feat(w, s2, f);
    return r;
}

/* Build a solvable windy-mine maze. Returns 0 on success. */
static inline int windy_make(rl_env *e, int rows, int cols, int n_mines, uint32_t seed){
    windy_ctx *w = malloc(sizeof(windy_ctx));
    if (rows < 2) rows = 2;
    if (cols < 2) cols = 2;
    w->R = rows; w->C = cols; w->NS = rows*cols;
    w->start = (rows-1)*cols; w->goal_r = rows-1; w->goal_c = cols-1; w->goal = w->goal_r*cols + w->goal_c;
    w->wind = malloc(sizeof(int)*cols); w->mine = malloc(sizeof(int)*w->NS);
    rl_seed(&w->rng, seed);
    if (n_mines > w->NS - 3) n_mines = w->NS - 3;
    if (n_mines < 0) n_mines = 0;
    for (int attempt = 0; attempt < 100000; attempt++){
        for (int c = 0; c < cols; c++){ uint32_t z = rl_rand(&w->rng)%100; w->wind[c] = z<12?2:z<38?1:0; }
        for (int i = 0; i < w->NS; i++) w->mine[i] = 0;
        int placed = 0, guard = 0;
        while (placed < n_mines && guard < n_mines*60 + 100){
            int s = rl_rand(&w->rng) % w->NS; guard++;
            if (s == w->start || s == w->goal || w->mine[s]) continue;
            w->mine[s] = 1; placed++;
        }
        w->optimal_len = windy__bfs(w);
        if (w->optimal_len >= 0) break;
    }
    w->spec[0].lo = 0; w->spec[0].hi = rows-1; w->spec[0].bins = rows;
    w->spec[1].lo = 0; w->spec[1].hi = cols-1; w->spec[1].bins = cols;
    e->n_actions = 4; e->n_features = 2; e->spec = w->spec;
    e->episodic = 1; e->max_steps = 6*(rows+cols) < 100 ? 100 : 6*(rows+cols);
    e->ctx = w; e->reset = windy_reset; e->step = windy_step;
    return 0;
}
static inline int  windy_optimal_len(rl_env *e){ return ((windy_ctx*)e->ctx)->optimal_len; }
static inline void windy_free(rl_env *e){ windy_ctx *w = e->ctx; free(w->wind); free(w->mine); free(w); }

#endif /* ENV_WINDY_H */
