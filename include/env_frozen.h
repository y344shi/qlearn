/*
 * env_frozen.h - "frozen lake": a big SLIPPERY hazard maze.
 * =========================================================
 * Harder than the cliff arena. The floor is slippery: with probability `slip`
 * the intended move is deflected 90 degrees (left or right). Holes are
 * scattered across the grid; stepping in one costs -100 and resets to start
 * (the episode keeps going). Reaching the goal ends the episode; every step is
 * -1. State = (row, col); 4 actions. Pure integer.
 *
 * What it stresses: robustness to stochastic transitions (a greedy edge-hugging
 * policy gets deflected into holes), plus harder exploration (a larger grid with
 * many traps). On-policy / safety-aware methods keep more clearance.
 */
#ifndef ENV_FROZEN_H
#define ENV_FROZEN_H

#include <stdlib.h>
#include "rl_core.h"

#define FROZEN_STEP_R RL_INT(-1)
#define FROZEN_HOLE_R RL_INT(-100)

typedef struct {
    int R, C, NS, start, goal_r, goal_c, goal, cur, slip;
    int *hole;
    long long deaths;
    rl_rng rng;
    rl_feature spec[2];
} frozen_ctx;

static inline int frozen__hole(frozen_ctx *w, int r, int c){ return w->hole[r*w->C+c]; }
static inline void frozen__feat(frozen_ctx *w, int s, rl_fp *f){ f[0] = s/w->C; f[1] = s%w->C; }

/* BFS solvability check over the deterministic (no-slip) graph avoiding holes */
static inline int frozen__reachable(frozen_ctx *w){
    int *seen = calloc(w->NS, sizeof(int)), *q = malloc(sizeof(int)*w->NS);
    int head=0, tail=0, ok=0; seen[w->start]=1; q[tail++]=w->start;
    const int dr[4]={-1,0,1,0}, dc[4]={0,1,0,-1};
    while (head<tail){
        int s=q[head++]; if (s==w->goal){ ok=1; break; }
        int r=s/w->C, c=s%w->C;
        for (int a=0;a<4;a++){
            int nr=r+dr[a], nc=c+dc[a];
            if (nr<0||nr>w->R-1||nc<0||nc>w->C-1) continue;
            int ns=nr*w->C+nc;
            if (w->hole[ns] || seen[ns]) continue;
            seen[ns]=1; q[tail++]=ns;
        }
    }
    free(seen); free(q); return ok;
}

static inline void frozen_reset(rl_env *e, rl_fp *f){
    frozen_ctx *w=(frozen_ctx*)e->ctx; w->cur=w->start; frozen__feat(w, w->cur, f);
}
static inline rl_fp frozen_step(rl_env *e, int a, rl_fp *f, int *done){
    frozen_ctx *w=(frozen_ctx*)e->ctx;
    if ((int)(rl_rand(&w->rng)%1000) < w->slip)        /* slip: deflect 90 deg */
        a = (rl_rand(&w->rng)&1) ? (a+1)&3 : (a+3)&3;
    int r=w->cur/w->C, c=w->cur%w->C, nr=r, nc=c;
    switch (a){ case 0: nr--; break; case 1: nc++; break; case 2: nr++; break; case 3: nc--; break; }
    if (nr<0) nr=0;
    if (nr>w->R-1) nr=w->R-1;
    if (nc<0) nc=0;
    if (nc>w->C-1) nc=w->C-1;
    rl_fp rew; int ns=nr*w->C+nc;
    if (w->hole[ns]){ rew=FROZEN_HOLE_R; w->deaths++; ns=w->start; *done=0; }
    else if (nr==w->goal_r && nc==w->goal_c){ rew=FROZEN_STEP_R; *done=1; }
    else { rew=FROZEN_STEP_R; *done=0; }
    w->cur=ns; frozen__feat(w, ns, f); return rew;
}
static inline void frozen_make(rl_env *e, int rows, int cols, int n_holes, int slip, uint32_t seed){
    frozen_ctx *w=malloc(sizeof(frozen_ctx));
    w->R=rows; w->C=cols; w->NS=rows*cols; w->slip=slip; w->deaths=0;
    w->start=0; w->goal_r=rows-1; w->goal_c=cols-1; w->goal=w->NS-1;
    w->hole=malloc(sizeof(int)*w->NS); rl_seed(&w->rng, seed);
    if (n_holes>w->NS-3) n_holes=w->NS-3;
    for (int att=0; att<100000; att++){
        for (int i=0;i<w->NS;i++) w->hole[i]=0;
        int placed=0, guard=0;
        while (placed<n_holes && guard<n_holes*60+100){
            int s=rl_rand(&w->rng)%w->NS; guard++;
            if (s==w->start || s==w->goal || w->hole[s]) continue;
            w->hole[s]=1; placed++;
        }
        if (frozen__reachable(w)) break;
    }
    w->spec[0].lo=0; w->spec[0].hi=rows-1; w->spec[0].bins=rows;
    w->spec[1].lo=0; w->spec[1].hi=cols-1; w->spec[1].bins=cols;
    e->n_actions=4; e->n_features=2; e->spec=w->spec;
    e->episodic=1; e->max_steps=10*(rows+cols);
    e->ctx=w; e->reset=frozen_reset; e->step=frozen_step;
}
static inline long long frozen_deaths(rl_env *e){ return ((frozen_ctx*)e->ctx)->deaths; }
static inline void frozen_free(rl_env *e){ frozen_ctx *w=e->ctx; free(w->hole); free(w); }

#endif /* ENV_FROZEN_H */
