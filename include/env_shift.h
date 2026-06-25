/*
 * env_shift.h - a NON-STATIONARY "blocking" maze.
 * ===============================================
 * A wall spans a middle row with a single gap. The gap's position is switched
 * partway through training (call shift_set_phase). The agent's state is only
 * (row, col) -- it cannot see the phase -- so when the gap moves, the world has
 * silently changed underneath a policy it already learned. Reaching the goal
 * ends the episode; every step (including bumping a wall) is -1.
 *
 * What it stresses: ADAPTABILITY to non-stationarity. This is the classic
 * Dyna-Q+ "blocking" experiment (Sutton & Barto 8.3): MODEL-BASED methods
 * (Dyna-Q, Prioritized Sweeping) can be SLOWER to recover because their learned
 * model still "remembers" the old open gap and keeps planning the stale route
 * until reality is re-sampled, whereas model-free methods adapt straight from
 * new experience.
 *
 * Phase 0: gap near the RIGHT (short path to the top-right goal).
 * Phase 1: that gap is walled and a new gap opens on the LEFT (long detour).
 * Pure integer.
 */
#ifndef ENV_SHIFT_H
#define ENV_SHIFT_H

#include <stdlib.h>
#include "rl_core.h"

#define SHIFT_STEP_R RL_INT(-1)

typedef struct {
    int R, C, NS, start, goal_r, goal_c, goal, cur;
    int wall_row, gap0, gap1, phase;
    rl_feature spec[2];
} shift_ctx;

static inline int shift__wall(shift_ctx *w, int r, int c){
    int gap = w->phase ? w->gap1 : w->gap0;
    return (r == w->wall_row && c != gap);
}
static inline void shift__feat(shift_ctx *w, int s, rl_fp *f){ f[0]=s/w->C; f[1]=s%w->C; }

static inline void shift_reset(rl_env *e, rl_fp *f){
    shift_ctx *w=(shift_ctx*)e->ctx; w->cur=w->start; shift__feat(w, w->cur, f);
}
static inline rl_fp shift_step(rl_env *e, int a, rl_fp *f, int *done){
    shift_ctx *w=(shift_ctx*)e->ctx;
    int r=w->cur/w->C, c=w->cur%w->C, nr=r, nc=c;
    switch (a){ case 0: nr--; break; case 1: nc++; break; case 2: nr++; break; case 3: nc--; break; }
    if (nr<0) nr=0;
    if (nr>w->R-1) nr=w->R-1;
    if (nc<0) nc=0;
    if (nc>w->C-1) nc=w->C-1;
    if (shift__wall(w, nr, nc)){ nr=r; nc=c; }      /* blocked: stay put */
    int ns=nr*w->C+nc;
    *done = (nr==w->goal_r && nc==w->goal_c);
    w->cur=ns; shift__feat(w, ns, f); return SHIFT_STEP_R;
}
static inline void shift_make(rl_env *e, uint32_t seed){
    (void)seed;
    shift_ctx *w=malloc(sizeof(shift_ctx));
    w->R=6; w->C=9; w->NS=54;
    w->start=(w->R-1)*w->C + 0; w->goal_r=0; w->goal_c=w->C-1; w->goal=w->goal_c;
    w->wall_row=3; w->gap0=w->C-2; w->gap1=1; w->phase=0; w->cur=w->start;
    w->spec[0].lo=0; w->spec[0].hi=w->R-1; w->spec[0].bins=w->R;
    w->spec[1].lo=0; w->spec[1].hi=w->C-1; w->spec[1].bins=w->C;
    e->n_actions=4; e->n_features=2; e->spec=w->spec;
    e->episodic=1; e->max_steps=8*(w->R+w->C);
    e->ctx=w; e->reset=shift_reset; e->step=shift_step;
}
static inline void shift_set_phase(rl_env *e, int phase){ ((shift_ctx*)e->ctx)->phase = phase; }
static inline void shift_free(rl_env *e){ free(e->ctx); }

#endif /* ENV_SHIFT_H */
