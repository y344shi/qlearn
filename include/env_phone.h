/*
 * env_phone.h - the "frequency tuning" task as a reusable rl_env.
 * ===============================================================
 * Simulated-phone DVFS: each frame a workload demands CPU work; the agent picks
 * a CPU frequency from the big-cluster table. Too low -> jank (missed deadline);
 * too high -> wasted power (~f^2). Continuing task. State features = (util,
 * refresh). Reward = avoid jank, then minimise power. Pure integer.
 *
 * Also exposes baseline governors + a metrics() scorer so any agent can be
 * compared against performance / powersave / ondemand / oracle.
 */
#ifndef ENV_PHONE_H
#define ENV_PHONE_H

#include "rl_core.h"

static const int PH_FREQ[] = { 1239, 1530, 1740, 1930, 2150, 2350, 2620 };
#define PH_NFREQ ((int)(sizeof(PH_FREQ)/sizeof(PH_FREQ[0])))
#define PH_FMAX 2620
static const int PH_BUDGET_US[] = { 16667, 11111, 8333, 6944 };  /* 60/90/120/144Hz */
#define PH_NREFRESH 4
#define PH_LOAD_BINS 24
#define PH_JANK_PENALTY 3000

static inline int ph_power(int f){ return 50 + (f*f)/10000; }
static inline int ph_util(int req){ long u = (long)req*1024/PH_FMAX; return (int)(u>1023?1023:u); }
static inline int ph_freq_ceil(int req){ for (int i=0;i<PH_NFREQ;i++) if (PH_FREQ[i]>=req) return i; return PH_NFREQ-1; }

typedef struct { int req_base, jitter, refresh, len; } ph_phase;
static const ph_phase PH_PHASES[] = {
    { 1239, 120, 0, 70 }, { 1740, 150, 0, 70 }, { 2150, 220, 2, 70 },
    { 2350, 260, 1, 70 }, { 2620, 350, 2, 50 },
};
#define PH_NPHASES ((int)(sizeof(PH_PHASES)/sizeof(PH_PHASES[0])))

typedef struct { rl_rng rng; int phase, left; } ph_gen;
static inline void ph_gen_init(ph_gen *g, uint32_t seed){ rl_seed(&g->rng, seed); g->phase = 0; g->left = 0; }
static inline void ph_gen_next(ph_gen *g, int *req, int *refresh){
    if (g->left <= 0){ g->phase = (rl_rand(&g->rng)>>8) % PH_NPHASES; g->left = PH_PHASES[g->phase].len; }
    const ph_phase *p = &PH_PHASES[g->phase];
    int j = (int)(rl_rand(&g->rng) % (2*p->jitter+1)) - p->jitter;
    int r = p->req_base + j; if (r < 600) r = 600;
    *req = r; *refresh = p->refresh; g->left--;
}

/* ------- environment (streaming, continuing) ------- */
typedef struct { ph_gen gen; uint32_t seed; int req, refresh; rl_feature spec[2]; } phone_ctx;

static inline void phone__feat(phone_ctx *p, rl_fp *f){ f[0] = ph_util(p->req); f[1] = p->refresh; }
static inline void phone_reset(rl_env *e, rl_fp *f){
    phone_ctx *p = (phone_ctx*)e->ctx; ph_gen_init(&p->gen, p->seed);
    ph_gen_next(&p->gen, &p->req, &p->refresh); phone__feat(p, f);
}
static inline rl_fp phone_step(rl_env *e, int a, rl_fp *f, int *done){
    phone_ctx *p = (phone_ctx*)e->ctx;
    if (a < 0) a = 0;
    if (a >= PH_NFREQ) a = PH_NFREQ-1;
    int budget = PH_BUDGET_US[p->refresh], fr = PH_FREQ[a];
    int exec = (p->req * budget) / fr, jank = exec > budget, pw = ph_power(fr);
    rl_fp reward = RL_INT(jank ? (-PH_JANK_PENALTY - pw) : (1000 - pw));
    ph_gen_next(&p->gen, &p->req, &p->refresh);     /* advance to next frame */
    phone__feat(p, f); *done = 0;
    return reward;
}
static inline void phone_make(rl_env *e, uint32_t seed){
    phone_ctx *p = malloc(sizeof(phone_ctx));
    p->seed = seed ? seed : 7;
    p->spec[0].lo = 0; p->spec[0].hi = 1023; p->spec[0].bins = PH_LOAD_BINS;
    p->spec[1].lo = 0; p->spec[1].hi = PH_NREFRESH-1; p->spec[1].bins = PH_NREFRESH;
    e->n_actions = PH_NFREQ; e->n_features = 2; e->spec = p->spec;
    e->episodic = 0; e->max_steps = 1;
    e->ctx = p; e->reset = phone_reset; e->step = phone_step;
}
static inline void phone_free(rl_env *e){ free(e->ctx); }

/* ------- scoring: run a frequency-picker over a fresh eval trace ------- */
typedef struct { int jank_ppm; long long energy; int avg_freq; } phone_metrics;
static inline phone_metrics phone_score(int (*pick)(int util, int refresh, void *c), void *pctx,
                                 int frames, uint32_t seed){
    ph_gen g; ph_gen_init(&g, seed);
    long long energy = 0, fsum = 0; int jank = 0;
    for (int t = 0; t < frames; t++){
        int req, refresh; ph_gen_next(&g, &req, &refresh);
        int budget = PH_BUDGET_US[refresh];
        int a = pick(ph_util(req), refresh, pctx);
        if (a < 0) a = 0;
        if (a >= PH_NFREQ) a = PH_NFREQ-1;
        int fr = PH_FREQ[a], exec = (req*budget)/fr;
        if (exec > budget) jank++;
        energy += (long long)ph_power(fr) * exec; fsum += fr;
    }
    phone_metrics m; m.jank_ppm = (int)((long long)jank*1000000/frames);
    m.energy = energy; m.avg_freq = (int)(fsum/frames); return m;
}
/* baseline governors */
static inline int phone_pick_perf(int u,int r,void*c){ (void)u;(void)r;(void)c; return PH_NFREQ-1; }
static inline int phone_pick_power(int u,int r,void*c){ (void)u;(void)r;(void)c; return 0; }
static inline int phone_pick_optimal(int u,int r,void*c){ (void)r;(void)c; return ph_freq_ceil((int)((long)u*PH_FMAX/1024)); }
static inline int phone_pick_ondemand(int u,int r,void*c){ (void)r;(void)c; return ph_freq_ceil((int)((long)u*PH_FMAX/1024*118/100)); }

/* adapter: score an rl_agent's greedy policy */
typedef struct { rl_agent *ag; } phone_agent_ctx;
static inline int phone_pick_agent(int u,int r,void*c){
    rl_agent *ag = ((phone_agent_ctx*)c)->ag; rl_fp f[2] = { (rl_fp)u, (rl_fp)r };
    return ag->act_greedy(ag, f);
}

#endif /* ENV_PHONE_H */
