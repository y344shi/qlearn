/*
 * env_swirl.h - "swirl escape": phone DVFS as physics, PARTIALLY OBSERVED.
 * ========================================================================
 * Unexpected SWIRLS hit a craft; it must THRUST >= the swirl's TRUE strength to
 * escape, or it is "caught" (a jank-like penalty). Thrust costs fuel, and we
 * simply REWARD using less of it (no hard tank -- minimal fuel is the soft cost,
 * exactly like minimizing power on a phone). Each step:
 *
 *     reward = -(caught ? CAUGHT : 0)  -  FUELW * thrust
 *
 * The twist that makes a learned policy beat the naive reactive one is PARTIAL
 * OBSERVABILITY: the true strength S is HIDDEN; the agent sees only a noisy
 * sensor reading O = S +/- 1 (like estimating next-frame CPU workload). Thrust
 * "O" (match the reading) is caught whenever the sensor under-reads. The optimal
 * policy HEDGES -- thrust a margin above the reading -- trading a little fuel for
 * far fewer catches. The agent learns that margin per reading.
 *
 * Phone mapping: reading = estimated workload, thrust = CPU frequency, caught =
 * jank, FUELW*thrust = power, the hedge = a learned frequency safety margin.
 * State = (sensor reading); actions = thrust 0..SMAX. Pure integer.
 */
#ifndef ENV_SWIRL_H
#define ENV_SWIRL_H

#include "rl_core.h"

#define SW_SMAX     5
#define SW_NACT     (SW_SMAX+1)
#define SW_HORIZON  160
#define SW_PSWIRL   400          /* per-step swirl probability (per-mille) */
#define SW_CAUGHT   20           /* penalty for being caught               */
#define SW_FUELW    1            /* fuel/power cost per thrust level        */

typedef struct { rl_rng rng; int strength, obs, t; rl_feature spec[1]; } swirl_ctx;

static inline int sw__draw(rl_rng *r){
    return ((int)(rl_rand(r)%1000) < SW_PSWIRL) ? (1 + (int)(rl_rand(r)%SW_SMAX)) : 0;
}
/* noisy sensor: O = clamp(S +/- 1, 1, SMAX); calm reads as 0 (no false alarm). */
static inline int sw__observe(rl_rng *r, int s){
    if (s == 0) return 0;
    int o = s + ((int)(rl_rand(r)%3) - 1);
    if (o < 1) o = 1;
    if (o > SW_SMAX) o = SW_SMAX;
    return o;
}
static inline void sw__feat(swirl_ctx *w, rl_fp *f){ f[0]=w->obs; }

static inline void swirl_reset(rl_env *e, rl_fp *f){
    swirl_ctx *w=(swirl_ctx*)e->ctx;        /* rng keeps advancing across episodes */
    w->t=0; w->strength=sw__draw(&w->rng); w->obs=sw__observe(&w->rng, w->strength);
    sw__feat(w, f);
}
static inline rl_fp swirl_step(rl_env *e, int thrust, rl_fp *f, int *done){
    swirl_ctx *w=(swirl_ctx*)e->ctx;
    if (thrust<0) thrust=0;
    if (thrust>SW_SMAX) thrust=SW_SMAX;
    int caught = (w->strength > 0) && (thrust < w->strength);    /* judged on TRUE S */
    rl_fp reward = RL_INT(-(caught ? SW_CAUGHT : 0) - SW_FUELW*thrust);
    w->t++; *done = (w->t >= SW_HORIZON);
    w->strength=sw__draw(&w->rng); w->obs=sw__observe(&w->rng, w->strength);
    sw__feat(w, f);
    return reward;
}
static inline void swirl_make(rl_env *e, uint32_t seed){
    swirl_ctx *w=malloc(sizeof(swirl_ctx)); rl_seed(&w->rng, seed?seed:1);
    w->spec[0].lo=0; w->spec[0].hi=SW_SMAX; w->spec[0].bins=SW_NACT;
    e->n_actions=SW_NACT; e->n_features=1; e->spec=w->spec;
    e->episodic=1; e->max_steps=SW_HORIZON;
    e->ctx=w; e->reset=swirl_reset; e->step=swirl_step;
}
static inline void swirl_free(rl_env *e){ free(e->ctx); }

/* ---- fair comparison: 5 policies on the SAME hidden swirl sequence ----
   0 coast(0)  1 max(SMAX)  2 match-reading(O)  3 omniscient(true S)  4 agent  */
enum { SWP_COAST, SWP_MAX, SWP_MATCH, SWP_OMNI, SWP_AGENT, SWP_N };
typedef struct { int reward[SWP_N], caught_pct[SWP_N]; } swirl_eval;
static inline swirl_eval swirl_eval_all(rl_agent *ag, int episodes, uint32_t seed){
    long long rew[SWP_N]={0}; int caught[SWP_N]={0}, swirls=0;
    for (int ep=0; ep<episodes; ep++){
        rl_rng g; rl_seed(&g, seed + (uint32_t)ep*2654435761u);
        for (int t=0; t<SW_HORIZON; t++){
            int s = sw__draw(&g), o = sw__observe(&g, s);
            if (s>0) swirls++;
            for (int p=0;p<SWP_N;p++){
                int thr;
                switch (p){
                    case SWP_COAST: thr=0; break;
                    case SWP_MAX:   thr=SW_SMAX; break;
                    case SWP_MATCH: thr=o; break;
                    case SWP_OMNI:  thr=s; break;
                    default: { rl_fp f[1]={(rl_fp)o}; thr=ag->act_greedy(ag,f); }
                }
                if (thr<0) thr=0;
                if (thr>SW_SMAX) thr=SW_SMAX;
                if (s>0 && thr<s) caught[p]++;
                rew[p] += -(s>0 && thr<s ? SW_CAUGHT : 0) - SW_FUELW*thr;
            }
        }
    }
    swirl_eval r;
    for (int p=0;p<SWP_N;p++){
        r.reward[p]=(int)(rew[p]/episodes);
        r.caught_pct[p]= swirls? (int)((long long)caught[p]*100/swirls):0;
    }
    return r;
}

#endif /* ENV_SWIRL_H */
