/*
 * tuner_plug_example.c - plug-and-play into an async, init-once tuner.
 * ===================================================================
 * Shows the EXACT lifecycle dfc_tuner_qlearn_misc.c needs, using rl_controller:
 *
 *   qlm_init / qlm_start  (ONCE):   build agent + controller with a reward fn
 *   handle_load_change    (EACH ASYNC EVENT):   action = rl_controller_tick(feat)
 *   qlm_destroy           (ONCE):   free
 *
 * Pure integer. Compile on a host (no kernel headers):
 *   cc -O2 -std=c99 -Wall -Iinclude -o tuner_plug tunner/tuner_plug_example.c && ./tuner_plug
 */
#include <stdio.h>
#include <stdlib.h>
#include "rl_core.h"
#include "rl_controller.h"
#include "algo_qlearn.h"     /* swap for algo_sarsa.h / algo_dynaq.h / ... freely */

#define NLEVELS 6            /* workload levels == frequency levels (the actions) */
#define JANK_PEN 30
#define POWER_W  2

/* ---- domain state the reward function reads (your "user" pointer) ---- */
typedef struct { int last_jank, last_freq; } outcome;

/* ---- YOUR reward function: scores the PREVIOUS action. On the real tuner this
   is computed from janky_frames delta + curr_power in the new features. ---- */
static rl_fp tuner_reward(const rl_fp *prev_feat, int prev_action,
                          const rl_fp *cur_feat, void *user){
    (void)prev_feat; (void)prev_action; (void)cur_feat;
    outcome *o = (outcome *)user;
    return RL_INT(-(o->last_jank ? JANK_PEN : 0) - POWER_W * o->last_freq);
}

/* a tiny xorshift world: true workload wanders; the sensor reading is noisy */
static uint32_t wr = 999;
static int wrand(int n){ wr^=wr<<13; wr^=wr>>17; wr^=wr<<5; return (int)(wr%n); }

int main(void){
    /* ---------- qlm_init / qlm_start : ONCE ---------- */
    rl_feature spec[1];
    rl_fp lo[1] = {0}, hi[1] = {NLEVELS-1}; int bins[1] = {NLEVELS};
    rl_make_spec(spec, 1, lo, hi, bins);                  /* 1 input: workload reading */

    rl_agent agent = qlearn_agent_make(spec, /*n_inputs=*/1, /*n_actions=*/NLEVELS, 0xC0FFEE);
    qlearn_agent_set(&agent, RL_FRAC(1,8), 0, -1);        /* per-event task -> gamma 0 */

    outcome world = { 0, 0 };
    rl_controller ctrl;
    rl_controller_init(&ctrl, &agent, tuner_reward, &world);

    /* ---------- handle_load_change : EACH ASYNC EVENT ---------- */
    int true_load = 0, ticks = 200000, jank = 0, seen = 0;
    for (int t = 0; t < ticks; t++){
        rl_controller_set_epsilon(&ctrl, 200 - 190*t/ticks);   /* anneal exploration */

        /* 1. read features for this event (noisy workload estimate) */
        int est = true_load + (wrand(3) - 1);
        if (est < 0) est = 0;
        if (est > NLEVELS-1) est = NLEVELS-1;
        rl_fp feat[1] = { (rl_fp)est };

        /* 2. ONE call: learns from last action, returns the freq to set now */
        int freq = rl_controller_tick(&ctrl, feat);

        /* 3. apply it + observe the outcome (-> reward for the next tick) */
        int janked = freq < true_load;
        world.last_jank = janked; world.last_freq = freq;
        if (t > ticks*9/10){ seen++; jank += janked; }       /* measure once trained */

        true_load += wrand(3) - 1;                            /* workload wanders */
        if (true_load < 0) true_load = 0;
        if (true_load > NLEVELS-1) true_load = NLEVELS-1;
    }

    /* ---------- inspect the learned controller ---------- */
    printf("Plug-and-play tuner (rl_controller + reward fn, init once)\n\n");
    printf("Learned policy (workload reading -> frequency level):\n  ");
    int ok = 1;
    for (int r = 0; r < NLEVELS; r++){
        rl_fp f[1] = { (rl_fp)r }; int a = agent.act_greedy(&agent, f);
        printf("%d->%d  ", r, a);
        if (a < r) ok = 0;                                    /* must cover the load */
    }
    printf("\nLate-training jank rate: %d/%d = %d%%\n", jank, seen, seen?100*jank/seen:0);
    int pass = ok && seen && (100*jank/seen <= 15);
    printf("\nRESULT: %s  (frequency tracks/hedges the workload; jank kept low)\n",
           pass ? "PASS" : "FAIL");

    /* ---------- qlm_destroy : ONCE ---------- */
    agent.destroy(&agent);
    return pass ? 0 : 1;
}
