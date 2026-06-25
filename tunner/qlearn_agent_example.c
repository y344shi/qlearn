/*
 * qlearn_agent_example.c
 * ----------------------
 * Shows how the frequency tuner (dfc_tuner_qlearn_misc.c) can use the
 * general-purpose agent in ../include/qlearn.h as a drop-in include that
 * consumes a FEATURE VECTOR and emits an action -- with no kernel headers, so
 * it compiles and runs on a host:
 *
 *     cc -O2 -std=c99 -I../include -o agent_demo qlearn_agent_example.c && ./agent_demo
 *
 * The toy control task mirrors the tuner's job: each tick we observe features
 * (here: required performance "util" level + whether the last tick janked) and
 * must pick a CPU-frequency level. Picking a frequency below the demand causes
 * jank (big penalty); higher frequencies cost more power (small penalty). The
 * optimal policy is therefore "pick the lowest frequency >= demand".
 *
 * The real tuner would replace the simulated features/reward with the live
 * sched-indicator features and an outcome-based reward (see notes at the end).
 */
#include <stdio.h>
#include <stdlib.h>
#include "qlearn.h"

#define LEVELS      6        /* freq/util levels 0..5  (also the action count) */
#define JANK_PEN    30       /* reward penalty for under-provisioning (jank)   */
#define POWER_W      2       /* reward penalty per frequency level (power)     */
#define TICKS    200000      /* online control ticks to train over             */

/* simple integer PRNG so the demo is self-contained + reproducible */
static uint32_t rng = 12345u;
static uint32_t rnd(void) { rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }

int main(void) {
    /* --- describe the state to the agent: a 2-element feature vector --------
       feature 0 = util level   in [0, LEVELS-1], one bin each
       feature 1 = last-jank flag in {0,1}                                     */
    ql_feature spec[2] = {
        { 0, LEVELS - 1, LEVELS },
        { 0, 1,          2      },
    };
    int n_actions = LEVELS;                       /* choose a frequency level  */

    ql_fp *Q = malloc(sizeof(ql_fp) * qlearn_qcount(spec, 2, n_actions));
    qlearn_t ag;
    qlearn_init(&ag, spec, 2, n_actions, Q, 0xC0FFEE);
    qlearn_set_params(&ag, QL_FRAC(1,4), QL_FRAC(1,2), 200);  /* a=.25 g=.5 e=.20 */

    /* --- online control loop (this is the shape of handle_load_change) ----- */
    int util = 0, last_jank = 0;
    ql_fp reward_prev = 0;
    for (int t = 0; t < TICKS; t++) {
        qlearn_set_epsilon(&ag, 200 - 190 * t / TICKS);   /* anneal 0.20 -> 0.01 */

        ql_fp features[2] = { (ql_fp)util, (ql_fp)last_jank };
        int freq = qlearn_step(&ag, reward_prev, features, 0);   /* <-- the agent */

        /* apply the action + observe the outcome (the "environment") */
        int jank = (freq < util) ? 1 : 0;
        reward_prev = QL_INT(-(jank ? JANK_PEN : 0) - POWER_W * freq);
        last_jank = jank;

        /* demand wanders as a random walk */
        int step = (int)(rnd() % 3) - 1;
        util += step;
        if (util < 0) util = 0;
        if (util > LEVELS - 1) util = LEVELS - 1;
    }

    /* --- inspect the learned greedy policy --------------------------------- */
    printf("Learned frequency policy (feature vector -> chosen freq level)\n");
    printf("  util | chosen freq | optimal | ok\n");
    int correct = 0;
    for (int u = 0; u < LEVELS; u++) {
        ql_fp f[2] = { (ql_fp)u, 0 };
        int chosen = qlearn_greedy(&ag, f);
        int ok = (chosen == u);                   /* optimal = lowest freq >= u */
        correct += ok;
        printf("  %4d | %11d | %7d | %s\n", u, chosen, u, ok ? "yes" : "no");
    }
    int pass = (correct == LEVELS);
    printf("\nRESULT: %s  (%d/%d states optimal)\n",
           pass ? "PASS" : "FAIL", correct, LEVELS);

    free(Q);
    return pass ? 0 : 1;
}

/*
 * --- How the real tuner would use it ------------------------------------
 * In dfc_tuner_qlearn_misc.c, keep one qlearn_t + its Q buffer in tuner_priv,
 * initialise it in qlm_init(), then in handle_load_change():
 *
 *     ql_feature spec[] = {            // once, at init
 *         { 0, MAX_UTIL,  UTIL_BINS },
 *         { 0, MAX_JANK,  JANK_BINS },
 *         { min_freq, max_freq, FREQ_BINS },
 *     };
 *     qlearn_init(&priv->agent, spec, 3, NUM_FREQ_ACTIONS, priv->qbuf, seed);
 *
 *     // each sched-indicator event:
 *     ql_fp feat[3] = { chg->features.util,
 *                       chg->features.janky_frames,
 *                       priv->target_freq };
 *     ql_fp reward = QL_INT(-jank_delta) - QL_FRAC(power_cost, SCALE);
 *     int action = qlearn_step(&priv->agent, reward, feat, 0);
 *     unsigned int new_freq = freq_table[action];
 *     dfc_driver_set_freq(priv->prop, new_freq, FREQ_TABLE_CEIL_METHOD);
 *
 * The Q buffer is sized once with qlearn_qcount(spec, 3, NUM_FREQ_ACTIONS) and
 * can be a static array -- no malloc, no float, kernel-friendly.
 */
