/*
 * swirl_sim.c - swirl escape with a noisy sensor; minimal-fuel reward.
 * ===================================================================
 * The swirl strength is HIDDEN; the agent sees only a noisy reading (O = S+/-1).
 * Thrusting the reading ("match") is caught whenever the sensor under-reads. The
 * learned agent discovers a HEDGE -- thrust a margin above the reading -- trading
 * a little fuel for far fewer catches, beating the reactive oracle and matching
 * the omniscient (full-information) policy. Maps directly to choosing a CPU
 * frequency a notch above the estimated workload to avoid jank.
 *
 *   cc -O2 -std=c99 -Wall -Iinclude -o swirl_sim src/swirl_sim.c && ./swirl_sim
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "rl_env.h"
#include "algo_qlearn.h"

int main(void){
    rl_env env; swirl_make(&env, 12345);
    rl_agent ag = qlearn_agent_make(env.spec, env.n_features, env.n_actions, 0xBEEF);
    qlearn_agent_set(&ag, RL_FRAC(1,16), 0, -1);   /* a=.0625  g=0 (per-step task) */

    printf("Swirl escape - PARTIALLY OBSERVED (noisy sensor), minimal-fuel reward\n");
    printf("true swirl<=%d  sensor=O+/-1  thrust 0..%d  caught_pen=%d  fuel_cost/level=%d\n\n",
           SW_SMAX, SW_SMAX, SW_CAUGHT, SW_FUELW);

    rl_train(&env, &ag, 600000, 200, 20);

    swirl_eval r = swirl_eval_all(&ag, 800, 99);
    const char *names[SWP_N] = { "coast (thr 0)", "max (full thr)", "match-reading",
                                 "omniscient", "qlearn (learned)" };
    printf("Policy comparison (same hidden swirls; reward higher=better):\n");
    printf("  %-18s %8s %8s\n", "policy", "reward", "caught%");
    for (int p=0;p<SWP_N;p++)
        printf("  %-18s %8d %7d%%%s\n", names[p], r.reward[p], r.caught_pct[p],
               p==SWP_AGENT ? "  <- learned" : p==SWP_OMNI ? "  (full-info bound)" :
               p==SWP_MATCH ? "  (reactive ORACLE)" : "");

    printf("\nLearned thrust vs sensor reading:\n");
    printf("  %-9s", "reading"); for (int o=0;o<=SW_SMAX;o++) printf("%4d", o); printf("\n");
    printf("  %-9s", "thrust");
    for (int o=0;o<=SW_SMAX;o++){ rl_fp f[1]={o}; printf("%4d", ag.act_greedy(&ag,f)); }
    printf("\n  (thrust > reading for mid readings = learned safety margin/hedge)\n");

    mkdir("results", 0755);
    FILE *cf = fopen("results/swirl_trace.csv", "w");
    if (cf){
        fprintf(cf, "t,swirl,thrust,fuel,caught\n");
        rl_rng g; rl_seed(&g, 99);
        for (int t=0;t<SW_HORIZON;t++){
            int s=sw__draw(&g), o=sw__observe(&g,s);
            rl_fp f[1]={o}; int thr=ag.act_greedy(&ag,f);
            fprintf(cf, "%d,%d,%d,%d,%d\n", t, s, thr, thr, (s>0 && thr<s));
        }
        fclose(cf);
        printf("\nSaved: results/swirl_trace.csv\n");
    }

    int pass = (r.reward[SWP_AGENT] > r.reward[SWP_MATCH])
            && (r.reward[SWP_AGENT] > r.reward[SWP_MAX])
            && (r.reward[SWP_AGENT] > r.reward[SWP_COAST]);
    printf("\n--- Validation ---\n");
    printf("learned %d  vs  reactive-oracle %d  (omniscient bound %d)\n",
           r.reward[SWP_AGENT], r.reward[SWP_MATCH], r.reward[SWP_OMNI]);
    printf("RESULT: %s%s\n", pass ? "PASS" : "FAIL",
           pass ? "  (beat the reactive oracle by hedging the noisy sensor)" : "");

    ag.destroy(&ag); swirl_free(&env);
    return pass ? 0 : 1;
}
