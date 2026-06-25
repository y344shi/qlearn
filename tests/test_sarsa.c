/*
 * test_sarsa.c - validates the SARSA agent (algo_sarsa.h) on the two mandated
 * contest tasks: windy maze (episodic) and phone DVFS (continuing). Copied from
 * tests/test_reference.c with sarsa_agent_make swapped in for the reference
 * Q-learning constructor. Pure integer; prints PASS / exits 0 on success.
 *
 *   gcc-12 -O2 -std=c99 -Wall -Wextra -Iinclude -o /tmp/test_sarsa tests/test_sarsa.c && /tmp/test_sarsa
 */
#include <stdio.h>
#include "rl_env.h"
#include "algo_sarsa.h"

int main(void){
    int pass = 1;

    /* ---- task 1: windy maze ---- */
    rl_env we; windy_make(&we, 8, 12, 14, 7);
    rl_agent a1 = sarsa_agent_make(we.spec, we.n_features, we.n_actions, 123);
    /* On-policy SARSA explores via OPTIMISTIC INITIALISATION (see algo_sarsa.h),
       so the behaviour policy here is greedy (eps 0 -> 0); the optimism alone
       drives the systematic exploration that finds the goal past the mines. */
    rl_train(&we, &a1, 400000, 0, 0);
    rl_rollout r = rl_eval_episode(&we, &a1);
    int opt = windy_optimal_len(&we);
    int maze_ok = r.reached && r.steps <= 2*opt + 5;
    printf("[windy ] reached=%d steps=%d (opt %d) return=%d  -> %s\n",
           r.reached, r.steps, opt, r.ret, maze_ok ? "PASS" : "FAIL");
    pass &= maze_ok;
    a1.destroy(&a1); windy_free(&we);

    /* ---- task 2: frequency tuning ---- */
    rl_env pe; phone_make(&pe, 7);
    rl_agent a2 = sarsa_agent_make(pe.spec, pe.n_features, pe.n_actions, 55);
    rl_train(&pe, &a2, 200000, 300, 10);
    phone_agent_ctx pc = { &a2 };
    phone_metrics mq = phone_score(phone_pick_agent,   &pc, 6000, 99);
    phone_metrics mp = phone_score(phone_pick_perf,    NULL, 6000, 99);
    phone_metrics mo = phone_score(phone_pick_optimal, NULL, 6000, 99);
    int phone_ok = (mq.jank_ppm <= mo.jank_ppm + 20000) && (mq.energy < mp.energy*95/100);
    printf("[phone ] jank=%d.%d%% (oracle %d.%d%%) energy=%lld%% of perf  -> %s\n",
           mq.jank_ppm/10000, (mq.jank_ppm/1000)%10, mo.jank_ppm/10000, (mo.jank_ppm/1000)%10,
           100*mq.energy/(mp.energy?mp.energy:1), phone_ok ? "PASS" : "FAIL");
    pass &= phone_ok;
    a2.destroy(&a2); phone_free(&pe);

    printf("\nSARSA: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
