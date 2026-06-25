/*
 * test_reference.c - validates the contest infrastructure end to end using the
 * reference Q-learning agent on all three tasks. This is also the TEMPLATE that
 * each algorithm's own test (tests/test_<algo>.c) should follow.
 *
 *   cc -O2 -std=c99 -Wall -Iinclude -o test_reference tests/test_reference.c
 */
#include <stdio.h>
#include "rl_env.h"
#include "algo_qlearn.h"

int main(void){
    int pass = 1;

    /* ---- task 1: windy maze ---- */
    rl_env we; windy_make(&we, 8, 12, 14, 7);
    rl_agent a1 = qlearn_agent_make(we.spec, we.n_features, we.n_actions, 123);
    rl_train(&we, &a1, 400000, 400, 10);
    rl_rollout r = rl_eval_episode(&we, &a1);
    int opt = windy_optimal_len(&we);
    int maze_ok = r.reached && r.steps <= 2*opt + 5;
    printf("[windy ] reached=%d steps=%d (opt %d) return=%d  -> %s\n",
           r.reached, r.steps, opt, r.ret, maze_ok ? "PASS" : "FAIL");
    pass &= maze_ok;
    a1.destroy(&a1); windy_free(&we);

    /* ---- task 2: frequency tuning ---- */
    rl_env pe; phone_make(&pe, 7);
    rl_agent a2 = qlearn_agent_make(pe.spec, pe.n_features, pe.n_actions, 55);
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

    /* ---- task 3: contest arena (slippery cliff) ---- */
    rl_env ae; arena_make(&ae, 4, 8, 100, 999);
    rl_agent a3 = qlearn_agent_make(ae.spec, ae.n_features, ae.n_actions, 77);
    rl_train(&ae, &a3, 300000, 400, 10);
    long long train_falls = arena_falls(&ae);
    arena_result ar = arena_eval(&ae, &a3, 300);
    int arena_ok = ar.success_pct >= 80;
    printf("[arena ] eval_return=%d success=%d%% eval_falls=%lld train_falls=%lld -> %s\n",
           ar.avg_return, ar.success_pct, ar.falls, train_falls, arena_ok ? "PASS" : "FAIL");
    pass &= arena_ok;
    a3.destroy(&a3); arena_free(&ae);

    printf("\nINFRASTRUCTURE: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
