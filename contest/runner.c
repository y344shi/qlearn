/*
 * runner.c - run ANY agent (or all 11) on ANY environment, and rank them.
 * =======================================================================
 * Usage:  runner <env> <agent|all> [p1] [p2] [p3]
 *   env   : windy | arena | frozen | shift | swirl | phone
 *   agent : all | qlearn sarsa expsarsa doubleq qlambda sarsalambda
 *                 dynaq mc nstep psweep actorcritic
 *   sizes : windy  rows cols mines      arena  rows cols slip
 *           frozen rows cols holes      shift/swirl/phone : (none)
 *
 * Every env reports a single "score" (higher = better) so `all` can rank.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rl_env.h"

#define DECL(s) extern rl_agent reg_##s(const rl_feature*, int, int, uint32_t);
DECL(qlearn) DECL(sarsa) DECL(expsarsa) DECL(doubleq) DECL(qlambda)
DECL(sarsalambda) DECL(dynaq) DECL(mc) DECL(nstep) DECL(psweep) DECL(actorcritic)
typedef rl_agent (*makefn)(const rl_feature*, int, int, uint32_t);
typedef struct { const char *name; makefn make; } entry;
static entry REG[] = {
    {"qlearn",reg_qlearn},{"sarsa",reg_sarsa},{"expsarsa",reg_expsarsa},
    {"doubleq",reg_doubleq},{"qlambda",reg_qlambda},{"sarsalambda",reg_sarsalambda},
    {"dynaq",reg_dynaq},{"mc",reg_mc},{"nstep",reg_nstep},
    {"psweep",reg_psweep},{"actorcritic",reg_actorcritic},
};
#define NALG ((int)(sizeof(REG)/sizeof(REG[0])))

typedef struct { int score; char detail[96]; } result;

/* ---- per-environment trial of one agent ---- */
static result trial(const char *env, makefn mk, int p1, int p2, int p3){
    result r; r.score = -999999; r.detail[0] = 0;
    if (!strcmp(env,"swirl")){
        rl_env e; swirl_make(&e, 12345);
        rl_agent a = mk(e.spec, e.n_features, e.n_actions, 0xBEEF);
        rl_train(&e, &a, 600000, 200, 20);
        swirl_eval s = swirl_eval_all(&a, 800, 99);
        r.score = s.reward[SWP_AGENT];
        snprintf(r.detail, sizeof r.detail, "reward=%d caught=%d%%  (oracle %d, omni %d)",
                 s.reward[SWP_AGENT], s.caught_pct[SWP_AGENT], s.reward[SWP_MATCH], s.reward[SWP_OMNI]);
        a.destroy(&a); swirl_free(&e);
    } else if (!strcmp(env,"phone")){
        rl_env e; phone_make(&e, 7);
        rl_agent a = mk(e.spec, e.n_features, e.n_actions, 55);
        rl_train(&e, &a, 200000, 300, 10);
        phone_agent_ctx pc = { &a };
        phone_metrics mq = phone_score(phone_pick_agent, &pc, 6000, 99);
        phone_metrics mp = phone_score(phone_pick_perf,  NULL, 6000, 99);
        int epct = (int)(100*mq.energy/(mp.energy?mp.energy:1));
        r.score = -(mq.jank_ppm/100) - epct;       /* low jank + low energy = high score */
        snprintf(r.detail, sizeof r.detail, "jank=%d.%d%% energy=%d%% of perf",
                 mq.jank_ppm/10000, (mq.jank_ppm/1000)%10, epct);
        a.destroy(&a); phone_free(&e);
    } else {                                         /* grid envs: windy/arena/frozen/shift */
        rl_env tr, ev; int opt = -1, episodic_steps;
        if (!strcmp(env,"windy")){ windy_make(&tr,p1,p2,p3,7); windy_make(&ev,p1,p2,p3,7);
                                   opt = windy_optimal_len(&tr); episodic_steps = 60*p1*p2; }
        else if (!strcmp(env,"arena")){ arena_make(&tr,p1,p2,p3,4242); arena_make(&ev,p1,p2,p3,9001);
                                   episodic_steps = 200000; }
        else if (!strcmp(env,"frozen")){ frozen_make(&tr,p1,p2,p3,150,0xF00D); frozen_make(&ev,p1,p2,p3,150,0xF00D);
                                   episodic_steps = 250000; }
        else { shift_make(&tr,0); shift_make(&ev,0); episodic_steps = 150000; }   /* shift */
        if (episodic_steps < 120000) episodic_steps = 120000;
        rl_agent a = mk(tr.spec, tr.n_features, tr.n_actions, 777);
        if (!strcmp(env,"shift")){
            rl_train(&tr,&a, episodic_steps, 100,100);
            shift_set_phase(&tr,1); shift_set_phase(&ev,1);
            rl_train(&tr,&a, 8000, 100,100);             /* small budget = adaptation speed */
        } else {
            rl_train(&tr,&a, episodic_steps, 400, 10);
        }
        rl_eval_stats s = rl_eval_many(&ev, &a, 200);
        r.score = s.avg_return;
        if (opt >= 0) snprintf(r.detail, sizeof r.detail, "return=%d success=%d%%  (optimal %d)",
                               s.avg_return, s.success_pct, opt);
        else          snprintf(r.detail, sizeof r.detail, "return=%d success=%d%%",
                               s.avg_return, s.success_pct);
        a.destroy(&a);
        if (!strcmp(env,"windy")){ windy_free(&tr); windy_free(&ev); }
        else if (!strcmp(env,"arena")){ arena_free(&tr); arena_free(&ev); }
        else if (!strcmp(env,"frozen")){ frozen_free(&tr); frozen_free(&ev); }
        else { shift_free(&tr); shift_free(&ev); }
    }
    return r;
}

int main(int argc, char **argv){
    if (argc < 3){ fprintf(stderr,"usage: runner <env> <agent|all> [p1 p2 p3]\n"); return 2; }
    const char *env = argv[1], *who = argv[2];
    int p1 = argc>3?atoi(argv[3]):0, p2 = argc>4?atoi(argv[4]):0, p3 = argc>5?atoi(argv[5]):0;
    /* sensible size defaults */
    if (!strcmp(env,"windy")){ if(!p1)p1=8; if(!p2)p2=12; if(!p3)p3=14; }
    if (!strcmp(env,"arena")){ if(!p1)p1=5; if(!p2)p2=10; if(!p3)p3=100; }
    if (!strcmp(env,"frozen")){ if(!p1)p1=8; if(!p2)p2=8; if(!p3)p3=12; }

    printf("Environment: %s   agent(s): %s\n", env, who);

    if (strcmp(who,"all")){                          /* a single named agent */
        for (int i=0;i<NALG;i++) if (!strcmp(REG[i].name,who)){
            result r = trial(env, REG[i].make, p1,p2,p3);
            printf("  %-12s  %s\n", who, r.detail);
            return 0;
        }
        fprintf(stderr,"unknown agent '%s'\n", who); return 2;
    }

    /* ALL agents: trial each, then rank by score */
    result res[NALG]; const char *nm[NALG];
    for (int i=0;i<NALG;i++){ nm[i]=REG[i].name; res[i]=trial(env,REG[i].make,p1,p2,p3);
        printf("  ran %-12s %s\n", nm[i], res[i].detail); }
    int idx[NALG]; for (int i=0;i<NALG;i++) idx[i]=i;
    for (int i=0;i<NALG;i++) for (int j=i+1;j<NALG;j++)
        if (res[idx[j]].score > res[idx[i]].score){ int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }
    printf("\n=== RANKING on %s (best first) ===\n", env);
    for (int k=0;k<NALG;k++){ int i=idx[k];
        printf("  %2d. %-12s %s\n", k+1, nm[i], res[i].detail); }
    return 0;
}
