/*
 * contest.c - the RL ALGORITHM CONTEST on the slippery-cliff arena.
 * =================================================================
 * Runs every algorithm (all integer-only, all implementing the rl_agent vtable)
 * on the SAME discriminating environment (env_arena.h: a cliff-walk where
 * actions slip), under the SAME training budget and exploration schedule, and
 * ranks them on four axes:
 *
 *   eval_return  : greedy return averaged over many STOCHASTIC episodes. The
 *                  truly optimal policy under slip avoids the cliff edge, so
 *                  this rewards robust (safe) policies. PRIMARY score.
 *   train_falls  : cliff falls during learning  (safety / on-policy benefit).
 *   online_ret   : average reward per episode WHILE training (exploration cost).
 *   steps2thresh : environment steps until greedy eval first clears a bar
 *                  (sample efficiency -- planning & traces win here).
 *
 * Writes results/contest_scores.csv and results/contest_curves.csv for plots.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "rl_env.h"

/* registration wrappers (one TU each, so static helpers never clash) */
#define DECL(s) extern rl_agent reg_##s(const rl_feature*, int, int, uint32_t);
DECL(qlearn) DECL(sarsa) DECL(expsarsa) DECL(doubleq) DECL(qlambda)
DECL(sarsalambda) DECL(dynaq) DECL(mc) DECL(nstep) DECL(psweep) DECL(actorcritic)

typedef rl_agent (*makefn)(const rl_feature*, int, int, uint32_t);
typedef struct { const char *name; makefn make; } entry;
static entry REG[] = {
    { "Q-learning",    reg_qlearn },     { "SARSA",         reg_sarsa },
    { "Expected-SARSA",reg_expsarsa },   { "Double-Q",      reg_doubleq },
    { "Q(lambda)",     reg_qlambda },    { "SARSA(lambda)", reg_sarsalambda },
    { "Dyna-Q",        reg_dynaq },      { "Monte-Carlo",   reg_mc },
    { "n-step-SARSA",  reg_nstep },      { "Prio-Sweeping", reg_psweep },
    { "Actor-Critic",  reg_actorcritic },
};
#define NALG ((int)(sizeof(REG)/sizeof(REG[0])))

/* arena + contest configuration (identical for every algorithm).
   Deterministic cliff + PERSISTENT exploration is the textbook setup that
   separates on-policy from off-policy control (Sutton & Barto Fig. 6.4). */
#define A_ROWS 4
#define A_COLS 8
#define A_SLIP 0            /* deterministic cliff; discrimination via exploration */
#define A_SEED 4242
#define E_SEED 9001
#define TRAIN_STEPS 300000
#define EXPLORE_EPS 100     /* CONSTANT 10% throughout -> reveals on/off-policy gap */
#define CKPTS 30
#define CKPT_EVAL 20
#define FINAL_EVAL 200
#define THRESHOLD (-30)     /* eval-return bar for the sample-efficiency metric */

typedef struct {
    const char *name;
    int eval_return, success, online_ret, steps2thresh;
    long long train_falls;
    int curve[CKPTS];
} record;

int main(void){
    mkdir("results", 0755);
    FILE *cv = fopen("results/contest_curves.csv", "w");
    if (cv) fprintf(cv, "algo,step,eval_return\n");
    record rec[NALG];

    for (int i = 0; i < NALG; i++){
        rl_env tr, ev;
        arena_make(&tr, A_ROWS, A_COLS, A_SLIP, A_SEED);
        arena_make(&ev, A_ROWS, A_COLS, A_SLIP, E_SEED);
        rl_agent ag = REG[i].make(tr.spec, tr.n_features, tr.n_actions, 1234u + 7u*i);

        rl_fp feat[RL_MAX_FEAT]; tr.reset(&tr, feat);
        rl_fp r = 0; int done = 0;
        long long online2 = 0, falls_half = 0; int episodes2 = 0, s2t = -1, ck = 0;
        int every = TRAIN_STEPS / CKPTS, half = TRAIN_STEPS / 2;
        rec[i].name = REG[i].name;
        for (int t = 0; t < TRAIN_STEPS; t++){
            if (t == half) falls_half = arena_falls(&tr);
            if (t % every == 0 && ck < CKPTS){
                arena_result e = arena_eval(&ev, &ag, CKPT_EVAL);
                rec[i].curve[ck++] = e.avg_return;
                if (s2t < 0 && e.avg_return >= THRESHOLD) s2t = t;
                if (cv) fprintf(cv, "%s,%d,%d\n", REG[i].name, t, e.avg_return);
            }
            ag.set_epsilon(&ag, EXPLORE_EPS);
            int a = ag.step(&ag, r, feat, done, 1);
            if (done){ tr.reset(&tr, feat); r = 0; done = 0; if (t >= half) episodes2++; continue; }
            r = tr.step(&tr, a, feat, &done);
            if (t >= half) online2 += r;          /* steady-state online return */
        }
        while (ck < CKPTS){ rec[i].curve[ck] = rec[i].curve[ck ? ck-1 : 0]; ck++; }

        arena_result fin = arena_eval(&ev, &ag, FINAL_EVAL);
        rec[i].eval_return = fin.avg_return;
        rec[i].success = fin.success_pct;
        rec[i].train_falls = arena_falls(&tr) - falls_half;     /* falls in 2nd half */
        rec[i].online_ret = episodes2 ? (int)(online2 / episodes2 / RL_FP_ONE) : -999;
        rec[i].steps2thresh = s2t;

        ag.destroy(&ag); arena_free(&tr); arena_free(&ev);
        printf("  ran %-16s online=%d eval=%d success=%d%% falls=%lld\n",
               REG[i].name, rec[i].online_ret, rec[i].eval_return,
               rec[i].success, rec[i].train_falls);
    }
    if (cv) fclose(cv);

    /* PRIMARY rank = steady-state ONLINE return (the cliff's on/off-policy axis,
       Sutton & Barto Fig. 6.4): safe on-policy methods score higher because they
       fall less while still exploring. */
    for (int i = 0; i < NALG; i++) for (int j = i+1; j < NALG; j++)
        if (rec[j].online_ret > rec[i].online_ret){ record t = rec[i]; rec[i] = rec[j]; rec[j] = t; }

    printf("\n=== CONTEST SCOREBOARD : deterministic cliff, persistent 10%% exploration ===\n");
    printf("  %-15s %8s %8s %8s %10s %10s\n",
           "algorithm", "online", "eval", "success", "falls/2", "steps2bar");
    for (int i = 0; i < NALG; i++){
        char s2[16];
        if (rec[i].steps2thresh < 0) snprintf(s2, sizeof s2, "never");
        else snprintf(s2, sizeof s2, "%d", rec[i].steps2thresh);
        printf("  %2d. %-11s %8d %8d %7d%% %10lld %10s\n",
               i+1, rec[i].name, rec[i].online_ret, rec[i].eval_return,
               rec[i].success, rec[i].train_falls, s2);
    }
    printf("  online = avg reward/episode WHILE exploring (higher=safer/better)\n");
    printf("  eval   = greedy return (off-policy methods learn the optimal edge path)\n");
    printf("  falls/2= cliff falls in 2nd half of training; steps2bar = sample efficiency\n");

    FILE *sc = fopen("results/contest_scores.csv", "w");
    if (sc){
        fprintf(sc, "rank,algo,online_ret,eval_return,success,train_falls,steps2thresh\n");
        for (int i = 0; i < NALG; i++)
            fprintf(sc, "%d,%s,%d,%d,%d,%lld,%d\n", i+1, rec[i].name,
                    rec[i].online_ret, rec[i].eval_return, rec[i].success,
                    rec[i].train_falls, rec[i].steps2thresh);
        fclose(sc);
        printf("\nSaved: results/contest_scores.csv, results/contest_curves.csv\n");
    }
    return 0;
}
