/*
 * tournament.c - the 11 integer RL agents across the HARDER environments.
 * =======================================================================
 * Three escalating challenges, each stressing a different weakness:
 *   ARENA  : slippery cliff (5x10, 10% slip)         -> robustness to noise
 *   FROZEN : 8x8 slippery lake, 12 deadly holes       -> hazardous exploration
 *   SHIFT  : non-stationary blocking maze (gap moves)  -> adaptability
 *
 * Every agent is trained from scratch on each env under the same budget and a
 * persistent 10% exploration rate, then scored greedily. We rank per-env by
 * average greedy return and combine by average rank.
 *
 * Writes results/tournament_scores.csv.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "rl_env.h"

#define DECL(s) extern rl_agent reg_##s(const rl_feature*, int, int, uint32_t);
DECL(qlearn) DECL(sarsa) DECL(expsarsa) DECL(doubleq) DECL(qlambda)
DECL(sarsalambda) DECL(dynaq) DECL(mc) DECL(nstep) DECL(psweep) DECL(actorcritic)

typedef rl_agent (*makefn)(const rl_feature*, int, int, uint32_t);
typedef struct { const char *name; makefn make; } entry;
static entry REG[] = {
    { "Q-learning",reg_qlearn },{ "SARSA",reg_sarsa },{ "Expected-SARSA",reg_expsarsa },
    { "Double-Q",reg_doubleq },{ "Q(lambda)",reg_qlambda },{ "SARSA(lambda)",reg_sarsalambda },
    { "Dyna-Q",reg_dynaq },{ "Monte-Carlo",reg_mc },{ "n-step-SARSA",reg_nstep },
    { "Prio-Sweeping",reg_psweep },{ "Actor-Critic",reg_actorcritic },
};
#define NALG ((int)(sizeof(REG)/sizeof(REG[0])))
#define EPS 100
#define EVAL 200

typedef struct { int arena, frozen, frozen_succ, shift; int rank; } score;

int main(void){
    mkdir("results", 0755);
    score sc[NALG]; const char *nm[NALG];

    for (int i = 0; i < NALG; i++){
        nm[i] = REG[i].name;
        uint32_t seed = 555u + 31u*i;

        /* ARENA: slippery cliff */
        { rl_env tr, ev; arena_make(&tr,5,10,100,4242); arena_make(&ev,5,10,100,9001);
          rl_agent a = REG[i].make(tr.spec, tr.n_features, tr.n_actions, seed);
          rl_train(&tr,&a,200000,EPS,EPS);
          sc[i].arena = rl_eval_many(&ev,&a,EVAL).avg_return;
          a.destroy(&a); arena_free(&tr); arena_free(&ev); }

        /* FROZEN: deadly slippery lake (same layout for train & eval) */
        { rl_env tr, ev; frozen_make(&tr,8,8,12,150,0xF00D); frozen_make(&ev,8,8,12,150,0xF00D);
          rl_agent a = REG[i].make(tr.spec, tr.n_features, tr.n_actions, seed);
          rl_train(&tr,&a,250000,EPS,EPS);
          rl_eval_stats s = rl_eval_many(&ev,&a,EVAL);
          sc[i].frozen = s.avg_return; sc[i].frozen_succ = s.success_pct;
          a.destroy(&a); frozen_free(&tr); frozen_free(&ev); }

        /* SHIFT: train on phase 0, flip the gap, retrain, score phase 1 */
        { rl_env tr, ev; shift_make(&tr,0); shift_make(&ev,0);
          rl_agent a = REG[i].make(tr.spec, tr.n_features, tr.n_actions, seed);
          rl_train(&tr,&a,120000,EPS,EPS);
          shift_set_phase(&tr,1); shift_set_phase(&ev,1);
          rl_train(&tr,&a,4000,EPS,EPS);
          sc[i].shift = rl_eval_many(&ev,&a,EVAL).avg_return;
          a.destroy(&a); shift_free(&tr); shift_free(&ev); }

        printf("  %-15s arena=%d frozen=%d(%d%%) shift=%d\n",
               nm[i], sc[i].arena, sc[i].frozen, sc[i].frozen_succ, sc[i].shift);
    }

    /* per-env rank (1=best=highest return), combined = sum of the three ranks */
    int idx[NALG]; for (int i=0;i<NALG;i++) idx[i]=i;
    int combined[NALG]; for (int i=0;i<NALG;i++) combined[i]=0;
    for (int env=0; env<3; env++){
        for (int i=0;i<NALG;i++){
            int better=1;
            for (int j=0;j<NALG;j++){
                int vi = env==0?sc[i].arena : env==1?sc[i].frozen : sc[i].shift;
                int vj = env==0?sc[j].arena : env==1?sc[j].frozen : sc[j].shift;
                if (vj>vi) better++;
            }
            combined[i]+=better;
        }
    }
    /* sort by combined rank ascending */
    for (int i=0;i<NALG;i++) for (int j=i+1;j<NALG;j++)
        if (combined[idx[j]]<combined[idx[i]]){ int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }

    printf("\n=== GRAND TOURNAMENT (avg rank over 3 harder envs) ===\n");
    printf("  %-15s %7s %7s %7s %9s\n","algorithm","arena","frozen","shift","avgRank");
    for (int k=0;k<NALG;k++){ int i=idx[k];
        printf("  %2d. %-12s %7d %7d %7d %9d\n", k+1, nm[i],
               sc[i].arena, sc[i].frozen, sc[i].shift, combined[i]); }
    printf("  (returns: higher=better; avgRank = sum of per-env ranks, lower=better)\n");

    FILE *f = fopen("results/tournament_scores.csv","w");
    if (f){ fprintf(f,"rank,algo,arena,frozen,frozen_succ,shift,sumrank\n");
        for (int k=0;k<NALG;k++){ int i=idx[k];
            fprintf(f,"%d,%s,%d,%d,%d,%d,%d\n", k+1, nm[i], sc[i].arena,
                    sc[i].frozen, sc[i].frozen_succ, sc[i].shift, combined[i]); }
        fclose(f); printf("\nSaved: results/tournament_scores.csv\n"); }
    return 0;
}
