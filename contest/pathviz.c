/*
 * pathviz.c - visualize each agent's learned path vs the DFS/BFS-optimal path,
 * with its reward and Bellman-delta curves. Emits CSVs for tools/plot_svg.py.
 *
 *   pathviz [rows] [cols] [mines]        (default 6 9 6, on the windy maze)
 *
 * Writes:
 *   results/viz_meta.csv    grid + mines + the BFS-optimal path
 *   results/viz_paths.csv   agent, step, row, col   (each agent's greedy path)
 *   results/viz_curves.csv  agent, episode, reward, delta
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "rl_env.h"

#define A(s) extern rl_agent reg_##s(const rl_feature*,int,int,uint32_t); \
             extern rl_fp qval_##s(rl_agent*,const rl_fp*,int);
A(qlearn) A(sarsa) A(expsarsa) A(doubleq) A(qlambda) A(sarsalambda)
A(dynaq) A(mc) A(nstep) A(psweep) A(actorcritic)
typedef rl_agent (*makefn)(const rl_feature*,int,int,uint32_t);
typedef rl_fp (*qvalfn)(rl_agent*,const rl_fp*,int);
typedef struct { const char *name; makefn make; qvalfn qval; } entry;
static entry REG[] = {
    {"qlearn",reg_qlearn,qval_qlearn},{"sarsa",reg_sarsa,qval_sarsa},
    {"expsarsa",reg_expsarsa,qval_expsarsa},{"doubleq",reg_doubleq,qval_doubleq},
    {"qlambda",reg_qlambda,qval_qlambda},{"sarsalambda",reg_sarsalambda,qval_sarsalambda},
    {"dynaq",reg_dynaq,qval_dynaq},{"mc",reg_mc,qval_mc},{"nstep",reg_nstep,qval_nstep},
    {"psweep",reg_psweep,qval_psweep},{"actorcritic",reg_actorcritic,qval_actorcritic},
};
#define NALG ((int)(sizeof(REG)/sizeof(REG[0])))
#define GAMMAREF 230          /* fixed ~0.9 so the Bellman residual is comparable */
#define TRAIN_STEPS 300000
#define MAXEP 30000
#define CURVE_PTS 60

static int rir(rl_fp v){ return (v>=0)?(v+128)/256:-(((-v)+128)/256); }

int main(int argc, char **argv){
    int R = argc>1?atoi(argv[1]):6, C = argc>2?atoi(argv[2]):9, M = argc>3?atoi(argv[3]):6;
    mkdir("results",0755);
    rl_env env; windy_make(&env, R, C, M, 7);
    windy_ctx *w = (windy_ctx*)env.ctx;
    int NS = R*C, start = w->start, goal = w->goal;

    /* ---- BFS optimal PATH (parents) over deterministic dynamics ---- */
    int *prev = malloc(sizeof(int)*NS), *pa = malloc(sizeof(int)*NS), *q = malloc(sizeof(int)*NS);
    for (int i=0;i<NS;i++) prev[i]=-1;
    int head=0,tail=0; prev[start]=start; q[tail++]=start;
    while(head<tail){ int s=q[head++]; if(s==goal) break;
        for(int a=0;a<4;a++){ rl_fp rr; int dd; int s2=windy__next(w,s,a,&rr,&dd);
            if(s2==s) continue;
            if(prev[s2]<0){ prev[s2]=s; pa[s2]=a; q[tail++]=s2; } } }

    FILE *fm=fopen("results/viz_meta.csv","w");
    fprintf(fm,"key,a,b,c\n");
    fprintf(fm,"grid,%d,%d,0\n",R,C);
    fprintf(fm,"start,%d,%d,0\n",start/C,start%C);
    fprintf(fm,"goal,%d,%d,0\n",goal/C,goal%C);
    for(int i=0;i<NS;i++) if(w->mine[i]) fprintf(fm,"mine,%d,%d,0\n",i/C,i%C);
    /* optimal path cells (reverse the parent chain) */
    int opt[4096], on=0, s=goal; while(s!=start){ opt[on++]=s; s=prev[s]; } opt[on++]=start;
    for(int k=on-1;k>=0;k--) fprintf(fm,"opt,%d,%d,0\n",opt[k]/C,opt[k]%C);
    fclose(fm); free(prev); free(pa); free(q);

    FILE *fp=fopen("results/viz_paths.csv","w");  fprintf(fp,"agent,step,row,col\n");
    FILE *fc=fopen("results/viz_curves.csv","w"); fprintf(fc,"agent,episode,reward,delta\n");

    static int rew[MAXEP], del[MAXEP];
    for(int ai=0; ai<NALG; ai++){
        rl_agent ag = REG[ai].make(env.spec, env.n_features, env.n_actions, 777);
        qvalfn QV = REG[ai].qval;

        rl_fp feat[RL_MAX_FEAT]; env.reset(&env, feat);
        rl_fp rprev=0; int doneprev=0, nep=0;
        long long ep_rew=0, ep_del=0; int ep_n=0;
        for(int t=0; t<TRAIN_STEPS && nep<MAXEP; t++){
            ag.set_epsilon(&ag, 400 - 390*t/TRAIN_STEPS);
            rl_fp sfeat[RL_MAX_FEAT]; memcpy(sfeat,feat,sizeof sfeat);
            int a = ag.step(&ag, rprev, feat, doneprev, 1);
            if(doneprev){ rew[nep]=(int)ep_rew; del[nep]=ep_n?(int)(ep_del/ep_n):0; nep++;
                          ep_rew=0; ep_del=0; ep_n=0; env.reset(&env,feat); rprev=0; doneprev=0; continue; }
            rl_fp r = env.step(&env, a, feat, &doneprev);
            /* uniform Bellman residual: delta = r + gamma*max_a' Q(s') - Q(s,a) */
            rl_fp qsa = QV(&ag, sfeat, a), mx=0;
            if(!doneprev){ mx=QV(&ag,feat,0); for(int b=1;b<env.n_actions;b++){ rl_fp v=QV(&ag,feat,b); if(v>mx)mx=v; } }
            rl_fp delta = r + (doneprev?0:rl_mul(GAMMAREF,mx)) - qsa;
            ep_rew += rir(r); ep_del += (delta<0?-delta:delta); ep_n++;
            rprev = r;
        }
        /* greedy path */
        env.reset(&env, feat); int cur=start, steps=0, reached=0;
        fprintf(fp,"%s,%d,%d,%d\n",REG[ai].name,0,cur/C,cur%C);
        for(; steps<env.max_steps; steps++){
            int a = ag.act_greedy(&ag, feat);
            int dn; rl_fp r=env.step(&env,a,feat,&dn); (void)r;
            cur = (int)feat[0]*C + (int)feat[1];
            fprintf(fp,"%s,%d,%d,%d\n",REG[ai].name,steps+1,cur/C,cur%C);
            if(dn){ reached=1; steps++; break; }
        }
        /* downsampled curves */
        for(int k=0;k<CURVE_PTS && nep>0;k++){ int e=k*nep/CURVE_PTS;
            fprintf(fc,"%s,%d,%d,%d\n",REG[ai].name,e,rew[e],rir((rl_fp)del[e])); }
        printf("  %-12s path=%d steps reached=%s  episodes=%d\n",
               REG[ai].name, steps, reached?"yes":"no", nep);
        ag.destroy(&ag);
    }
    fclose(fp); fclose(fc);
    printf("\nSaved: results/viz_meta.csv, viz_paths.csv, viz_curves.csv  (optimal=%d steps)\n", on-1);
    windy_free(&env);
    return 0;
}
