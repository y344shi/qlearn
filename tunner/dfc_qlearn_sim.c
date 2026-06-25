/*
 * dfc_qlearn_sim.c - kernel-migration harness: swappable integer RL agents
 * driving a SIMULATED CPU-frequency tuner.
 * ========================================================================
 * Goal: the exact shape needed to drop into dfc_tuner_qlearn_misc.c. Every
 * agent is hidden behind a 3-pointer vtable -- init / select / destroy -- so any
 * of the 11 algorithms can be swapped in without touching the tuner:
 *
 *     freq_agent fa = freq_agent_make("dynaq", reg_dynaq, &prop);   // init
 *     unsigned int khz = fa.select(&fa, &features);                 // feature->freq
 *     fa.destroy(&fa);                                              // destroy
 *
 * Features are exactly those requested: avg_load, curr_refresh_rate, curr_power,
 * frame_budget -- a subset of the real struct __sched_ind_qlearn_features. The
 * "select" call also learns online (it scores the previous frequency from the
 * new features, the way handle_load_change sees the result on the next trigger).
 *
 * Pure integer. Build:
 *   cc -O2 -std=c99 -Iinclude -o dfc_sim tunner/dfc_qlearn_sim.c contest/reg_*.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rl_core.h"
#include "algo_qlearn.h"     /* qlearn_agent_set(): lets the sweep set gamma/alpha */

/* ============ kernel-like data structures (subset of the real ones) ======= */
/* mirrors struct __sched_ind_qlearn_features (hmkernel/power/sched_indicator.h) */
struct qlearn_features {
    unsigned int avg_load;          /* 0..1024 cluster load                    */
    unsigned int curr_refresh_rate; /* 0..3 -> 60/90/120/144 Hz                */
    unsigned int curr_power;        /* current power draw (model units)        */
    int          frame_budget;      /* >0 = ahead of deadline, <0 = jank       */
};
/* mirrors struct dfc_prop (freqmgr/driver) */
struct dfc_prop {
    const unsigned int *freq_table; /* available frequencies (kHz)             */
    int                 freq_table_cnt;
    unsigned int        min_freq, max_freq;
};

/* ============ the swappable agent vtable (the migration contract) ========= */
typedef struct freq_agent {
    const char  *name;
    void        *h;                                              /* opaque handle */
    unsigned int (*select)(struct freq_agent *, const struct qlearn_features *);
    void         (*destroy)(struct freq_agent *);
} freq_agent;

/* ============ generic adapter: wrap ANY rl_agent into a freq_agent ========= */
typedef rl_agent (*makefn)(const rl_feature *, int, int, uint32_t);

#define JANK_PEN 1000       /* missing the deadline dominates the reward */

typedef struct {
    rl_agent           ag;
    const struct dfc_prop *prop;
    rl_feature         spec[4];
    int                t, train;
    int                jank_pen, power_div;   /* reward weights (swept below) */
} adapter;

static unsigned int adapter_select(freq_agent *fa, const struct qlearn_features *f){
    adapter *a = (adapter *)fa->h;
    int fb = f->frame_budget; if (fb < -3000) fb = -3000; if (fb > 3000) fb = 3000;
    rl_fp feat[4] = { (rl_fp)f->avg_load, (rl_fp)f->curr_refresh_rate,
                      (rl_fp)f->curr_power, (rl_fp)fb };
    /* reward for the PREVIOUS action, read from this event's outcome. Jank
       (frame overshoot) dominates; among jank-free frames, lower power wins.   */
    int jank = (f->frame_budget < 0);
    rl_fp reward = jank ? RL_INT(-a->jank_pen)
                        : RL_INT(200 - (int)(f->curr_power) / a->power_div);
    /* anneal exploration while "training"; near-greedy afterwards */
    int eps = a->t < a->train ? (300 - 290 * a->t / a->train) : 10;
    a->ag.set_epsilon(&a->ag, eps);
    int action = a->ag.step(&a->ag, reward, feat, /*done=*/0, /*explore=*/1);
    a->t++;
    if (action < 0) action = 0;
    if (action >= a->prop->freq_table_cnt) action = a->prop->freq_table_cnt - 1;
    return a->prop->freq_table[action];
}
static void adapter_destroy(freq_agent *fa){
    adapter *a = (adapter *)fa->h; a->ag.destroy(&a->ag); free(a);
}
static freq_agent freq_agent_make(const char *name, makefn mk, const struct dfc_prop *prop,
                                  int train, int jank_pen, int power_div){
    adapter *a = malloc(sizeof *a);
    a->prop = prop; a->t = 0; a->train = train; a->jank_pen = jank_pen; a->power_div = power_div;
    a->spec[0].lo=0; a->spec[0].hi=1023; a->spec[0].bins=8;   /* load            */
    a->spec[1].lo=0; a->spec[1].hi=3;    a->spec[1].bins=4;   /* refresh         */
    a->spec[2].lo=0; a->spec[2].hi=800;  a->spec[2].bins=4;   /* power           */
    a->spec[3].lo=-3000; a->spec[3].hi=3000; a->spec[3].bins=3; /* frame budget  */
    a->ag = mk(a->spec, 4, prop->freq_table_cnt, 0xC0FFEE);
    freq_agent fa; fa.name = name; fa.h = a; fa.select = adapter_select; fa.destroy = adapter_destroy;
    return fa;
}

/* ============ the SIMULATED frequency environment ========================= */
static const unsigned int FREQ_KHZ[] = {1239000,1530000,1740000,1930000,2150000,2350000,2620000};
#define NFREQ ((int)(sizeof(FREQ_KHZ)/sizeof(FREQ_KHZ[0])))
#define FMAX_MHZ 2620
static const int BUDGET_US[] = {16667,11111,8333,6944};   /* 60/90/120/144 Hz */
static int powmodel(int mhz){ return 50 + mhz*mhz/10000; }   /* ~f^2 */

/* workload phases: required MHz to hit the deadline, with jitter + a refresh */
typedef struct { int req, jit, rr, len; } phase;
static const phase PH[] = {
    {1239,120,0,70},{1740,150,0,70},{2150,220,2,70},{2350,260,1,70},{2620,360,2,50}
};
#define NPH ((int)(sizeof(PH)/sizeof(PH[0])))
static uint32_t wr=12345; static int wrnd(int n){ wr^=wr<<13;wr^=wr>>17;wr^=wr<<5; return (int)(wr%n); }

typedef struct { int jank_ppm, avg_power, avg_mhz; } sim_result;

/* run a freq_agent through the simulated tuner; measure over the eval window */
static sim_result run_sim(freq_agent *fa, int ticks, int eval_from){
    wr = 777;                                  /* same workload for every agent (fair) */
    int phase_i=0, left=0, req=1239, rr=0;
    int prev_budget=0; unsigned int prev_power=200;
    long long pj=0, pp=0, pf=0; int n=0;
    for (int t=0;t<ticks;t++){
        if (left<=0){ phase_i=wrnd(NPH); left=PH[phase_i].len; }
        req = PH[phase_i].req + (wrnd(2*PH[phase_i].jit+1)-PH[phase_i].jit);
        if (req<600) req=600;
        rr=PH[phase_i].rr; left--;

        struct qlearn_features f;
        f.avg_load = (unsigned)((long)req*1024/FMAX_MHZ); if (f.avg_load>1023) f.avg_load=1023;
        f.curr_refresh_rate = rr; f.curr_power = prev_power; f.frame_budget = prev_budget;

        unsigned int khz = fa->select(fa, &f);
        int mhz = (int)(khz/1000);
        int budget = BUDGET_US[rr];
        int exec = (int)((long)req*budget/mhz);          /* time to finish frame */
        int fb = budget - exec;                          /* >0 ahead, <0 jank    */
        int power = powmodel(mhz);

        if (t>=eval_from){ pj += (fb<0); pp += power; pf += mhz; n++; }
        prev_budget = fb; prev_power = (unsigned)power;
    }
    sim_result r; r.jank_ppm=(int)((long long)pj*1000000/n); r.avg_power=(int)(pp/n); r.avg_mhz=(int)(pf/n);
    return r;
}

/* ============ swap every agent in and rank them =========================== */
#define D(s) extern rl_agent reg_##s(const rl_feature*,int,int,uint32_t);
D(qlearn) D(sarsa) D(expsarsa) D(doubleq) D(qlambda) D(sarsalambda)
D(dynaq) D(mc) D(nstep) D(psweep) D(actorcritic)
static struct { const char *n; makefn m; } REG[] = {
    {"qlearn",reg_qlearn},{"sarsa",reg_sarsa},{"expsarsa",reg_expsarsa},{"doubleq",reg_doubleq},
    {"qlambda",reg_qlambda},{"sarsalambda",reg_sarsalambda},{"dynaq",reg_dynaq},{"mc",reg_mc},
    {"nstep",reg_nstep},{"psweep",reg_psweep},{"actorcritic",reg_actorcritic},
};
#define NALG ((int)(sizeof(REG)/sizeof(REG[0])))

/* baseline governors as freq_agents (no learning) */
static unsigned int sel_perf(freq_agent *fa,const struct qlearn_features *f){ (void)f;
    const struct dfc_prop *p=fa->h; return p->freq_table[p->freq_table_cnt-1]; }
static unsigned int sel_ond(freq_agent *fa,const struct qlearn_features *f){
    const struct dfc_prop *p=fa->h; int req=(int)((long)f->avg_load*FMAX_MHZ/1024)*118/100;
    for(int i=0;i<p->freq_table_cnt;i++) if((int)(p->freq_table[i]/1000)>=req) return p->freq_table[i];
    return p->freq_table[p->freq_table_cnt-1]; }
static void sel_none(freq_agent *fa){ (void)fa; }

/* ============ hyperparameter sweep + Pareto front (jank vs power) ========= */
/* Sweeps gamma x alpha x reward-weights for Q-learning (it exposes a setter),
   then prints the non-dominated jank/power trade-offs.                        */
typedef struct { int jank_ppm, power; rl_fp g, al; int jp, pd; } sw_pt;

static void sweep(const struct dfc_prop *prop){
    int TICKS=120000, EVAL=20000, train=TICKS-EVAL;
    rl_fp gammas[] = {0, 128, 192, 230};      /* 0, .50, .75, .90  */
    rl_fp alphas[] = {16, 32, 64, 128};       /* .0625, .125, .25, .50 */
    int   jps[]    = {500, 1000, 2000};
    int   pds[]    = {2, 4, 8};
    sw_pt pts[4*4*3*3]; int np=0;
    for (int gi=0; gi<4; gi++) for (int ai=0; ai<4; ai++)
    for (int ji=0; ji<3; ji++) for (int pi=0; pi<3; pi++){
        freq_agent fa = freq_agent_make("qlearn", reg_qlearn, prop, train, jps[ji], pds[pi]);
        qlearn_agent_set(&((adapter*)fa.h)->ag, alphas[ai], gammas[gi], -1);
        sim_result r = run_sim(&fa, TICKS, train);
        pts[np++] = (sw_pt){ r.jank_ppm, r.avg_power, gammas[gi], alphas[ai], jps[ji], pds[pi] };
        fa.destroy(&fa);
    }
    /* Pareto front: keep points not dominated (lower jank AND lower power) by any other */
    printf("Hyperparameter sweep (Q-learning): %d configs; Pareto front of jank vs power\n", np);
    printf("  %7s %7s | %7s %7s %9s %9s\n","jank%","power","gamma","alpha","jank_pen","power_div");
    /* print front sorted by jank ascending */
    for (int i=0;i<np;i++){
        int dom=0;
        for (int j=0;j<np;j++) if (j!=i &&
            pts[j].jank_ppm<=pts[i].jank_ppm && pts[j].power<=pts[i].power &&
            (pts[j].jank_ppm<pts[i].jank_ppm || pts[j].power<pts[i].power)){ dom=1; break; }
        pts[i].jp = dom ? -pts[i].jp : pts[i].jp;   /* mark dominated with negative jp */
    }
    /* selection-sort the front (non-dominated) by jank, print */
    for (int n=0;;n++){
        int best=-1;
        for (int i=0;i<np;i++) if (pts[i].jp>0 && (best<0 || pts[i].jank_ppm<pts[best].jank_ppm)) best=i;
        if (best<0) break;
        printf("  %5d.%1d %7d | %4ld/256 %4ld/256 %9d %9d\n",
               pts[best].jank_ppm/10000,(pts[best].jank_ppm/1000)%10, pts[best].power,
               (long)pts[best].g,(long)pts[best].al, pts[best].jp, pts[best].pd);
        pts[best].jp = 0;
    }
    printf("\nLower-left of the front = better (less jank AND less power). Pick by product policy.\n");
}

int main(int argc, char **argv){
    struct dfc_prop prop = { FREQ_KHZ, NFREQ, FREQ_KHZ[0], FREQ_KHZ[NFREQ-1] };
    int TICKS=120000, EVAL=20000;
    if (argc > 1 && !strcmp(argv[1], "sweep")){ sweep(&prop); return 0; }

    printf("Simulated kernel frequency tuner  (features: load, refresh, power, frame_budget)\n");
    printf("freq table (MHz):"); for(int i=0;i<NFREQ;i++) printf(" %u", FREQ_KHZ[i]/1000); printf("\n\n");

    typedef struct { const char *n; sim_result r; int learned; } row;
    row rows[NALG+2]; int nr=0;

    /* baselines */
    freq_agent perf={ "performance", &prop, sel_perf, sel_none };
    freq_agent ond ={ "ondemand",    &prop, sel_ond,  sel_none };
    rows[nr++] = (row){ "performance", run_sim(&perf, TICKS, TICKS-EVAL), 0 };
    rows[nr++] = (row){ "ondemand",    run_sim(&ond,  TICKS, TICKS-EVAL), 0 };

    for (int i=0;i<NALG;i++){
        freq_agent fa = freq_agent_make(REG[i].n, REG[i].m, &prop, TICKS-EVAL, 1000, 4);
        rows[nr++] = (row){ REG[i].n, run_sim(&fa, TICKS, TICKS-EVAL), 1 };
        fa.destroy(&fa);
    }

    /* score = avoid jank first, then save power: lower (jank*10 + power) is better */
    for (int i=0;i<nr;i++) for(int j=i+1;j<nr;j++){
        int si=rows[i].r.jank_ppm/100 + rows[i].r.avg_power;
        int sj=rows[j].r.jank_ppm/100 + rows[j].r.avg_power;
        if (sj<si){ row t=rows[i]; rows[i]=rows[j]; rows[j]=t; }
    }
    printf("Swappable agents on the simulated tuner (ranked; lower jank+power = better):\n");
    printf("  %-14s %8s %8s %8s\n","agent","jank%","power","avgMHz");
    for (int i=0;i<nr;i++)
        printf("  %2d. %-12s %6d.%1d %8d %8d%s\n", i+1, rows[i].n,
               rows[i].r.jank_ppm/10000,(rows[i].r.jank_ppm/1000)%10,
               rows[i].r.avg_power, rows[i].r.avg_mhz, rows[i].learned?"":"  (baseline)");
    printf("\nAll agents driven through the SAME init/select/destroy vtable.\n");
    return 0;
}
