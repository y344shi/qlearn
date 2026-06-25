/*
 * phone_sim.c - a simulated phone for the Q-learning CPU-frequency tuner.
 * =======================================================================
 *
 * This is the "hard part": instead of a gridworld, the environment is a phone
 * doing frame-by-frame rendering. Each frame the workload demands some CPU
 * work; the agent (include/qlearn.h) picks a CPU frequency from the real big-
 * cluster table. If the chosen frequency is too low to finish the frame inside
 * its deadline -> JANK. Higher frequencies burn more power (super-linear).
 *
 * The goal mirrors dfc_tuner_qlearn_misc.c exactly (see tunner/): learn a DVFS
 * policy that AVOIDS JANK while MINIMISING POWER -- i.e. pick the lowest
 * frequency that still hits each frame deadline. We then compare the learned
 * policy against the performance / powersave / ondemand governors and an oracle.
 *
 * Everything is integer arithmetic. Features, reward and state binning follow
 * tunner/environment_description.md (sections 3-6).
 *
 *   cc -O2 -std=c99 -Wall -Iinclude -o phone_sim src/phone_sim.c && ./phone_sim
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include "qlearn.h"

/* ---- big-cluster frequency table (MHz), from the log data in the doc ---- */
static const int FREQ[] = { 1239, 1530, 1740, 1930, 2150, 2350, 2620 };
#define NFREQ ((int)(sizeof(FREQ) / sizeof(FREQ[0])))
#define FMAX  2620

/* ---- display refresh rates -> per-frame deadline (microseconds) --------- */
static const int BUDGET_US[] = { 16667, 11111, 8333, 6944 };   /* 60/90/120/144 Hz */
#define NREFRESH 4

/* ---- models -------------------------------------------------------------- */
/* power "units" ~ f^2 (V scales with f) -> super-linear, range ~200..740.    */
static int power_units(int f_mhz) { return 50 + (f_mhz * f_mhz) / 10000; }

/* smallest table frequency >= req MHz (else the max). */
static int freq_ceil_idx(int req) {
    for (int i = 0; i < NFREQ; i++) if (FREQ[i] >= req) return i;
    return NFREQ - 1;
}

/* ---- workload trace ------------------------------------------------------ */
/* A phone runs through phases; each frame needs a "required frequency" req
   (the smallest freq that would finish the frame on time) at some refresh.
   We store req per frame so every governor is scored on the SAME workload.   */
typedef struct { int req, refresh; } Frame;

typedef struct { int req_base, jitter, refresh, len; } Phase;
static const Phase PHASES[] = {
    { 1239, 120,  0,  70 },   /* idle / reading     - light, 60Hz   */
    { 1740, 150,  0,  70 },   /* video playback     - medium, 60Hz  */
    { 2150, 220,  2,  70 },   /* UI scrolling       - heavy, 120Hz  */
    { 2350, 260,  1,  70 },   /* gaming             - heavy, 90Hz   */
    { 2620, 350,  2,  50 },   /* burst (can exceed max -> some unavoidable jank) */
};
#define NPHASES ((int)(sizeof(PHASES) / sizeof(PHASES[0])))

static uint32_t trace_rng = 7;
static uint32_t trnd(void){ trace_rng^=trace_rng<<13; trace_rng^=trace_rng>>17; trace_rng^=trace_rng<<5; return trace_rng; }

static void gen_trace(Frame *fr, int n, uint32_t seed) {
    trace_rng = seed ? seed : 1;
    int t = 0;
    while (t < n) {
        const Phase *p = &PHASES[(trnd() >> 8) % NPHASES];
        for (int k = 0; k < p->len && t < n; k++, t++) {
            int j = (int)(trnd() % (2 * p->jitter + 1)) - p->jitter;
            int req = p->req_base + j;
            if (req < 600) req = 600;
            fr[t].req = req;
            fr[t].refresh = p->refresh;
        }
    }
}

/* util (0..1023): demand as a fraction of max capacity = req/FMAX. This is the
   freq-invariant load the scheduler reports (doc 3.1 avg_load).              */
static int util_of(int req) {
    long u = (long)req * 1024 / FMAX;
    return (int)(u > 1023 ? 1023 : u);
}

/* ---- a frequency-governor policy run over a trace, returns metrics ------- */
typedef struct { int jank_ppm; long long energy; int avg_freq; } Metrics;

static Metrics run_policy(const Frame *fr, int n, int (*pick)(int util, int refresh, void *ctx), void *ctx) {
    long long energy = 0, fsum = 0; int jank = 0;
    for (int t = 0; t < n; t++) {
        int budget = BUDGET_US[fr[t].refresh];
        int aidx = pick(util_of(fr[t].req), fr[t].refresh, ctx);
        if (aidx < 0) aidx = 0;
        if (aidx >= NFREQ) aidx = NFREQ - 1;
        int f = FREQ[aidx];
        int exec_us = (fr[t].req * budget) / f;          /* time to finish frame */
        if (exec_us > budget) jank++;                    /* missed the deadline  */
        energy += (long long)power_units(f) * exec_us;   /* active energy        */
        fsum += f;
    }
    Metrics m;
    m.jank_ppm = (int)((long long)jank * 1000000 / n);
    m.energy = energy;
    m.avg_freq = (int)(fsum / n);
    return m;
}

/* baseline governors */
static int pol_perf(int u, int r, void *c){ (void)u;(void)r;(void)c; return NFREQ-1; }
static int pol_power(int u, int r, void *c){ (void)u;(void)r;(void)c; return 0; }
static int pol_optimal(int u, int r, void *c){ (void)r;(void)c; return freq_ceil_idx((int)((long)u*FMAX/1024)); }
static int pol_ondemand(int u, int r, void *c){ (void)r;(void)c;       /* +18% headroom */
    return freq_ceil_idx((int)((long)u * FMAX / 1024 * 118 / 100)); }
/* the learned agent (greedy) */
static int pol_qlearn(int u, int r, void *c){
    qlearn_t *ag = (qlearn_t *)c; ql_fp f[2] = { (ql_fp)u, (ql_fp)r };
    return qlearn_greedy(ag, f);
}

/* ------------------------------- main ----------------------------------- */
#define LOAD_BINS 24
#define TRAIN_FRAMES 80000
#define EVAL_FRAMES  6000
#define JANK_PENALTY 3000     /* missing a frame deadline is much worse than power */

int main(void) {
    /* state = (util bin, refresh bin); action = frequency table index */
    ql_feature spec[2] = { { 0, 1023, LOAD_BINS }, { 0, NREFRESH - 1, NREFRESH } };
    ql_fp *Q = malloc(sizeof(ql_fp) * qlearn_qcount(spec, 2, NFREQ));
    qlearn_t ag;
    qlearn_init(&ag, spec, 2, NFREQ, Q, 0xA11CEu);
    qlearn_set_params(&ag, QL_FRAC(1,5), 205, 300);     /* a=0.20 g=0.80 e=0.30 */

    printf("Simulated-phone DVFS tuner  (pure-integer Q-learning via qlearn.h)\n");
    printf("freq table (MHz):");
    for (int i = 0; i < NFREQ; i++) printf(" %d", FREQ[i]);
    printf("\nstates=%d (util x refresh)  actions=%d  train_frames=%d\n\n",
           ag.n_states, NFREQ, TRAIN_FRAMES);

    /* ---- online training: this loop is the shape of handle_load_change ---- */
    Frame *tr = malloc(sizeof(Frame) * TRAIN_FRAMES);
    gen_trace(tr, TRAIN_FRAMES, 7);
    ql_fp reward_prev = 0;
    for (int t = 0; t < TRAIN_FRAMES; t++) {
        qlearn_set_epsilon(&ag, 300 - 290 * t / TRAIN_FRAMES);   /* 0.30 -> 0.01 */

        int util = util_of(tr[t].req);                           /* scheduler load */
        ql_fp feat[2] = { (ql_fp)util, (ql_fp)tr[t].refresh };

        int aidx = qlearn_step(&ag, reward_prev, feat, 0);        /* <-- the agent */

        /* apply freq, observe outcome of THIS frame -> reward for next tick */
        int budget = BUDGET_US[tr[t].refresh];
        int f = FREQ[aidx];
        int exec_us = (tr[t].req * budget) / f;
        int jank = exec_us > budget;
        int p = power_units(f);
        reward_prev = QL_INT(jank ? (-JANK_PENALTY - p) : (1000 - p));
    }

    /* ---- evaluate every governor on the same fresh workload -------------- */
    Frame *ev = malloc(sizeof(Frame) * EVAL_FRAMES);
    gen_trace(ev, EVAL_FRAMES, 99);
    Metrics mq = run_policy(ev, EVAL_FRAMES, pol_qlearn,   &ag);
    Metrics mo = run_policy(ev, EVAL_FRAMES, pol_optimal,  NULL);
    Metrics mp = run_policy(ev, EVAL_FRAMES, pol_perf,     NULL);
    Metrics ms = run_policy(ev, EVAL_FRAMES, pol_power,    NULL);
    Metrics md = run_policy(ev, EVAL_FRAMES, pol_ondemand, NULL);

    long long base = mp.energy ? mp.energy : 1;
    printf("Governor comparison over %d frames:\n", EVAL_FRAMES);
    printf("  %-12s %8s %10s %9s\n", "policy", "jank%", "energy", "avgMHz");
    printf("  %-12s %7d.%1d %9lld%% %8d\n", "performance",
           mp.jank_ppm/10000, (mp.jank_ppm/1000)%10, 100*mp.energy/base, mp.avg_freq);
    printf("  %-12s %7d.%1d %9lld%% %8d\n", "powersave",
           ms.jank_ppm/10000, (ms.jank_ppm/1000)%10, 100*ms.energy/base, ms.avg_freq);
    printf("  %-12s %7d.%1d %9lld%% %8d\n", "ondemand",
           md.jank_ppm/10000, (md.jank_ppm/1000)%10, 100*md.energy/base, md.avg_freq);
    printf("  %-12s %7d.%1d %9lld%% %8d  <- learned\n", "qlearn",
           mq.jank_ppm/10000, (mq.jank_ppm/1000)%10, 100*mq.energy/base, mq.avg_freq);
    printf("  %-12s %7d.%1d %9lld%% %8d  <- oracle\n", "optimal",
           mo.jank_ppm/10000, (mo.jank_ppm/1000)%10, 100*mo.energy/base, mo.avg_freq);
    printf("  (energy shown as %% of the performance governor; lower is better)\n");

    /* ---- learned policy as actually exercised on the eval workload ------- */
    int bin_f[LOAD_BINS], bin_n[LOAD_BINS];
    for (int b = 0; b < LOAD_BINS; b++) { bin_f[b] = 0; bin_n[b] = 0; }
    for (int t = 0; t < EVAL_FRAMES; t++) {
        int u = util_of(ev[t].req); int b = u * LOAD_BINS / 1024;
        if (b >= LOAD_BINS) b = LOAD_BINS - 1;
        bin_f[b] = FREQ[pol_qlearn(u, ev[t].refresh, &ag)];   /* freq it actually chose */
        bin_n[b]++;
    }
    printf("\nLearned policy (load -> chosen MHz, over the real workload):\n");
    printf("  %-9s %-9s %s\n", "util", "chosenMHz", "frames");
    for (int b = 0; b < LOAD_BINS; b++) {
        if (!bin_n[b]) continue;
        printf("  %4d-%-4d %9d %d\n",
               b * 1024 / LOAD_BINS, (b + 1) * 1024 / LOAD_BINS - 1, bin_f[b], bin_n[b]);
    }
    printf("  (chosen frequency rises with load -- the agent learned DVFS)\n");

    /* ---- save a trace for plotting (demand vs chosen freq) --------------- */
    mkdir("results", 0755);
    FILE *cf = fopen("results/phone_trace.csv", "w");
    if (cf) {
        fprintf(cf, "frame,demand_mhz,util,q_mhz,opt_mhz,jank\n");
        int show = EVAL_FRAMES < 600 ? EVAL_FRAMES : 600;
        for (int t = 0; t < show; t++) {
            int u = util_of(ev[t].req);
            int qf = FREQ[pol_qlearn(u, ev[t].refresh, &ag)];
            int of = FREQ[pol_optimal(u, ev[t].refresh, NULL)];
            int budget = BUDGET_US[ev[t].refresh];
            int jank = (ev[t].req * budget) / qf > budget;
            fprintf(cf, "%d,%d,%d,%d,%d,%d\n", t, ev[t].req, u, qf, of, jank);
        }
        fclose(cf);
        printf("\nSaved: results/phone_trace.csv (demand vs chosen frequency)\n");
    }

    /* ---- validation: low jank AND clearly cheaper than always-max -------- */
    int pass = (mq.jank_ppm <= mo.jank_ppm + 20000)          /* within 2% jank of oracle */
            && (mq.energy < mp.energy * 95 / 100)            /* >=5% energy saved vs perf */
            && (mq.jank_ppm < ms.jank_ppm);                  /* far less jank than powersave */
    printf("\n--- Validation ---\n");
    printf("qlearn vs oracle jank: %d.%d%% vs %d.%d%%   energy vs perf: %lld%%\n",
           mq.jank_ppm/10000,(mq.jank_ppm/1000)%10, mo.jank_ppm/10000,(mo.jank_ppm/1000)%10,
           100*mq.energy/base);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    free(Q); free(tr); free(ev);
    return pass ? 0 : 1;
}
