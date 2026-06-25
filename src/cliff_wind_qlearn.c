/*
 * cliff_wind_qlearn.c
 * --------------------
 * Minimal, dependency-free, PURELY-INTEGER Q-learning validation example on a
 * configurable "mines-with-wind" gridworld (a generalisation of Sutton &
 * Barto's Cliff Walking + Windy Gridworld).
 *
 *   - Arbitrary R x C grid (chosen at run time).
 *   - Start S = bottom-left, Goal G = bottom-right.
 *   - MINES are scattered at random: stepping onto one gives reward -100 and
 *     teleports the agent back to S (episode continues). Every other step is -1.
 *   - Some columns have an upward WIND that pushes the agent toward row 0 after
 *     it moves. Wind strengths are randomised per column.
 *   - Reaching G ends the episode.
 *
 * Everything in the learning algorithm uses ONLY integer arithmetic:
 *   - Q-values, alpha, gamma are Q8.8 fixed point (one int holds value * 256).
 *   - Randomness is a deterministic integer xorshift32 PRNG.
 *   - epsilon-greedy uses an integer probability out of 1000.
 *
 * The map is generated so that the goal is always reachable (checked with a
 * BFS). The same BFS yields the GROUND-TRUTH optimal (shortest mine-free) path,
 * which is compared against the path the agent actually learns.
 *
 * Usage:   ./cliff [rows] [cols] [n_mines] [seed]
 *   defaults: 4 6  (n_mines ~= cells/7)  seed 12345
 *
 * Build:   cc -O2 -std=c99 -Wall -o cliff src/cliff_wind_qlearn.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>   /* mkdir() for the results/ directory */

#include "qlearn.h"     /* the general-purpose integer Q-learning agent */

/* ----------------------------- Fixed point ------------------------------ */
#define FP_SHIFT 8
#define FP_ONE   (1 << FP_SHIFT)        /* 256 == 1.0 (same scale as qlearn.h) */

static long ri(long v) {                /* Q8.8 -> nearest integer */
    return (v >= 0) ? (v + FP_ONE / 2) / FP_ONE
                    : -(((-v) + FP_ONE / 2) / FP_ONE);
}

/* ------------------------- Hyper-parameters ----------------------------- */
#define ALPHA_FP  64        /* learning rate  0.25  */
#define GAMMA_FP  243       /* discount       ~0.95 */
#define NA 4                /* 0=up 1=right 2=down 3=left */
#define STEP_R  (-1 * FP_ONE)
#define MINE_R  (-100 * FP_ONE)

/* ------------------------- Runtime world state -------------------------- */
static int R, C, NS;            /* grid dims, NS = R*C */
static int START, GOAL_R, GOAL_C, GOAL;
static int  g_episodes, g_maxsteps;
static int *wind;               /* [C]  upward push per column */
static int *mine;               /* [NS] 1 if cell is a mine */
static long *ep_return, *ep_delta;  /* [g_episodes] per-episode metrics */

/* The agent. The gridworld state is the (row,col) cell, fed to the agent as a
   2-feature vector; with bins = (R, C) the quantisation is exact (one bin per
   cell), so the cell index equals the agent's state index. This file owns ZERO
   Q-learning logic -- it all lives in qlearn.h. */
static qlearn_t   agent;
static ql_fp     *agent_Q;
static ql_feature feat_spec[2];

static int is_mine_rc(int r, int c) { return mine[r * C + c]; }

/* ------------------------- Integer PRNG --------------------------------- */
static unsigned int rng_state = 2463534242u;
static unsigned int xrand(void) {
    unsigned int x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}

typedef struct { int next; long reward; int done; } Step;

/* Deterministic transition: move, then apply wind, then resolve mine/goal.  */
static Step env_step(int state, int a) {
    int row = state / C, col = state % C;
    int nr = row, nc = col;
    switch (a) {
        case 0: nr--; break; case 1: nc++; break;
        case 2: nr++; break; case 3: nc--; break;
    }
    if (nr < 0) nr = 0;
    if (nr > R - 1) nr = R - 1;
    if (nc < 0) nc = 0;
    if (nc > C - 1) nc = C - 1;
    nr -= wind[nc];
    if (nr < 0) nr = 0;

    Step s;
    if (mine[nr * C + nc])             { s.next = START; s.reward = MINE_R; s.done = 0; }
    else if (nr == GOAL_R && nc == GOAL_C) { s.next = nr*C+nc; s.reward = STEP_R; s.done = 1; }
    else                               { s.next = nr*C+nc; s.reward = STEP_R; s.done = 0; }
    return s;
}

/* The greedy action in cell `s` is just the agent's argmax (cell == state). */
#define argmax_a(s) qlearn_argmax_idx(&agent, (s))

/* ------------------------- Greedy evaluation ---------------------------- */
/* Greedy rollout from start; reports steps, goal reached, total reward, and
   whether the greedy policy ever stepped on a mine.                        */
static long eval_greedy(int *out_steps, int *out_reached, int *out_hitmine) {
    int s = START, steps = 0, reached = 0, hit = 0; long total = 0;
    for (; steps < g_maxsteps; steps++) {
        Step st = env_step(s, argmax_a(s));
        total += st.reward;
        if (st.reward == MINE_R) hit = 1;
        s = st.next;
        if (st.done) { steps++; reached = 1; break; }
    }
    *out_steps = steps; *out_reached = reached; *out_hitmine = hit;
    return total;
}

/* ------------------- Ground-truth optimal path (BFS) -------------------- */
/* BFS over the deterministic dynamics gives the shortest mine-free path
   (each step costs -1, so fewest steps == best return). Mine moves resolve to
   START, so mine cells are never enqueued and the path avoids them. O(cells). */
static int *opt_states, *opt_acts;   /* [NS+1] reconstructed optimal path */
static int  opt_len, opt_reachable;

static void find_optimal(void) {
    int *prev = malloc(sizeof(int) * NS);
    int *pact = malloc(sizeof(int) * NS);
    int *dist = malloc(sizeof(int) * NS);
    int *q    = malloc(sizeof(int) * NS);
    for (int i = 0; i < NS; i++) { prev[i] = -1; dist[i] = -1; }

    int head = 0, tail = 0;
    dist[START] = 0; q[tail++] = START;
    while (head < tail) {
        int s = q[head++];
        if (s == GOAL) break;
        for (int a = 0; a < NA; a++) {
            int s2 = env_step(s, a).next;
            if (s2 == s) continue;            /* wall: no movement */
            if (dist[s2] < 0) {
                dist[s2] = dist[s] + 1; prev[s2] = s; pact[s2] = a;
                q[tail++] = s2;
            }
        }
    }

    if (dist[GOAL] < 0) {
        opt_reachable = 0; opt_len = 0;
    } else {
        opt_reachable = 1; opt_len = dist[GOAL];
        /* walk back from GOAL to START, then reverse */
        int chain[1]; (void)chain;
        int s = GOAL, n = opt_len;
        opt_states[n] = GOAL;
        while (s != START) {
            int a = pact[s], p = prev[s];
            opt_acts[n - 1] = a;
            opt_states[n - 1] = p;
            s = p; n--;
        }
    }
    free(prev); free(pact); free(dist); free(q);
}

/* ------------------------- Pretty printers ------------------------------ */
static void print_grid_legend(FILE *out) {
    fprintf(out, "Map (S=start G=goal *=mine, digit=wind strength above column):\n   ");
    for (int c = 0; c < C; c++) fprintf(out, "%d", wind[c] % 10);
    fprintf(out, "   <- wind\n");
    for (int r = 0; r < R; r++) {
        fprintf(out, "   ");
        for (int c = 0; c < C; c++) {
            if (r == START / C && c == START % C)      fputc('S', out);
            else if (r == GOAL_R && c == GOAL_C)       fputc('G', out);
            else if (is_mine_rc(r, c))                 fputc('*', out);
            else                                       fputc('.', out);
        }
        fputc('\n', out);
    }
}
static void print_policy(void) {
    const char *arrow[NA] = { "^", ">", "v", "<" };
    printf("Greedy policy (S G *=mine):\n");
    for (int r = 0; r < R; r++) {
        printf("  ");
        for (int c = 0; c < C; c++) {
            int s = r * C + c;
            if (s == START)                       printf(" S ");
            else if (r == GOAL_R && c == GOAL_C)  printf(" G ");
            else if (is_mine_rc(r, c))            printf(" * ");
            else                                  printf(" %s ", arrow[argmax_a(s)]);
        }
        printf("\n");
    }
}

/* ------------------------- ASCII charts --------------------------------- */
#define PLOT_W 68
#define PLOT_H 14
static void ascii_plot(FILE *out, const char *title, const char *yunit,
                       const long *v, int n, long dispmin, long dispmax) {
    char grid[PLOT_H][PLOT_W];
    for (int r = 0; r < PLOT_H; r++)
        for (int c = 0; c < PLOT_W; c++) grid[r][c] = ' ';
    if (dispmax <= dispmin) dispmax = dispmin + FP_ONE;

    for (int c = 0; c < PLOT_W; c++) {
        long lo = (long)c * n / PLOT_W, hi = (long)(c + 1) * n / PLOT_W;
        if (hi <= lo) hi = lo + 1;
        long sum = 0; int cnt = 0;
        for (long i = lo; i < hi && i < n; i++) { sum += v[i]; cnt++; }
        long val = cnt ? sum / cnt : dispmin;
        if (val > dispmax) val = dispmax;
        if (val < dispmin) val = dispmin;
        int row = (int)(((dispmax - val) * (PLOT_H - 1)) / (dispmax - dispmin));
        if (row < 0) row = 0;
        if (row > PLOT_H - 1) row = PLOT_H - 1;
        grid[row][c] = '*';
    }
    fprintf(out, "%s\n", title);
    for (int r = 0; r < PLOT_H; r++) {
        long ylab = dispmax - (dispmax - dispmin) * r / (PLOT_H - 1);
        if (r == 0 || r == PLOT_H - 1 || r == PLOT_H / 2)
            fprintf(out, "  %6ld |", ri(ylab));
        else
            fprintf(out, "         |");
        for (int c = 0; c < PLOT_W; c++) fputc(grid[r][c], out);
        fputc('\n', out);
    }
    fprintf(out, "  %6s +", yunit);
    for (int c = 0; c < PLOT_W; c++) fputc('-', out);
    fprintf(out, "\n         0%*d episode\n\n", PLOT_W - 6, g_episodes);
}

/* median of a Q8.8 array (used to auto-scale the clipped reward plot) */
static int cmp_long(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}
static long median_fp(const long *v, int n) {
    long *tmp = malloc(sizeof(long) * n);
    for (int i = 0; i < n; i++) tmp[i] = v[i];
    qsort(tmp, n, sizeof(long), cmp_long);
    long m = tmp[n / 2];
    free(tmp);
    return m;
}

static void print_best_path(FILE *out) {
    const char *arrow[NA] = { "^", ">", "v", "<" };
    char *cell = malloc(NS);
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            cell[r*C+c] = is_mine_rc(r, c) ? '*' : '.';
    int s = START, steps = 0; long total = 0; int reached = 0;
    char *moves = malloc(g_maxsteps + 1); int nm = 0;
    for (; steps < g_maxsteps; steps++) {
        int a = argmax_a(s);
        int row = s / C, col = s % C;
        if (s != START && !(row == GOAL_R && col == GOAL_C)) cell[s] = arrow[a][0];
        if (nm < g_maxsteps) moves[nm++] = arrow[a][0];
        Step st = env_step(s, a);
        total += st.reward; s = st.next;
        if (st.done) { steps++; reached = 1; break; }
    }
    cell[GOAL] = 'G'; cell[START] = 'S'; moves[nm] = '\0';
    fprintf(out, "Best path found (greedy trajectory; S G *=mine, arrows=move taken):\n");
    for (int r = 0; r < R; r++) {
        fprintf(out, "  ");
        for (int c = 0; c < C; c++) fprintf(out, " %c ", cell[r*C+c]);
        fprintf(out, "\n");
    }
    fprintf(out, "  moves: %s   (%d steps, return %ld, reached_goal=%s)\n",
            moves, steps, ri(total), reached ? "yes" : "no");
    free(cell); free(moves);
}

static void write_report(FILE *out) {
    long med = median_fp(ep_return, g_episodes);   /* self-scaling clip floor */
    long floor = 4 * med;                          /* (med is negative)        */
    if (floor > -20 * FP_ONE) floor = -20 * FP_ONE;
    char title[96];
    snprintf(title, sizeof title,
             "Reward vs episode  (per-episode return, clipped at %ld):", ri(floor));
    ascii_plot(out, title, "reward", ep_return, g_episodes, floor, 0);
    long dmax = 0;
    for (int i = 0; i < g_episodes; i++) if (ep_delta[i] > dmax) dmax = ep_delta[i];
    ascii_plot(out, "Bellman delta vs episode  (mean |TD error| per episode):",
               "|delta|", ep_delta, g_episodes, 0, dmax);
    print_best_path(out);
}

static void write_csv(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "warning: cannot write %s\n", path); return; }
    fprintf(f, "episode,return,abs_delta\n");
    for (int i = 0; i < g_episodes; i++)
        fprintf(f, "%d,%ld,%ld\n", i, ri(ep_return[i]), ri(ep_delta[i]));
    fclose(f);
}

/* Export layout, wind, mines, greedy path and BFS-optimal path for the SVG. */
static void write_path_file(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "warning: cannot write %s\n", path); return; }
    fprintf(f, "grid %d %d\n", R, C);
    fprintf(f, "start %d %d\n", START / C, START % C);
    fprintf(f, "goal %d %d\n", GOAL_R, GOAL_C);
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (is_mine_rc(r, c)) fprintf(f, "mine %d %d\n", r, c);
    for (int c = 0; c < C; c++) fprintf(f, "wind %d %d\n", c, wind[c]);

    /* greedy (learned) path */
    int s = START, steps = 0; long total = 0; int reached = 0;
    for (; steps < g_maxsteps; steps++) {
        int a = argmax_a(s);
        fprintf(f, "step %d %d %d\n", s / C, s % C, a);
        Step st = env_step(s, a);
        total += st.reward; s = st.next;
        if (st.done) { steps++; reached = 1; break; }
    }
    fprintf(f, "end %d %d %d %ld %d\n", s / C, s % C, steps, ri(total), reached);

    /* BFS ground-truth optimal path */
    for (int i = 0; i < opt_len; i++)
        fprintf(f, "opt %d %d %d\n",
                opt_states[i] / C, opt_states[i] % C, opt_acts[i]);
    fprintf(f, "optend %d %d %d %d %d\n",
            GOAL_R, GOAL_C, opt_len, -opt_len, opt_reachable);
    fclose(f);
}

/* ------------------------- World generation ----------------------------- */
/* Randomise wind + mines until the goal is reachable (verified by BFS).     */
static void generate_world(int n_mines) {
    for (int attempt = 0; attempt < 100000; attempt++) {
        for (int c = 0; c < C; c++) {
            unsigned int r = xrand() % 100;       /* wind 0/1/2, biased to 0 */
            wind[c] = (r < 12) ? 2 : (r < 38) ? 1 : 0;
        }
        for (int i = 0; i < NS; i++) mine[i] = 0;
        int placed = 0, guard = 0;
        while (placed < n_mines && guard < n_mines * 60 + 100) {
            int s = xrand() % NS;
            guard++;
            if (s == START || s == GOAL || mine[s]) continue;
            mine[s] = 1; placed++;
        }
        find_optimal();
        if (opt_reachable) return;                /* solvable map found */
    }
    /* extremely unlikely fallback: no mines */
    for (int i = 0; i < NS; i++) mine[i] = 0;
    find_optimal();
}

/* ------------------------------- main ----------------------------------- */
int main(int argc, char **argv) {
    R = (argc > 1) ? atoi(argv[1]) : 4;
    C = (argc > 2) ? atoi(argv[2]) : 6;
    if (R < 2) R = 2;
    if (R > 40) R = 40;
    if (C < 2) C = 2;
    if (C > 40) C = 40;
    NS = R * C;
    int n_mines = (argc > 3) ? atoi(argv[3]) : NS / 7;
    if (n_mines < 0) n_mines = 0;
    if (n_mines > NS - 3) n_mines = NS - 3;       /* leave room for S, G, path */
    if (argc > 4) rng_state = (unsigned)atoi(argv[4]) | 1u;

    START = (R - 1) * C + 0; GOAL_R = R - 1; GOAL_C = C - 1; GOAL = GOAL_R * C + GOAL_C;
    g_maxsteps = 6 * (R + C); if (g_maxsteps < 100) g_maxsteps = 100;
    g_episodes = 80 * NS;     if (g_episodes < 3000) g_episodes = 3000;
    if (g_episodes > 40000) g_episodes = 40000;

    wind = malloc(sizeof(int) * C);
    mine = malloc(sizeof(int) * NS);
    ep_return = malloc(sizeof(long) * g_episodes);
    ep_delta  = malloc(sizeof(long) * g_episodes);
    opt_states = malloc(sizeof(int) * (NS + 1));
    opt_acts   = malloc(sizeof(int) * (NS + 1));

    /* Configure the general-purpose agent: state = 2 features (row, col), one
       bin per cell so the mapping is exact. Action space = the 4 moves. */
    feat_spec[0].lo = 0; feat_spec[0].hi = R - 1; feat_spec[0].bins = R;
    feat_spec[1].lo = 0; feat_spec[1].hi = C - 1; feat_spec[1].bins = C;
    agent_Q = malloc(sizeof(ql_fp) * qlearn_qcount(feat_spec, 2, NA));
    qlearn_init(&agent, feat_spec, 2, NA, agent_Q, rng_state ^ 0x5bd1e995u);
    qlearn_set_params(&agent, ALPHA_FP, GAMMA_FP, 100);

    generate_world(n_mines);

    printf("Mines-with-Wind Q-learning  (pure integer / Q8.8 fixed point)\n");
    printf("grid=%dx%d  mines=%d  episodes=%d  maxsteps=%d  alpha=%d/256 gamma=%d/256\n",
           R, C, n_mines, g_episodes, g_maxsteps, ALPHA_FP, GAMMA_FP);
    printf("optimal path (BFS): %d steps (return %d)\n\n",
           opt_len, -opt_len);
    print_grid_legend(stdout);

    printf("\nLearning curve (greedy rollout):\n");
    printf("  %-9s %-7s %-8s %s\n", "episode", "steps", "reward", "reached_goal");
    int interval = g_episodes / 6; if (interval < 1) interval = 1;

    for (int ep = 0; ep < g_episodes; ep++) {
        qlearn_set_epsilon(&agent, 500 - (500 - 10) * ep / g_episodes); /* 0.50->0.01 */
        int s = START;
        long ep_rew = 0, ep_absdelta = 0; int ep_nsteps = 0;
        for (int t = 0; t < g_maxsteps; t++) {
            int a = qlearn_select_idx(&agent, s);
            Step st = env_step(s, a);
            long td = qlearn_update_idx(&agent, s, a, (ql_fp)st.reward, st.next, st.done);
            ep_rew += st.reward;
            ep_absdelta += (td >= 0) ? td : -td;
            ep_nsteps++;
            s = st.next;
            if (st.done) break;
        }
        ep_return[ep] = ep_rew;
        ep_delta[ep]  = ep_nsteps ? ep_absdelta / ep_nsteps : 0;

        if (ep % interval == 0 || ep == g_episodes - 1) {
            int steps, reached, hit;
            long total = eval_greedy(&steps, &reached, &hit);
            printf("  %-9d %-7d %-8ld %s\n", ep, steps, ri(total), reached ? "yes" : "no");
        }
    }

    printf("\n");
    print_policy();

    printf("\n");
    write_report(stdout);

    mkdir("results", 0755);
    FILE *rep = fopen("results/convergence.txt", "w");
    if (rep) { print_grid_legend(rep); fprintf(rep, "\n"); write_report(rep); fclose(rep); }
    write_csv("results/metrics.csv");
    write_path_file("results/path.txt");
    printf("Saved: results/convergence.txt, results/metrics.csv, results/path.txt\n");

    /* ----------------------------- VALIDATION --------------------------- */
    int steps, reached, hit;
    long total = eval_greedy(&steps, &reached, &hit);
    printf("\n--- Validation ---\n");
    printf("Greedy: reached_goal=%s  steps=%d  return=%ld  hit_mine=%s\n",
           reached ? "yes" : "no", steps, ri(total), hit ? "yes" : "no");
    printf("Optimal (BFS): steps=%d  return=%d\n", opt_len, -opt_len);

    /* PASS: the learned greedy policy reaches the goal without ever stepping on
       a mine, and is within a small factor of the true shortest path. */
    int pass = reached && !hit && (steps <= 2 * opt_len + 5);
    printf("RESULT: %s%s\n", pass ? "PASS" : "FAIL",
           (reached && steps == opt_len) ? "  (matched the optimal path)" : "");
    return pass ? 0 : 1;
}
