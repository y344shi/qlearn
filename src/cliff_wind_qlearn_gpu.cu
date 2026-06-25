/*
 * cliff_wind_qlearn_gpu.cu
 * ------------------------
 * GPU version of the pure-integer Q-learning demo. Instead of training one
 * agent, it trains a whole ENSEMBLE of independent agents in parallel: one
 * CUDA thread per agent, each with its own private Q-table and its own integer
 * RNG seed, all learning the same "Cliff-with-Wind" gridworld.
 *
 * This does two things at once:
 *   1. Exercises the GPU (thousands of agents train concurrently).
 *   2. Validates the integer algorithm far more strongly than a single run --
 *      it shows the policy converges to the optimum across thousands of
 *      different random seeds, not just one lucky one.
 *
 * The per-agent maths is identical to src/cliff_wind_qlearn.c: Q8.8 fixed
 * point, xorshift32 PRNG, integer epsilon-greedy. No floating point is used in
 * the learning algorithm itself (only the wall-clock timer prints a float).
 *
 * Build (host compiler gcc-12; PTX for compute_90 is JIT'd to newer GPUs):
 *   nvcc -ccbin g++-12 -O2 -gencode arch=compute_90,code=compute_90 \
 *        -o cliff_gpu src/cliff_wind_qlearn_gpu.cu
 * Run:
 *   ./cliff_gpu
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* ------------------------- Fixed point (Q8.8) --------------------------- */
#define FP_SHIFT 8
#define FP_ONE   (1 << FP_SHIFT)

__host__ __device__ static inline long fx_mul(long a, long b) {
    long p = a * b;
    if (p >= 0) return  ((p + (FP_ONE / 2)) >> FP_SHIFT);
    else        return -(((-p) + (FP_ONE / 2)) >> FP_SHIFT);
}

/* ------------------------- Hyper-parameters ----------------------------- */
#define ALPHA_FP  64
#define GAMMA_FP  243
#define EPISODES  3000
#define MAX_STEPS 100
#define EPS_START 500
#define EPS_END   10

/* ------------------------- Environment ---------------------------------- */
#define R 4
#define C 6
#define NS (R * C)
#define NA 4

#define START  (3 * C + 0)
#define GOAL_R 3
#define GOAL_C 5
#define STEP_R  (-1 * FP_ONE)
#define CLIFF_R (-100 * FP_ONE)

__host__ __device__ static inline int is_cliff(int row, int col) {
    return (row == R - 1) && (col >= 1) && (col <= C - 2);
}

/* Deterministic transition; returns next state, writes reward and done flag. */
__host__ __device__ static int env_step(int state, int a, long *reward, int *done) {
    const int wind[C] = { 0, 0, 1, 1, 0, 0 };
    int row = state / C, col = state % C;
    int nr = row, nc = col;
    switch (a) {
        case 0: nr--; break; case 1: nc++; break;
        case 2: nr++; break; case 3: nc--; break;
    }
    if (nr < 0) nr = 0; if (nr > R - 1) nr = R - 1;
    if (nc < 0) nc = 0; if (nc > C - 1) nc = C - 1;
    nr -= wind[nc];
    if (nr < 0) nr = 0;

    if (is_cliff(nr, nc))               { *reward = CLIFF_R; *done = 0; return START; }
    if (nr == GOAL_R && nc == GOAL_C)   { *reward = STEP_R;  *done = 1; return nr * C + nc; }
    *reward = STEP_R; *done = 0; return nr * C + nc;
}

/* ------------------------- Integer PRNG --------------------------------- */
__host__ __device__ static inline unsigned int xr(unsigned int *s) {
    unsigned int x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x; return x;
}

/* ------------------------- Q-table helpers ------------------------------ */
__host__ __device__ static inline int argmax_a(const long *Q, int s) {
    int best = 0; long bv = Q[s * NA + 0];
    for (int a = 1; a < NA; a++) { long v = Q[s * NA + a]; if (v > bv) { bv = v; best = a; } }
    return best;
}
__host__ __device__ static inline long max_q(const long *Q, int s) {
    long bv = Q[s * NA + 0];
    for (int a = 1; a < NA; a++) { long v = Q[s * NA + a]; if (v > bv) bv = v; }
    return bv;
}

/* ------------------------- The training kernel -------------------------- */
/* One thread == one independent agent. Writes per-agent rollout results.    */
__global__ void train_kernel(int n_agents, unsigned int base_seed,
                             int *out_reached, int *out_steps, long *out_return) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_agents) return;

    long Q[NS * NA];
    for (int i = 0; i < NS * NA; i++) Q[i] = 0;

    /* distinct, non-zero seed per agent */
    unsigned int rng = (base_seed ^ (2654435761u * (unsigned)(tid + 1)));
    if (rng == 0) rng = 0x9e3779b9u;

    for (int ep = 0; ep < EPISODES; ep++) {
        int eps = EPS_START - (EPS_START - EPS_END) * ep / EPISODES;
        int s = START;
        for (int t = 0; t < MAX_STEPS; t++) {
            int a = ((int)(xr(&rng) % 1000) < eps) ? (int)(xr(&rng) % NA)
                                                   : argmax_a(Q, s);
            long r; int done;
            int s2 = env_step(s, a, &r, &done);

            long next_max = done ? 0 : max_q(Q, s2);
            long target   = r + fx_mul(GAMMA_FP, next_max);
            Q[s * NA + a] += fx_mul(ALPHA_FP, target - Q[s * NA + a]);

            s = s2;
            if (done) break;
        }
    }

    /* greedy validation rollout */
    int s = START, steps = 0, reached = 0; long total = 0;
    for (; steps < MAX_STEPS; steps++) {
        long r; int done;
        s = env_step(s, argmax_a(Q, s), &r, &done);
        total += r;
        if (done) { steps++; reached = 1; break; }
    }
    out_reached[tid] = reached;
    out_steps[tid]   = steps;
    out_return[tid]  = total;
}

/* ------------------------- Host driver ---------------------------------- */
#define CK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 2; } } while (0)

static int cmp_long(const void *a, const void *b) {
    long x = *(const long *)a, y = *(const long *)b;
    return (x > y) - (x < y);
}
static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}
static long ri(long v) { return (v >= 0) ? (v + FP_ONE / 2) / FP_ONE
                                         : -(((-v) + FP_ONE / 2) / FP_ONE); }

int main(int argc, char **argv) {
    int n_agents = (argc > 1) ? atoi(argv[1]) : 8192;
    if (n_agents < 1) n_agents = 8192;

    int dev = 0; cudaDeviceProp prop;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU ensemble Q-learning (pure integer / Q8.8)\n");
    printf("Device: %s  (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Agents: %d   episodes/agent: %d   alpha=%d/256 gamma=%d/256\n\n",
           n_agents, EPISODES, ALPHA_FP, GAMMA_FP);

    int  *d_reached, *d_steps; long *d_return;
    CK(cudaMalloc(&d_reached, n_agents * sizeof(int)));
    CK(cudaMalloc(&d_steps,   n_agents * sizeof(int)));
    CK(cudaMalloc(&d_return,  n_agents * sizeof(long)));

    int threads = 128, blocks = (n_agents + threads - 1) / threads;

    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    train_kernel<<<blocks, threads>>>(n_agents, 2463534242u, d_reached, d_steps, d_return);
    CK(cudaGetLastError());
    CK(cudaEventRecord(t1));
    CK(cudaDeviceSynchronize());
    float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));

    int  *h_reached = (int  *)malloc(n_agents * sizeof(int));
    int  *h_steps   = (int  *)malloc(n_agents * sizeof(int));
    long *h_return  = (long *)malloc(n_agents * sizeof(long));
    CK(cudaMemcpy(h_reached, d_reached, n_agents * sizeof(int),  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_steps,   d_steps,   n_agents * sizeof(int),  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_return,  d_return,  n_agents * sizeof(long), cudaMemcpyDeviceToHost));

    /* aggregate */
    int reached = 0, optimal = 0;
    for (int i = 0; i < n_agents; i++) {
        reached += h_reached[i];
        if (h_reached[i] && ri(h_return[i]) == -9 && h_steps[i] == 9) optimal++;
    }
    qsort(h_return, n_agents, sizeof(long), cmp_long);
    qsort(h_steps,  n_agents, sizeof(int),  cmp_int);
    long med_ret  = ri(h_return[n_agents / 2]);
    long best_ret = ri(h_return[n_agents - 1]);
    int  med_step = h_steps[n_agents / 2];

    printf("Trained %d agents on the GPU in %d ms (%.0f agents/sec)\n",
           n_agents, (int)(ms + 0.5f), n_agents / (ms / 1000.0f));
    printf("  reached goal : %d / %d  (%d%%)\n", reached, n_agents,
           100 * reached / n_agents);
    printf("  optimal (9 steps, return -9): %d / %d  (%d%%)\n", optimal, n_agents,
           100 * optimal / n_agents);
    printf("  greedy return  median=%ld  best=%ld\n", med_ret, best_ret);
    printf("  greedy steps   median=%d\n", med_step);

    /* VALIDATION: the overwhelming majority of independently-seeded integer
       agents must reach the goal, and the median agent must be optimal. */
    int pass = (100 * reached / n_agents >= 90) && (med_ret == -9) && (med_step == 9);
    printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");

    free(h_reached); free(h_steps); free(h_return);
    cudaFree(d_reached); cudaFree(d_steps); cudaFree(d_return);
    return pass ? 0 : 1;
}
