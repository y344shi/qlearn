/*
 * algo_nstep.h - n-step SARSA: on-policy n-step TD control as an rl_agent.
 * =======================================================================
 * On-policy n-step temporal-difference control (Sutton & Barto, S7.2), wrapped
 * in the uniform rl_agent vtable (see rl_core.h) using INTEGER-ONLY Q8.8 math.
 * Structure mirrors algo_sarsa.h. Where one-step SARSA bootstraps after a
 * SINGLE reward, n-step SARSA accumulates n real rewards before bootstrapping
 * off the action ACTUALLY taken n steps later:
 *
 *   G = R_{t+1} + gamma*R_{t+2} + ... + gamma^{n-1}*R_{t+n}
 *         + gamma^n * Q(S_{t+n}, A_{t+n})
 *   Q(S_tau,A_tau) += alpha * ( G - Q(S_tau,A_tau) )                  (tau=t-n+1)
 *
 * This is the bias/variance BRIDGE between one-step TD (n=1, low variance, high
 * bootstrap bias) and Monte-Carlo (n=infinity, unbiased sample return, high
 * variance): credit from a reward propagates n cells back per update instead of
 * one, which on a long sparse-reward path (the windy maze) reaches the start
 * dramatically faster. See docs/rl/nstep.md.
 *
 * IMPLEMENTATION: a circular buffer of the last (n+1) (state, action, reward)
 * entries. Each online step() fills in the reward of the previously remembered
 * (s,a) with reward_prev, chooses a' on-policy for the current state, pushes it,
 * and -- once n full rewards plus the bootstrap pair are available -- emits the
 * update for the entry n steps back (tau bookkeeping). On `done` we flush every
 * still-buffered entry with a bootstrap-free return (the tail of the episode).
 *
 * EXPLORATION via OPTIMISTIC INITIALISATION (Sutton & Barto S2.6), exactly as in
 * algo_sarsa.h: the windy maze's mines merely teleport to start (only the goal
 * terminates), so a near-greedy on-policy agent would loop forever and never see
 * the goal. Seeding Q(s,a)=100.0 (above any achievable return, all <= 0 here)
 * makes each visit drive a cell DOWN, so the greedy policy sweeps the least
 * visited actions until it discovers the goal -- the optimism IS the
 * exploration, and the behaviour policy can be held greedy (eps 0). We seed
 * Q0 = 80.0 (Q8.8), above every achievable return.
 *
 * gamma^i powers are precomputed once in Q8.8 at construction.
 */
#ifndef ALGO_NSTEP_H
#define ALGO_NSTEP_H

#include <stdlib.h>
#include "rl_core.h"

#define NSTEP_N 5                       /* steps to look ahead (bridge n) */
#define NSTEP_CAP (NSTEP_N + 1)         /* circular buffer capacity        */

typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *Q;                           /* [nstates * nact] Q8.8 */
    rl_fp alpha, gamma; int eps;        /* eps in per-mille */
    rl_fp gpow[NSTEP_CAP + 1];          /* gpow[i] = gamma^i in Q8.8 */
    rl_rng rng;
    /* circular buffer of recent transitions */
    int    bs[NSTEP_CAP];               /* state index */
    int    ba[NSTEP_CAP];               /* action */
    rl_fp  br[NSTEP_CAP];               /* reward FOLLOWING (bs,ba) */
    int    head, count;                 /* head = index of oldest entry */
    int    pending;                     /* 1 if newest entry awaits its reward */
} nst_t;

static inline int nst__argmax(nst_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv){ bv = r[k]; b = k; }
    return b;
}
static inline int nst__choose(nst_t *q, int s, int explore){
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps)
        return (int)(rl_rand(&q->rng) % q->nact);
    return nst__argmax(q, s);
}

/* index helpers into the circular buffer (head is oldest, +i moves newer) */
static inline int nst__at(nst_t *q, int i){ return (q->head + i) % NSTEP_CAP; }

/* Apply the n-step update to the OLDEST buffered entry (entry index 0), using
   `nsteps` real rewards (entries 0..nsteps-1) and, if boot==1, bootstrapping off
   Q(bs[nsteps], ba[nsteps]); if boot==0 (episode tail) the bootstrap is dropped.
   Then pop the oldest entry. */
static inline void nst__emit(nst_t *q, int nsteps, int boot){
    int i0 = nst__at(q, 0);
    rl_fp G = 0;
    for (int i = 0; i < nsteps; i++)
        G += rl_mul(q->gpow[i], q->br[nst__at(q, i)]);   /* gamma^i * R_{tau+i+1} */
    if (boot){
        int bj = nst__at(q, nsteps);
        rl_fp qb = q->Q[(size_t)q->bs[bj]*q->nact + q->ba[bj]];
        G += rl_mul(q->gpow[nsteps], qb);                /* gamma^n * Q(S,A) */
    }
    rl_fp *cell = &q->Q[(size_t)q->bs[i0]*q->nact + q->ba[i0]];
    *cell += rl_mul(q->alpha, G - *cell);
    q->head = nst__at(q, 1);                             /* pop oldest */
    q->count--;
}

static inline void nst__reset_buf(nst_t *q){ q->head = 0; q->count = 0; q->pending = 0; }

static inline int nst_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    nst_t *q = (nst_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);

    /* attach reward_prev to the newest (still-pending) entry */
    if (q->pending){
        int last = nst__at(q, q->count - 1);
        q->br[last] = reward_prev;
        q->pending = 0;
    }

    if (done){
        /* episode tail: flush every remaining entry with a bootstrap-free
           n-step return (its own rewards through the terminal reward). */
        while (q->count > 0) nst__emit(q, q->count, 0);
        nst__reset_buf(q);
        return -1;
    }

    /* choose a' on-policy for the current state, push (s,a') as newest entry */
    int act = nst__choose(q, s, explore);
    int slot = nst__at(q, q->count);
    q->bs[slot] = s; q->ba[slot] = act; q->count++;
    q->pending = 1;                                       /* reward arrives next call */

    /* once we hold n filled rewards + the bootstrap pair (count == n+1), the
       oldest entry is n steps back: emit its full n-step bootstrapped update. */
    if (q->count == NSTEP_CAP) nst__emit(q, NSTEP_N, 1);

    return act;
}

static inline int  nst_greedy(rl_agent *a, const rl_fp *feat){
    nst_t *q = (nst_t*)a->ctx; return nst__argmax(q, rl_state_of(q->spec, q->nfeat, feat));
}
static inline void nst_seteps(rl_agent *a, int m){ ((nst_t*)a->ctx)->eps = m; }
static inline void nst_destroy(rl_agent *a){ nst_t *q = (nst_t*)a->ctx; free(q->Q); free(q); }

static inline rl_agent nstep_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    nst_t *q = malloc(sizeof(nst_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    q->Q = malloc((size_t)q->nstates*nact*sizeof(rl_fp));
    /* OPTIMISTIC initial values: above any achievable return (all <= 0 here),
       which gives on-policy n-step SARSA its exploration drive (see header). */
    {
        rl_fp q0 = RL_INT(80);
        for (size_t i = 0; i < (size_t)q->nstates*nact; i++) q->Q[i] = q0;
    }
    q->alpha = 48;              /* ~0.1875 in Q8.8: small, n rewards already
                                   propagate credit n cells per update */
    q->gamma = 254;             /* ~0.992 in Q8.8: high so credit from the
                                   single -1/step signal reaches the start */
    q->eps = 0;
    /* precompute gamma^i in Q8.8 */
    q->gpow[0] = RL_FP_ONE;
    for (int i = 1; i <= NSTEP_CAP; i++) q->gpow[i] = rl_mul(q->gpow[i-1], q->gamma);
    nst__reset_buf(q);
    rl_seed(&q->rng, seed);
    rl_agent a; a.name = "nstep"; a.ctx = q;
    a.step = nst_step; a.act_greedy = nst_greedy;
    a.set_epsilon = nst_seteps; a.destroy = nst_destroy;
    return a;
}

#endif /* ALGO_NSTEP_H */
