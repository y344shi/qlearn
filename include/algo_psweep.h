/*
 * algo_psweep.h - Prioritized Sweeping: model-based planning with a priority
 * ==========================================================================
 * queue, wrapped in the uniform rl_agent vtable (see rl_core.h) using
 * INTEGER-ONLY Q8.8 math. Structure mirrors algo_qlearn.h.
 *
 * One-step Q-learning is model-FREE: every sampled transition is used exactly
 * once and then thrown away. Dyna-Q (Sutton & Barto, 2nd ed., S8.2) adds a
 * learned MODEL and then, after each real step, performs N *planning* updates on
 * RANDOMLY remembered transitions -- reusing experience to propagate value
 * faster. Prioritized Sweeping (S8.4) keeps Dyna's model but makes the planning
 * SMART: instead of sampling uniformly, it focuses computation where it matters
 * most -- on the transitions whose values are likely to change a lot.
 *
 * The mechanism is a PRIORITY QUEUE keyed by the magnitude of the TD error a
 * (s,a) pair would experience:
 *
 *     P(s,a) = | r + gamma * max_a' Q(s',a') - Q(s,a) |
 *
 * After each real transition we update the deterministic model, compute P for
 * the just-seen (s,a), and -- if P > theta (a small integer threshold) -- push
 * it. Then for up to N iterations we POP the highest-priority pair, apply the
 * Q-learning update to it, and look at its PREDECESSORS: every (sbar,abar) whose
 * model says it leads into s. Each predecessor's new priority is computed and,
 * if it exceeds theta, pushed (or its key raised if already queued). Big value
 * changes thus ripple BACKWARD along the model graph exactly where they are
 * needed, in priority order -- so the few updates that move the needle happen
 * first. This is "backward focusing".
 *
 * Model (deterministic):  r(s,a), s'(s,a), done(s,a), seen(s,a).
 * Predecessors are found by scanning the (small) model: any seen (sbar,abar)
 * with s'(sbar,abar) == s is a predecessor of s.
 *
 * Update (the same Q-learning backup, just applied in priority order):
 *     delta   = r + gamma * max_a' Q(s',a') - Q(s,a)   (0 bootstrap if done)
 *     Q(s,a) += alpha * delta
 */
#ifndef ALGO_PSWEEP_H
#define ALGO_PSWEEP_H

#include <stdlib.h>
#include "rl_core.h"

/* Tunables (overridable at compile time for experimentation). */
#ifndef PSW_ALPHA
#define PSW_ALPHA RL_FRAC(1,4)      /* 0.25 learning rate */
#endif
#ifndef PSW_GAMMA
#define PSW_GAMMA 230               /* ~0.90 discount in Q8.8 */
#endif
#ifndef PSW_THETA
#define PSW_THETA RL_FRAC(1,8)      /* ~0.125 priority threshold */
#endif
#ifndef PSW_NPLAN
#define PSW_NPLAN 20                /* planning updates per real step */
#endif

/* ----- integer binary max-heap of (priority, sa) keyed by priority ------- */
/* `pos[sa]` is the heap index of pair `sa`, or -1 if not queued, so we can
   raise a pair's key in place instead of inserting a duplicate.             */
typedef struct {
    rl_fp *prio;   /* heap[i].priority  (Q8.8, always >= 0) */
    int   *sa;     /* heap[i].sa        (flat s*nact+a index) */
    int   *pos;    /* pos[sa] = heap index of sa, or -1 */
    int    n;      /* number of entries currently in the heap */
} psw_pq;

static inline void psw_pq__swap(psw_pq *h, int i, int j){
    rl_fp tp = h->prio[i]; h->prio[i] = h->prio[j]; h->prio[j] = tp;
    int   ts = h->sa[i];   h->sa[i]   = h->sa[j];   h->sa[j]   = ts;
    h->pos[h->sa[i]] = i;  h->pos[h->sa[j]] = j;
}
static inline void psw_pq__up(psw_pq *h, int i){
    while (i > 0){
        int p = (i - 1) / 2;
        if (h->prio[p] >= h->prio[i]) break;
        psw_pq__swap(h, i, p); i = p;
    }
}
static inline void psw_pq__down(psw_pq *h, int i){
    for (;;){
        int l = 2*i + 1, r = 2*i + 2, m = i;
        if (l < h->n && h->prio[l] > h->prio[m]) m = l;
        if (r < h->n && h->prio[r] > h->prio[m]) m = r;
        if (m == i) break;
        psw_pq__swap(h, i, m); i = m;
    }
}
/* Push `sa` with priority `p`, or raise its key if already queued with less. */
static inline void psw_pq__push(psw_pq *h, int sa, rl_fp p){
    int i = h->pos[sa];
    if (i >= 0){                         /* already queued: keep the larger key */
        if (p > h->prio[i]){ h->prio[i] = p; psw_pq__up(h, i); }
        return;
    }
    i = h->n++;
    h->prio[i] = p; h->sa[i] = sa; h->pos[sa] = i;
    psw_pq__up(h, i);
}
static inline int psw_pq__pop(psw_pq *h){    /* returns sa of the max, or -1 */
    if (h->n == 0) return -1;
    int top = h->sa[0];
    h->pos[top] = -1;
    h->n--;
    if (h->n > 0){
        h->prio[0] = h->prio[h->n]; h->sa[0] = h->sa[h->n];
        h->pos[h->sa[0]] = 0;
        psw_pq__down(h, 0);
    }
    return top;
}

/* ------------------------------ agent ----------------------------------- */
typedef struct {
    const rl_feature *spec; int nfeat, nact, nstates;
    rl_fp *Q;                       /* [nstates * nact] Q8.8 action values */
    rl_fp *model_r;                 /* [nstates * nact] learned reward (Q8.8) */
    int   *model_s2;                /* [nstates * nact] learned successor state */
    unsigned char *model_seen;      /* [nstates * nact] 1 if (s,a) ever taken */
    unsigned char *model_done;      /* [nstates * nact] 1 if (s,a) is terminal */
    psw_pq pq;                      /* priority queue over (s,a) pairs */
    rl_fp alpha, gamma, theta;      /* Q8.8; theta = priority threshold */
    int nplan;                      /* planning iterations per real step */
    int eps;                        /* exploration rate, per-mille */
    rl_rng rng;
    int last_s, last_a, have;
} psw_t;

static inline int psw__argmax(psw_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; int b = 0; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv){ bv = r[k]; b = k; }
    return b;
}
static inline rl_fp psw__maxq(psw_t *q, int s){
    rl_fp *r = &q->Q[(size_t)s*q->nact]; rl_fp bv = r[0];
    for (int k = 1; k < q->nact; k++) if (r[k] > bv) bv = r[k];
    return bv;
}
static inline rl_fp psw__abs(rl_fp x){ return x < 0 ? -x : x; }

/* Priority of (s,a) = |TD error| under the current model & Q. */
static inline rl_fp psw__priority(psw_t *q, int s, int a){
    size_t i = (size_t)s*q->nact + a;
    rl_fp nmax = q->model_done[i] ? 0 : rl_mul(q->gamma, psw__maxq(q, q->model_s2[i]));
    rl_fp target = q->model_r[i] + nmax;
    return psw__abs(target - q->Q[i]);
}

/* Apply one Q-learning backup to (s,a) using the model; return its TD error. */
static inline void psw__backup(psw_t *q, int s, int a){
    size_t i = (size_t)s*q->nact + a;
    rl_fp nmax = q->model_done[i] ? 0 : rl_mul(q->gamma, psw__maxq(q, q->model_s2[i]));
    rl_fp target = q->model_r[i] + nmax;
    q->Q[i] += rl_mul(q->alpha, target - q->Q[i]);
}

/* The planning loop: pop the most-urgent pair, back it up, then re-prioritise
   all of its predecessors (those whose model successor is s). */
static inline void psw__plan(psw_t *q){
    for (int it = 0; it < q->nplan; it++){
        int sa = psw_pq__pop(&q->pq);
        if (sa < 0) break;                       /* queue empty: nothing urgent */
        int s = sa / q->nact, a = sa % q->nact;
        psw__backup(q, s, a);                    /* the high-priority update */

        /* backward focusing: scan the model for predecessors (sbar,abar)->s */
        for (int sbar = 0; sbar < q->nstates; sbar++){
            for (int abar = 0; abar < q->nact; abar++){
                size_t j = (size_t)sbar*q->nact + abar;
                if (!q->model_seen[j] || q->model_done[j]) continue;
                if (q->model_s2[j] != s) continue;
                rl_fp p = psw__priority(q, sbar, abar);
                if (p > q->theta) psw_pq__push(&q->pq, (int)j, p);
            }
        }
    }
}

static inline int psw_step(rl_agent *a, rl_fp reward_prev, const rl_fp *feat, int done, int explore){
    psw_t *q = (psw_t*)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);

    /* 1. learn the MODEL from the pending (last_s,last_a)->(reward_prev, s) and
          seed the priority queue with its TD error. */
    if (q->have){
        size_t i = (size_t)q->last_s*q->nact + q->last_a;
        q->model_r[i]    = reward_prev;
        q->model_s2[i]   = s;
        q->model_done[i] = (unsigned char)(done ? 1 : 0);
        q->model_seen[i] = 1;
        rl_fp p = psw__priority(q, q->last_s, q->last_a);
        if (p > q->theta) psw_pq__push(&q->pq, (int)i, p);
        psw__plan(q);                            /* focused planning sweep */
    }

    if (done){ q->have = 0; return -1; }

    int act;
    if (explore && (int)(rl_rand(&q->rng) % 1000) < q->eps) act = (int)(rl_rand(&q->rng) % q->nact);
    else act = psw__argmax(q, s);
    q->last_s = s; q->last_a = act; q->have = 1;
    return act;
}

static inline int psw_greedy(rl_agent *a, const rl_fp *feat){
    psw_t *q = (psw_t*)a->ctx;
    return psw__argmax(q, rl_state_of(q->spec, q->nfeat, feat));
}
static inline void psw_seteps(rl_agent *a, int m){ ((psw_t*)a->ctx)->eps = m; }
static inline void psw_destroy(rl_agent *a){
    psw_t *q = (psw_t*)a->ctx;
    free(q->pq.prio); free(q->pq.sa); free(q->pq.pos);
    free(q->model_done); free(q->model_seen); free(q->model_s2); free(q->model_r);
    free(q->Q); free(q);
}

static inline rl_agent psweep_agent_make(const rl_feature *spec, int nfeat, int nact, uint32_t seed){
    psw_t *q = malloc(sizeof(psw_t));
    q->spec = spec; q->nfeat = nfeat; q->nact = nact;
    q->nstates = rl_state_count(spec, nfeat);
    size_t n = (size_t)q->nstates * nact;
    q->Q          = calloc(n, sizeof(rl_fp));
    q->model_r    = calloc(n, sizeof(rl_fp));
    q->model_s2   = calloc(n, sizeof(int));
    q->model_seen = calloc(n, sizeof(unsigned char));
    q->model_done = calloc(n, sizeof(unsigned char));
    q->pq.prio = malloc(n * sizeof(rl_fp));
    q->pq.sa   = malloc(n * sizeof(int));
    q->pq.pos  = malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++) q->pq.pos[i] = -1;
    q->pq.n = 0;

    q->alpha = PSW_ALPHA;           /* learning rate (Q8.8) */
    q->gamma = PSW_GAMMA;           /* discount (Q8.8) */
    q->theta = PSW_THETA;           /* small priority threshold */
    q->nplan = PSW_NPLAN;           /* planning updates per real step */
    q->eps = 100; q->have = 0;
    rl_seed(&q->rng, seed);

    rl_agent a; a.name = "psweep"; a.ctx = q;
    a.step = psw_step; a.act_greedy = psw_greedy;
    a.set_epsilon = psw_seteps; a.destroy = psw_destroy;
    return a;
}

#endif /* ALGO_PSWEEP_H */
