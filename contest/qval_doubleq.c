#include "rl_core.h"
#include "algo_doubleq.h"
/* Read this agent's value estimate for (state, action) -- lets the visualizer
   compute a uniform Bellman residual. Own TU, so no static-symbol clashes. */
rl_fp qval_doubleq(rl_agent *a, const rl_fp *feat, int action){
    dq_t *q = (dq_t *)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);
    return q->QA[(size_t)s*q->nact+action] + q->QB[(size_t)s*q->nact+action];
}
