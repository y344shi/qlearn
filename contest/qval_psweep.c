#include "rl_core.h"
#include "algo_psweep.h"
/* Read this agent's value estimate for (state, action) -- lets the visualizer
   compute a uniform Bellman residual. Own TU, so no static-symbol clashes. */
rl_fp qval_psweep(rl_agent *a, const rl_fp *feat, int action){
    psw_t *q = (psw_t *)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);
    return q->Q[(size_t)s*q->nact+action];
}
