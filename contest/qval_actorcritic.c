#include "rl_core.h"
#include "algo_actorcritic.h"
/* Read this agent's value estimate for (state, action) -- lets the visualizer
   compute a uniform Bellman residual. Own TU, so no static-symbol clashes. */
rl_fp qval_actorcritic(rl_agent *a, const rl_fp *feat, int action){
    aca_t *q = (aca_t *)a->ctx;
    int s = rl_state_of(q->spec, q->nfeat, feat);
    (void)action; return q->V[s];   /* critic state-value */
}
