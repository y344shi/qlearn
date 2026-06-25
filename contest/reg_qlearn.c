#include "rl_env.h"
#include "algo_qlearn.h"
rl_agent reg_qlearn(const rl_feature *spec, int nf, int na, uint32_t seed){
    return qlearn_agent_make(spec, nf, na, seed);
}
