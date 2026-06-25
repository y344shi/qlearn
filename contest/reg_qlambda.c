#include "rl_env.h"
#include "algo_qlambda.h"
rl_agent reg_qlambda(const rl_feature *spec, int nf, int na, uint32_t seed){
    return qlambda_agent_make(spec, nf, na, seed);
}
