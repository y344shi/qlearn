#include "rl_env.h"
#include "algo_sarsalambda.h"
rl_agent reg_sarsalambda(const rl_feature *spec, int nf, int na, uint32_t seed){
    return sarsalambda_agent_make(spec, nf, na, seed);
}
