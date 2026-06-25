#include "rl_env.h"
#include "algo_mc.h"
rl_agent reg_mc(const rl_feature *spec, int nf, int na, uint32_t seed){
    return mc_agent_make(spec, nf, na, seed);
}
