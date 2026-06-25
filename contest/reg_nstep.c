#include "rl_env.h"
#include "algo_nstep.h"
rl_agent reg_nstep(const rl_feature *spec, int nf, int na, uint32_t seed){
    return nstep_agent_make(spec, nf, na, seed);
}
