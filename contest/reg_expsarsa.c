#include "rl_env.h"
#include "algo_expsarsa.h"
rl_agent reg_expsarsa(const rl_feature *spec, int nf, int na, uint32_t seed){
    return expsarsa_agent_make(spec, nf, na, seed);
}
