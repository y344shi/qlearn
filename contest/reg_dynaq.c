#include "rl_env.h"
#include "algo_dynaq.h"
rl_agent reg_dynaq(const rl_feature *spec, int nf, int na, uint32_t seed){
    return dynaq_agent_make(spec, nf, na, seed);
}
