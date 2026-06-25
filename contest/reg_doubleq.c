#include "rl_env.h"
#include "algo_doubleq.h"
rl_agent reg_doubleq(const rl_feature *spec, int nf, int na, uint32_t seed){
    return doubleq_agent_make(spec, nf, na, seed);
}
