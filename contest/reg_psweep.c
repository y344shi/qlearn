#include "rl_env.h"
#include "algo_psweep.h"
rl_agent reg_psweep(const rl_feature *spec, int nf, int na, uint32_t seed){
    return psweep_agent_make(spec, nf, na, seed);
}
