#include "rl_env.h"
#include "algo_actorcritic.h"
rl_agent reg_actorcritic(const rl_feature *spec, int nf, int na, uint32_t seed){
    return actorcritic_agent_make(spec, nf, na, seed);
}
