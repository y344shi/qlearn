/*
 * rl_harness.h - generic training & evaluation over the rl_agent interface.
 * =========================================================================
 * Drives ANY algorithm (that implements rl_agent) on ANY rl_env, using the
 * online one-call step protocol. Pure integer.
 */
#ifndef RL_HARNESS_H
#define RL_HARNESS_H

#include "rl_core.h"

#define RL_MAX_FEAT 8

/* Train for `total_steps` environment steps; epsilon anneals linearly from
   eps0 to eps1 (per-mille). Episodic envs reset automatically at episode end. */
static inline void rl_train(rl_env *env, rl_agent *ag, int total_steps, int eps0, int eps1){
    rl_fp feat[RL_MAX_FEAT];
    env->reset(env, feat);
    rl_fp reward = 0; int done = 0;
    for (int t = 0; t < total_steps; t++){
        ag->set_epsilon(ag, eps0 + (long)(eps1 - eps0) * t / total_steps);
        int a = ag->step(ag, reward, feat, done, 1);
        if (done){ env->reset(env, feat); reward = 0; done = 0; continue; }
        reward = env->step(env, a, feat, &done);
    }
    /* deliver the final pending transition if we stopped mid-episode */
    if (done) ag->step(ag, reward, feat, 1, 1);
}

/* Many greedy episodes on a (possibly stochastic) episodic env: average return
   and success rate (% of episodes that reached the goal, i.e. terminated).     */
typedef struct { int avg_return, success_pct; } rl_eval_stats;
static inline rl_eval_stats rl_eval_many(rl_env *env, rl_agent *ag, int episodes){
    long long ret = 0; int succ = 0;
    for (int e = 0; e < episodes; e++){
        rl_fp feat[RL_MAX_FEAT]; env->reset(env, feat);
        int steps = 0, done = 0; long long r = 0;
        while (steps < env->max_steps && !done){
            r += env->step(env, ag->act_greedy(ag, feat), feat, &done);
            steps++;
        }
        ret += r; if (done) succ++;
    }
    rl_eval_stats s; s.avg_return = (int)(ret / episodes / RL_FP_ONE);
    s.success_pct = succ * 100 / episodes; return s;
}

/* One greedy episode (for episodic envs): returns reached/steps/return(int).  */
typedef struct { int reached, steps, ret; } rl_rollout;
static inline rl_rollout rl_eval_episode(rl_env *env, rl_agent *ag){
    rl_fp feat[RL_MAX_FEAT];
    env->reset(env, feat);
    int steps = 0, done = 0; long long ret = 0;
    while (steps < env->max_steps && !done){
        int a = ag->act_greedy(ag, feat);
        ret += env->step(env, a, feat, &done);
        steps++;
    }
    rl_rollout r; r.reached = done; r.steps = steps; r.ret = (int)(ret / RL_FP_ONE);
    return r;
}

#endif /* RL_HARNESS_H */
