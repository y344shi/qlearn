/*
 * rl_controller.h - plug-and-play wrapper for an async, init-once controller.
 * ==========================================================================
 * Wraps any rl_agent (Q-learning, SARSA, Dyna-Q, ...) into the shape a real
 * controller wants: you INITIALISE ONCE with the input dimensionality, the
 * number of actions, and a REWARD FUNCTION, then call ONE function per async
 * tick that (a) scores the previous action, (b) learns, and (c) returns the
 * next action. No floating point; the agent owns its Q-table.
 *
 * This is exactly the dfc_tuner_qlearn_misc.c lifecycle:
 *     qlm_init/qlm_start  ->  rl_controller_init(...)            // once
 *     handle_load_change  ->  action = rl_controller_tick(...)   // each event
 *     qlm_destroy         ->  rl_controller_free(...)
 *
 * The reward function is YOUR domain logic. It is called with the previous
 * feature vector + action and the current feature vector, and returns the Q8.8
 * reward earned by that previous action (e.g. -jank_penalty - power_cost):
 *
 *     rl_fp my_reward(const rl_fp *prev_feat, int prev_action,
 *                     const rl_fp *cur_feat, void *user) { ... }
 */
#ifndef RL_CONTROLLER_H
#define RL_CONTROLLER_H

#include "rl_core.h"
#ifndef RL_MAX_FEAT
#define RL_MAX_FEAT 8
#endif

typedef rl_fp (*rl_reward_fn)(const rl_fp *prev_feat, int prev_action,
                              const rl_fp *cur_feat, void *user);

typedef struct {
    rl_agent     *ag;
    rl_reward_fn  reward;
    void         *user;
    int           nfeat;
    int           explore;                 /* 1 = keep learning, 0 = frozen greedy */
    int           have_prev;
    int           prev_action;
    rl_fp         prev_feat[RL_MAX_FEAT];
} rl_controller;

/* Initialise once. `ag` must already be constructed (any algorithm's *_make).
   `reward` may be NULL if you prefer to pass rewards yourself via _tick_r().   */
static inline void rl_controller_init(rl_controller *c, rl_agent *ag,
                                      rl_reward_fn reward, void *user){
    c->ag = ag; c->reward = reward; c->user = user;
    c->nfeat = 0; c->explore = 1; c->have_prev = 0; c->prev_action = 0;
}
static inline void rl_controller_set_explore(rl_controller *c, int on){ c->explore = on; }
static inline void rl_controller_set_epsilon(rl_controller *c, int milli){ c->ag->set_epsilon(c->ag, milli); }

/* One async tick with an EXPLICIT reward for the previous action (use this if
   you compute the reward yourself). Returns the action to apply now.          */
static inline int rl_controller_tick_r(rl_controller *c, rl_fp reward_prev,
                                       const rl_fp *cur_feat){
    int a = c->ag->step(c->ag, reward_prev, cur_feat, /*done=*/0, c->explore);
    for (int i = 0; i < RL_MAX_FEAT; i++) c->prev_feat[i] = cur_feat[i];
    c->prev_action = a; c->have_prev = 1;
    return a;
}

/* One async tick using the registered reward function. Pass the latest feature
   vector; it scores the previous action, learns, and returns the next action.  */
static inline int rl_controller_tick(rl_controller *c, const rl_fp *cur_feat){
    rl_fp r = 0;
    if (c->have_prev && c->reward)
        r = c->reward(c->prev_feat, c->prev_action, cur_feat, c->user);
    return rl_controller_tick_r(c, r, cur_feat);
}

/* End an episode (episodic tasks only); harmless for continuing controllers.   */
static inline void rl_controller_end_episode(rl_controller *c, rl_fp reward_prev){
    if (c->have_prev) c->ag->step(c->ag, reward_prev, c->prev_feat, /*done=*/1, c->explore);
    c->have_prev = 0;
}

/* ---- convenience: build a feature spec from plain dimensions ----
   Fill `spec[i] = {lo[i], hi[i], bins[i]}`. Lets you say "3 inputs with these
   ranges, N outputs" without hand-writing the struct.                          */
static inline void rl_make_spec(rl_feature *spec, int n_inputs,
                                const rl_fp *lo, const rl_fp *hi, const int *bins){
    for (int i = 0; i < n_inputs; i++){ spec[i].lo = lo[i]; spec[i].hi = hi[i]; spec[i].bins = bins[i]; }
}

#endif /* RL_CONTROLLER_H */
