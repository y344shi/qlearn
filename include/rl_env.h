/*
 * rl_env.h - umbrella header for the integer RL contest infrastructure.
 * =====================================================================
 * Include this to get the fixed-point core, the agent interface, the generic
 * training/eval harness, and all three tasks:
 *   - windy maze       (env_windy.h)   : episodic gridworld + mines + wind
 *   - frequency tuning (env_phone.h)   : continuing DVFS control
 *   - contest arena    (env_arena.h)   : slippery cliff-walk (discriminating)
 *
 * An algorithm only needs: implement the rl_agent vtable (see rl_core.h) using
 * INTEGER-ONLY math, then it can be trained and scored on every task.
 */
#ifndef RL_ENV_H
#define RL_ENV_H

#include "rl_core.h"
#include "rl_harness.h"
#include "env_windy.h"
#include "env_phone.h"
#include "env_arena.h"
#include "env_frozen.h"
#include "env_shift.h"
#include "env_swirl.h"

#endif /* RL_ENV_H */
