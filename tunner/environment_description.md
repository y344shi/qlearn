# Q-Learning Simulation Environment for Frequency Control

**File**: `uapps/tppmgr/freqmgr/tuner/dfc_tuner_qlearn_misc.c`
**Purpose**: Integer-based Q-learning verification on real hardware frequency control

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   sched_indicator (kernel)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  sched_ind_notify_load_change()                          │   │
│  │    type = cpufreq_tuner_type_of(cpu_id)                  │   │
│  │    if (type == QLEARN_MISC) → populate features struct   │   │
│  │    → kvic_trigger → userspace                            │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                              │ trigger data                      │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  sched_ind_listener (tppmgr-sched-in thread)              │   │
│  │  load_chg_data_handler(data)                              │   │
│  │    if (policy & HM_QLEARN) → qlearn_sum_update()          │   │
│  │    → pm_notifier_notify(g_load_change_notifier, data)    │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                              │ PM_NOTIFIER_LOAD_CHANGE           │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  qlearn_misc tuner (callback in tppmgr-sched-in thread)   │   │
│  │  handle_load_change(listener, data)                        │   │
│  │    → extract features from data->features                  │   │
│  │    → run Q-learning step (epsilon-greedy)                  │   │
│  │    → select action → set frequency via dfc_driver          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 2. When Is the Tuner Invoked?

The tuner's `handle_load_change()` callback is called **once per sched_ind trigger**. Based on log data, triggers arrive approximately every **0.6 seconds** (average sample interval). This is your Q-learning step interval.

**Lifecycle:**

```
echo qlearn_misc > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  → dfc_tuner framework calls:
      1. qlm_init(prop)         → alloc priv, register listener
      2. qlm_start(prop, ctx)   → add PM_NOTIFIER_LOAD_CHANGE listener
                                   set initial freq to min_freq
  → Kernel sends sched_ind triggers → handle_load_change() called
  → echo performance > .../scaling_governor
      1. qlm_stop(prop)         → stop
      2. qlm_destroy(prop)      → free
```

## 3. Feature Structure (Input to Q-Learning)

The `data->features` field is a `struct __sched_ind_qlearn_features` (defined in `kernel/include/mapi/uapi/hmkernel/power/sched_indicator.h`). Your handler receives a pointer to this struct.

### 3.1 Per-Cluster Features (3 clusters: C0=little, C1=mid, C2=big)

```c
struct {
    unsigned long long power_used;  // accumulated power
    unsigned int avg_load;          // avg CPU load (0-1024 scale)
    unsigned int top_load;          // max CPU load
    unsigned int min_load;          // min CPU load
    unsigned int max_tcb_load;      // max per-task load
    unsigned int irq_load;          // IRQ load
    unsigned int uclamp_min_util;   // uclamp min (0-1024)
    unsigned int uclamp_max_util;   // uclamp max (0-1024)
    unsigned int nr_running;        // CPUs with non-zero load
    unsigned int curr_power;        // current power draw
} cluster[SCHED_IND_MAX_NR_CLUSTER];
```

Pick `cluster[priv->cluster_id]` for the cluster Q-learning controls.

### 3.2 Global Features

```c
unsigned long long janky_frames;    // total jank since boot (monotonically increasing)
unsigned long long curr_time;       // current timestamp (us)
unsigned long long time_of_next_frame;  // expected next frame time (us)
unsigned int frame_interval_us;     // frame deadline (us)
unsigned int curr_refresh_rate;     // display refresh rate bin (0-5)
int frame_budget;                   // remaining budget (>0 = ahead, <0 = overshoot)
int frame_budget_avg;               // EMA of frame budget
int frame_progress;                 // % of frame deadline elapsed (0-100+)
unsigned int cur_checkpoint;        // current checkpoint index (0-15)
unsigned long long prev_window_exec;  // CPU time last frame consumed (ns)
unsigned long long curr_window_exec;  // CPU time current frame used so far (ns)
unsigned long long interval_ns;       // expected frame interval (ns)
```

### 3.3 MISC Snapshot (read-only, may be stale if MISC inactive)

```c
// g_misc_snapshot contains:
struct misc_cpu_snapshot {
    unsigned int load;           // MISC CPU load %
    unsigned int up_threshold;   // MISC up threshold (80)
    unsigned int down_threshold; // MISC down threshold (50)
    unsigned int green_freq;     // MISC target freq when overloaded (kHz)
    unsigned int floor_freq;     // MISC min freq (kHz)
    unsigned int cur_freq;       // MISC's last set freq (kHz)
};
struct misc_l3c_snapshot {
    unsigned int load;           // L3C bandwidth load % (can exceed 100)
    unsigned int target_freq;    // L3C target frequency (kHz)
};
struct misc_mem_snapshot {
    unsigned int cache_ratio;    // inst/cache-miss (higher = better)
    unsigned int target_freq;    // MEM controller target freq (kHz)
};
```

Access via `g_misc_snapshot` (needs `raw_mutex_lock(&g_misc_snapshot_lock)`).

## 4. Action Space (Frequency Selection)

The action is setting CPU frequency via:

```c
dfc_driver_set_freq(priv->prop, new_freq, FREQ_TABLE_CEIL_METHOD);
```

### 4.1 Discrete Actions from Frequency Table

The hardware has a fixed set of available frequencies. Read them from the prop:

```c
struct dfc_prop *prop = priv->prop;
// prop->freq_table[] has the available frequencies
// prop->freq_table_cnt = number of entries
// prop->min_freq / prop->max_freq = hardware limits
```

Example discrete actions (10 actions with HALF_ACTION skip pattern, like the original qlearn):

```c
// Map action index (0-9) to frequency table index
// Skip every other frequency (HALF_ACTION pattern)
// action 0 = min_freq, action 9 = max_freq
action_idx = action * (freq_table_cnt / 10);
new_freq = freq_table[action_idx];
```

From your log data, typical frequencies on C2 (big cluster):
- Min: 1239 MHz
- Max: 2620 MHz  
- Common: 1239, 1530, 1740, 1930, 2150, 2350, 2620 MHz

### 4.2 Action Constraints

- **Low load bypass**: If `min_load < threshold` (~50), force min frequency
- **Passthrough**: If `frame_budget < 0` (jank happening), use kernel's `clusters_data[cluster_id].target_freq`
- **Safety**: Always clamp to `[prop->min_freq, prop->max_freq]`

## 5. Reward Function

The reward is computed AFTER the frequency is set and we see the result in the NEXT trigger.

### 5.1 Primary Reward

```c
// From the trigger data:
bool frame_ok = (janky_frames == prev_janky_frames);  // no new jank
unsigned int power = curr_power;

// Reward: high when no jank AND low power
int reward;
if (frame_ok) {
    reward = (int)(1000 - power);  // 1000 - power_used, max reward when no jank
} else {
    reward = -(int)(power);  // penalize by power when jank happens
}
```

### 5.2 Shaped Reward (Optional)

Add budget and progress shaping:

```c
// Frame budget bonus: positive budget = good
int budget_bonus = max(frame_budget_avg, 0) * 5;

// Efficiency bonus: low power per MHz = efficient
int eff_bonus = (power > 0 && freq > 0) ? (1000 * power / freq) : 0;
// lower eff_bonus = more efficient
eff_bonus = 100 - min(eff_bonus, 100);

// Energy penalty: avoid high power even when jank-free
int energy_penalty = power / 10;

reward = base_reward + budget_bonus + eff_bonus - energy_penalty;
```

## 6. State Representation

### 6.1 Discrete State Binning

The original Q-learning uses a 3D state space. For your implementation:

```c
// 3D state: [refresh_rate_bin][time_to_frame_bin][avg_load_bin]

// Refresh rate bins (6):
//   0: 0Hz (no display), 1: ~30Hz, 2: ~60Hz, 3: ~72Hz, 4: ~90Hz, 5: ~120Hz
int rr_bin = min(curr_refresh_rate, 5);

// Time to next frame bins (4):
//   0: UNDEFINED, 1: > 10ms, 2: 5-10ms, 3: < 5ms
u64 time_to_frame = time_of_next_frame - curr_time;
int ttf_bin = 0;
if (time_to_frame < 5000) ttf_bin = 3;
else if (time_to_frame < 10000) ttf_bin = 2;
else if (time_to_frame < 200000) ttf_bin = 1;

// Load bins (3 per cluster, determined experimentally):
//   From your log data, C2 load ranges:
//   light: 0-300, medium: 300-500, heavy: 500-1024
int load_bin = (avg_load < 300) ? 0 : (avg_load < 500) ? 1 : 2;

int state_idx = rr_bin * 12 + ttf_bin * 3 + load_bin;
// Total: 6 * 4 * 3 = 72 states per cluster
```

### 6.2 Alternative: Combined Feature State

For richer representation, use multiple features:

```c
// Combine load_bin + fbud_bin + progress_bin
int load_bin = discretize(curr_power, pwr_thresholds);  // 3 bins
int fba_bin = (frame_budget_avg < -5) ? 0 : (frame_budget_avg < 5) ? 1 : 2;
int prog_bin = (frame_progress < 30) ? 0 : (frame_progress < 70) ? 1 : 2;
int state = load_bin * 9 + fba_bin * 3 + prog_bin;  // 27 states
```

## 7. Integer Q-Learning Algorithm

Since the target environment may not support floating point, use fixed-point (`fix64.h` is available in the repo).

### 7.1 Q-Table

```c
#include "fix64.h"

#define NUM_STATES  72
#define NUM_ACTIONS 10
#define NUM_CLUSTERS 3

fix64_t q_table[NUM_CLUSTERS][NUM_STATES][NUM_ACTIONS];
```

### 7.2 Algorithm Parameters (from original qlearn)

```c
fix64_t learning_rate = fix64_from_int(20) / 100;   // 0.20
fix64_t discount    = fix64_from_int(95) / 100;      // 0.95
fix64_t epsilon     = fix64_from_int(10) / 100;      // 0.10 (exploration rate)
```

### 7.3 Action Selection (Epsilon-Greedy + UCB)

```c
int select_action(struct tuner_priv *priv, int state)
{
    fix64_t r = fix64_random();  // [0, 1)
    if (r < epsilon) {
        // Explore: random action
        return rand() % NUM_ACTIONS;
    }
    
    // Exploit: highest Q-value
    int best = 0;
    fix64_t best_val = q_table[cluster_id][state][0];
    for (int a = 1; a < NUM_ACTIONS; a++) {
        if (q_table[cluster_id][state][a] > best_val) {
            best_val = q_table[cluster_id][state][a];
            best = a;
        }
    }
    return best;
}
```

### 7.4 Q-Value Update (Bellman Equation)

```c
void update_q(struct tuner_priv *priv, int state, int action, int reward, int next_state)
{
    fix64_t cur_q = q_table[cluster_id][state][action];
    
    // Find max Q for next state
    fix64_t max_next_q = q_table[cluster_id][next_state][0];
    for (int a = 1; a < NUM_ACTIONS; a++) {
        if (q_table[cluster_id][next_state][a] > max_next_q)
            max_next_q = q_table[cluster_id][next_state][a];
    }
    
    // td_target = reward + discount * max_next_q
    fix64_t td_target = fix64_add(fix64_from_int(reward), 
                                   fix64_mul(discount, max_next_q));
    
    // td_error = td_target - cur_q
    fix64_t td_error = fix64_sub(td_target, cur_q);
    
    // new_q = cur_q + learning_rate * td_error
    fix64_t new_q = fix64_add(cur_q, fix64_mul(learning_rate, td_error));
    
    q_table[cluster_id][state][action] = new_q;
}
```

### 7.5 End-to-End Step

```c
static int handle_load_change(const struct pm_listener *listener, void *data)
{
    // 1. Extract features from data
    struct __sched_ind_qlearn_features *f = &((struct __sched_ind_load_chg_data *)data)->features;
    
    // 2. Discretize into state
    int state = discretize_state(f, priv->cluster_id);
    
    // 3. Compute reward from previous action
    int reward = compute_reward(priv->prev_jank, f->janky_frames, f->cluster[priv->cluster_id].curr_power);
    
    // 4. If this isn't the first step, update Q
    if (priv->prev_state >= 0)
        update_q(priv, priv->prev_state, priv->prev_action, reward, state);
    
    // 5. Select next action
    int action = select_action(priv, state);
    unsigned int new_freq = action_to_freq(priv, action);
    
    // 6. Apply frequency
    dfc_driver_set_freq(priv->prop, new_freq, FREQ_TABLE_CEIL_METHOD);
    
    // 7. Save state for next iteration
    priv->prev_state = state;
    priv->prev_action = action;
    priv->prev_jank = f->janky_frames;
    priv->prev_power = f->cluster[priv->cluster_id].curr_power;
}
```

## 8. Exploration Strategies

### 8.1 Epsilon-Greedy (Simple)
```c
if (random < epsilon) explore else exploit
```

### 8.2 UCB (Upper Confidence Bound, used in original qlearn)
```c
// visit_count[a] = how many times action a was tried
fix64_t ucb_value = q_table[state][a] + UCB_CONST * sqrt(log(total_visits) / visit_count[a]);
```

## 9. Persistence

Save/load Q-table to persist learned behavior across reboots:

```c
#define QTABLE_PATH "/data/qtable%d.bin"

void save_qtable(struct tuner_priv *priv, int cluster_id)
{
    char path[64];
    snprintf(path, sizeof(path), QTABLE_PATH, cluster_id);
    vfs_write(path, &q_table[cluster_id], sizeof(q_table[cluster_id]));
}

void load_qtable(struct tuner_priv *priv, int cluster_id)
{
    char path[64];
    snprintf(path, sizeof(path), QTABLE_PATH, cluster_id);
    vfs_read(path, &q_table[cluster_id], sizeof(q_table[cluster_id]));
}
```

## 10. Logging for Debug

```c
hm_error("QLSIM: state=%d action=%d reward=%d freq=%u jank=%llu power=%u fba=%d prog=%d",
         state, action, reward, new_freq, f->janky_frames,
         f->cluster[cid].curr_power, f->frame_budget_avg, f->frame_progress);
```

## 11. Simulation Verification Checklist

- [ ] Tuner switches to `qlearn_misc` via sysfs
- [ ] `handle_load_change()` is called on each sched_ind trigger (~0.6s)
- [ ] Features are extracted correctly (compare to known QLFEAT log values)
- [ ] State discretization produces 72 unique states across all clusters
- [ ] Actions map to distinct frequencies
- [ ] Frequency changes are visible in `/sys/.../scaling_cur_freq`
- [ ] Q-values converge (replay log shows action selection stabilizes)
- [ ] Jank rate decreases over time (Q-learning learns to avoid jank)
- [ ] Power decreases over time (Q-learning learns efficiency)
- [ ] Q-table persists across `stop` → `start` cycles
- [ ] Triangle wave test produces expected frequency oscillation

## 12. Quick Start

1. Build userspace with the new tuner
2. Switch to test tuner: `echo qlearn_misc > /sys/.../scaling_governor`
3. Monitor: `dmesg | grep QLSIM`
4. Check frequency: `cat /sys/.../scaling_cur_freq`
5. For triangle wave test, remove the PM listener and use the periodic thread from the original `triangle_thread_fn`
