#!/usr/bin/env bash
# Interactive launcher for the integer-RL demos. Terminal menus only (no HTML).
# Asks, in order: demonstration method -> test/env -> agent(s) -> grid size.
set -e
cd "$(dirname "$0")"
CC=${CC:-$(command -v cc >/dev/null 2>&1 && echo cc || echo gcc-12)}
CF="-O2 -std=c99 -Wall -Wextra -Iinclude"
PS3=$'\n select> '

echo "=============================================="
echo " integer-RL interactive launcher"
echo "=============================================="

# ---- 1) demonstration method ----
echo; echo "[1] Demonstration method:"
select METHOD in "Scoreboard / ranking (console table)" "Plot (SVG file)" \
                 "Path visualizer: all agents vs DFS-optimal (SVG)"; do
  [ -n "$METHOD" ] && break; done

if [[ $METHOD == Path* ]]; then
  echo; echo "[2] Maze size (the visualizer always runs all 11 agents):"
  select SZ in "small (4x6)" "medium (6x9)" "large (8x12)" "custom"; do [ -n "$SZ" ] && break; done
  case "$SZ" in small*) R=4;C=6;M=4;; medium*) R=6;C=9;M=6;; large*) R=8;C=12;M=14;;
                custom*) read -rp " rows cols mines: " R C M;; esac
  $CC $CF -o pathviz contest/pathviz.c contest/reg_*.c contest/qval_*.c
  ./pathviz "$R" "$C" "$M"
  python3 tools/plot_svg.py results/metrics.csv results
  echo ">> results/viz_paths.svg  results/viz_rewards.svg  results/viz_delta.svg"
  exit 0
fi

# ---- 2) test / environment (filtered by method) ----
echo; echo "[2] Test / environment:"
if [[ $METHOD == Plot* ]]; then
  select TEST in "windy maze (convergence curves)" "phone tuner (DVFS trace)" \
                 "swirl escape (sensor/thrust trace)" "contest (cliff scoreboard)"; do
    [ -n "$TEST" ] && break; done
else
  select TEST in "windy maze" "cliff arena" "frozen lake" "shifting maze" \
                 "swirl escape" "phone tuner" "contest (all 11 on cliff)"; do
    [ -n "$TEST" ] && break; done
fi
case "$TEST" in
  windy*)        ENV=windy;;   "cliff arena"*) ENV=arena;;  frozen*)  ENV=frozen;;
  shifting*)     ENV=shift;;   swirl*)         ENV=swirl;;  phone*)   ENV=phone;;
  contest*)      ENV=contest;;
esac

# ---- 3) agent(s)  (only meaningful for the console scoreboard on a single env) ----
AGENT=all
if [[ $METHOD == Scoreboard* && $ENV != contest ]]; then
  echo; echo "[3] Agent(s):  (all = rank every algorithm)"
  select AGENT in all qlearn sarsa expsarsa doubleq qlambda sarsalambda \
                  dynaq mc nstep psweep actorcritic; do
    [ -n "$AGENT" ] && break; done
fi

# ---- 4) grid size  (only for grid environments) ----
P=()
if [[ $ENV == windy || $ENV == arena || $ENV == frozen ]]; then
  echo; echo "[4] Grid size:"
  select SZ in "small (4x6)" "medium (8x12)" "large (12x16)" "custom"; do
    [ -n "$SZ" ] && break; done
  case "$SZ" in
    small*)  R=4;  C=6;;   medium*) R=8;  C=12;;
    large*)  R=12; C=16;;  custom*) read -rp " rows cols: " R C;;
  esac
  P=("$R" "$C")
fi

echo; echo "----------------------------------------------"
PLOT="python3 tools/plot_svg.py results/metrics.csv results"

if [[ $METHOD == Plot* ]]; then
  case "$ENV" in
    windy)   $CC $CF -o cliff src/cliff_wind_qlearn.c; ./cliff "${P[@]:-8 12}" >/dev/null; $PLOT
             echo ">> results/reward.svg  results/delta.svg  results/path.svg";;
    phone)   $CC $CF -o phone_sim src/phone_sim.c; ./phone_sim >/dev/null; $PLOT
             echo ">> results/phone.svg";;
    swirl)   $CC $CF -o swirl_sim src/swirl_sim.c; ./swirl_sim >/dev/null; $PLOT
             echo ">> results/swirl.svg";;
    contest) $CC $CF -o contest_run contest/contest.c contest/reg_*.c; ./contest_run >/dev/null; $PLOT
             echo ">> results/contest_scores.svg  results/contest_curves.svg";;
  esac
  echo "(open the .svg in your browser/IDE)"
else
  if [[ $ENV == contest ]]; then
    $CC $CF -o contest_run contest/contest.c contest/reg_*.c; ./contest_run
  else
    $CC $CF -o runner contest/runner.c contest/reg_*.c; ./runner "$ENV" "$AGENT" "${P[@]}"
  fi
fi
