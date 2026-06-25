#!/bin/sh
# Build, run, and generate all plots automatically.
#   ./run.sh
# Produces (in results/):
#   convergence.txt  - the ASCII plots + best path (written by ./cliff)
#   metrics.csv      - per-episode raw data        (written by ./cliff)
#   reward.svg       - reward vs episode           (written by plot_svg.py)
#   delta.svg        - Bellman delta vs episode    (written by plot_svg.py)
set -e
HERE=$(dirname "$0")
cd "$HERE"

# 1. build (honours $CC; falls back to gcc-12 then cc)
CC=${CC:-$(command -v cc >/dev/null 2>&1 && echo cc || echo gcc-12)}
echo "[build] $CC -Iinclude -> ./cliff"
"$CC" -O2 -std=c99 -Wall -Wextra -Iinclude -o cliff src/cliff_wind_qlearn.c

# 2. run (writes results/convergence.txt and results/metrics.csv)
#    extra args pass straight through, e.g.  ./run.sh 8 12 14 7  (rows cols mines seed)
echo "[run] ./cliff $*"
./cliff "$@"

# 3. render SVGs from the CSV if python3 is available (stdlib only)
if command -v python3 >/dev/null 2>&1; then
    echo "[plot] python3 tools/plot_svg.py"
    python3 tools/plot_svg.py results/metrics.csv results
else
    echo "[plot] python3 not found - skipping SVGs (ASCII plots still in results/convergence.txt)"
fi
echo "Done. See results/."
