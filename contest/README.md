# contest/ — benchmarking and visualization

Programs that drive the eleven agents (from `include/algo_*.h`) on the shared
environments and compare them. Each `main` links the agent **registration
wrappers** so static symbols never clash.

| File | What it produces |
|------|------------------|
| `contest.c` | Ranks all 11 agents on the cliff arena (online return, eval, falls). `make contest`. |
| `tournament.c` | Ranks all 11 across the harder envs (arena/frozen/shift). `make tournament`. |
| `runner.c` | Runs **any one agent or `all`** on **any env**: `./runner swirl all`, `./runner windy dynaq 10 10 18`. |
| `pathviz.c` | Trains all 11, exports each learned path vs the BFS-optimal + reward/δ curves. `make viz`. |
| `reg_<algo>.c` | One tiny translation unit per algorithm exposing `reg_<algo>(...)` (its constructor). |
| `qval_<algo>.c` | One TU per algorithm exposing `qval_<algo>(...)` (reads its Q value; used by `pathviz`). |

Why `reg_*` / `qval_*` are separate files: the agent headers use file-local
(`static`) helpers, so two of them cannot be included in the same translation
unit. Each wrapper includes exactly one agent header and exports a non-static
function, letting the contest binaries reference every algorithm without clashes.

Build pattern (no `make` required):

    gcc-12 -O2 -std=c99 -Iinclude -o contest_run contest/contest.c contest/reg_*.c
    gcc-12 -O2 -std=c99 -Iinclude -o pathviz contest/pathviz.c contest/reg_*.c contest/qval_*.c
