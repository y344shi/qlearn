# tools/ — plotting

`plot_svg.py` renders the CSVs emitted by the demos into SVG, using only the
Python standard library (no matplotlib/numpy/gnuplot). It auto-detects whichever
CSVs are present in the results directory and writes the matching SVGs.

    python3 tools/plot_svg.py results/metrics.csv results

| Input CSV | Output SVG(s) |
|-----------|---------------|
| `metrics.csv` | `reward.svg`, `delta.svg` (gridworld convergence) |
| `path.txt` | `path.svg` (learned vs optimal path) |
| `phone_trace.csv` | `phone.svg` |
| `swirl_trace.csv` | `swirl.svg` |
| `contest_scores.csv`, `contest_curves.csv` | `contest_scores.svg`, `contest_curves.svg` |
| `viz_meta.csv`, `viz_paths.csv`, `viz_curves.csv` | `viz_paths.svg`, `viz_rewards.svg`, `viz_delta.svg` |

An SVG is plain XML text; the script computes pixel coordinates and writes
`<rect>`/`<line>`/`<polyline>`/`<text>` elements directly.
