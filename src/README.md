# src/ — standalone demonstrations

Self-contained programs (each has its own `main`). All pure integer.

| File | What it does | Build / run |
|------|--------------|-------------|
| `cliff_wind_qlearn.c` | Gridworld (mines + wind); learns the optimal path, writes plots. | `gcc-12 -O2 -std=c99 -Iinclude -o cliff src/cliff_wind_qlearn.c && ./cliff 8 12 14 7` |
| `phone_sim.c` | Simulated CPU-frequency tuner; learns to avoid jank while saving power. | `... -o phone_sim src/phone_sim.c && ./phone_sim` |
| `swirl_sim.c` | Partially-observed control task; learns a hedge under a noisy sensor. | `... -o swirl_sim src/swirl_sim.c && ./swirl_sim` |
| `cliff_wind_qlearn_gpu.cu` | CUDA ensemble of agents (optional; needs an NVIDIA GPU). | `nvcc -ccbin g++-12 -O2 -gencode arch=compute_90,code=compute_90 -Iinclude -o cliff_gpu src/cliff_wind_qlearn_gpu.cu` |

These use the original `include/qlearn.h` directly; the reusable zoo lives in
`include/algo_*.h` and is exercised by `contest/` and `tunner/`.
