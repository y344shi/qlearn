#!/bin/sh
# Build script that does not require `make`.
#   ./build.sh        -> build the CPU (pure-integer C) demo
#   ./build.sh gpu    -> also build the CUDA ensemble demo (needs nvcc + GPU)
#
# Override the compilers via environment variables if needed:
#   CC=gcc-12  NVCC=nvcc  NVCCBIN=g++-12  ./build.sh gpu
set -e

CC=${CC:-cc}
NVCC=${NVCC:-nvcc}
NVCCBIN=${NVCCBIN:-g++-12}
# Blackwell (sm_120) is newer than CUDA 12.0; emit compute_90 PTX and let the
# driver JIT it to the installed GPU at load time.
NVFLAGS=${NVFLAGS:--O2 -gencode arch=compute_90,code=compute_90}

echo "[cpu] $CC -O2 -std=c99 -Wall -Iinclude -o cliff src/cliff_wind_qlearn.c"
"$CC" -O2 -std=c99 -Wall -Wextra -Iinclude -o cliff src/cliff_wind_qlearn.c
echo "[cpu] built ./cliff"

echo "[lib] $CC -Iinclude -o agent_demo tunner/qlearn_agent_example.c"
"$CC" -O2 -std=c99 -Wall -Wextra -Iinclude -o agent_demo tunner/qlearn_agent_example.c
echo "[lib] built ./agent_demo (qlearn.h feature-vector example)"

echo "[sim] $CC -Iinclude -o phone_sim src/phone_sim.c"
"$CC" -O2 -std=c99 -Wall -Wextra -Iinclude -o phone_sim src/phone_sim.c
echo "[sim] built ./phone_sim (simulated-phone DVFS tuner)"

if [ "$1" = "gpu" ]; then
    echo "[gpu] $NVCC -ccbin $NVCCBIN $NVFLAGS -o cliff_gpu src/cliff_wind_qlearn_gpu.cu"
    # shellcheck disable=SC2086
    "$NVCC" -ccbin "$NVCCBIN" $NVFLAGS -o cliff_gpu src/cliff_wind_qlearn_gpu.cu
    echo "[gpu] built ./cliff_gpu"
fi
