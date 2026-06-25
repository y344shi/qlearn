CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Iinclude
BIN      = cliff
SRC      = src/cliff_wind_qlearn.c
AGENTBIN = agent_demo
AGENTSRC = tunner/qlearn_agent_example.c
PHONEBIN = phone_sim
PHONESRC = src/phone_sim.c
SWIRLBIN = swirl_sim
SWIRLSRC = src/swirl_sim.c
REGSRC      = $(wildcard contest/reg_*.c)
CONTESTBIN  = contest_run
CONTESTSRC  = contest/contest.c $(REGSRC)
TOURNEYBIN  = tournament_run
TOURNEYSRC  = contest/tournament.c $(REGSRC)

# --- GPU (optional) ---
# nvcc needs a GNU host compiler it supports (gcc/g++ <= 12 for CUDA 12.0).
# Blackwell (sm_120) is newer than CUDA 12.0 knows about, so we emit PTX for
# compute_90 and let the driver JIT-compile it to the installed GPU at load.
NVCC     ?= nvcc
NVCCBIN  ?= g++-12
NVFLAGS  ?= -O2 -gencode arch=compute_90,code=compute_90
GPUBIN    = cliff_gpu
GPUSRC    = src/cliff_wind_qlearn_gpu.cu

.PHONY: all run plots agent plug phone swirl viz contest tournament tests gpu run-gpu clean
all: $(BIN) $(AGENTBIN) $(PHONEBIN)

$(BIN): $(SRC) include/qlearn.h
	$(CC) $(CFLAGS) -o $(BIN) $(SRC)

# the general-purpose agent driven by a feature vector (tuner-style usage)
agent: $(AGENTBIN)
$(AGENTBIN): $(AGENTSRC) include/qlearn.h
	$(CC) $(CFLAGS) -o $(AGENTBIN) $(AGENTSRC)
	./$(AGENTBIN)

# plug-and-play async controller (init-once + reward fn): the tuner integration
plug: tuner_plug
	./tuner_plug
tuner_plug: tunner/tuner_plug_example.c include/rl_controller.h include/algo_qlearn.h
	$(CC) $(CFLAGS) -o tuner_plug tunner/tuner_plug_example.c

# simulated-phone DVFS environment: build, run, and plot the trace
phone: $(PHONEBIN)
	./$(PHONEBIN)
	python3 tools/plot_svg.py results/metrics.csv results
$(PHONEBIN): $(PHONESRC) include/qlearn.h
	$(CC) $(CFLAGS) -o $(PHONEBIN) $(PHONESRC)

run: $(BIN)
	./$(BIN)

# run + render SVG plots from the emitted CSV (pure-stdlib python, no deps)
plots: $(BIN)
	./$(BIN)
	python3 tools/plot_svg.py results/metrics.csv results

# swirl-escape physics env (phone DVFS + a fuel budget): build, run, plot
swirl: $(SWIRLBIN)
	./$(SWIRLBIN)
	python3 tools/plot_svg.py results/metrics.csv results
$(SWIRLBIN): $(SWIRLSRC) $(wildcard include/*.h)
	$(CC) $(CFLAGS) -o $(SWIRLBIN) $(SWIRLSRC)

# the RL algorithm CONTEST: all 11 integer agents ranked on the cliff arena
contest: $(CONTESTBIN)
	./$(CONTESTBIN)
	python3 tools/plot_svg.py results/metrics.csv results
$(CONTESTBIN): $(CONTESTSRC) $(wildcard include/*.h)
	$(CC) $(CFLAGS) -o $(CONTESTBIN) $(CONTESTSRC)

# path visualizer: every agent's learned path vs the BFS-optimal, + curves
viz: pathviz
	./pathviz 6 9 6
	python3 tools/plot_svg.py results/metrics.csv results
pathviz: contest/pathviz.c $(wildcard contest/reg_*.c) $(wildcard contest/qval_*.c) $(wildcard include/*.h)
	$(CC) $(CFLAGS) -o pathviz contest/pathviz.c $(wildcard contest/reg_*.c) $(wildcard contest/qval_*.c)

# the GRAND TOURNAMENT: all 11 agents across the harder envs (arena/frozen/shift)
tournament: $(TOURNEYBIN)
	./$(TOURNEYBIN)
$(TOURNEYBIN): $(TOURNEYSRC) $(wildcard include/*.h)
	$(CC) $(CFLAGS) -o $(TOURNEYBIN) $(TOURNEYSRC)

# build + run every algorithm's own test (windy maze + frequency tuning)
tests:
	@for t in tests/test_*.c; do \
	  b=/tmp/$$(basename $$t .c); \
	  $(CC) $(CFLAGS) -o $$b $$t && $$b | grep -iE 'PASS|FAIL' | tail -1 | sed "s#^#$$(basename $$t): #"; \
	done

gpu: $(GPUBIN)
$(GPUBIN): $(GPUSRC)
	$(NVCC) -ccbin $(NVCCBIN) $(NVFLAGS) -o $(GPUBIN) $(GPUSRC)

run-gpu: $(GPUBIN)
	./$(GPUBIN)

clean:
	rm -f $(BIN) $(AGENTBIN) tuner_plug $(PHONEBIN) $(SWIRLBIN) $(CONTESTBIN) $(TOURNEYBIN) pathviz runner $(GPUBIN)
	rm -rf results
