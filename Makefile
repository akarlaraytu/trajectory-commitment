# TLoT — Experiment Pipeline
# Run experiments in order. Each depends on the previous.

.PHONY: setup e00 e01 e02 all clean

MODEL ?= pythia-1.4b
DEVICE ?= auto

setup:
	pip install -r requirements.txt

# Experiment 0: Does Φ exist?
# Output: results/phi_direction_*.npy
e00:
	cd experiments/e00_find_phi && python find_phi.py \
		--model $(MODEL) \
		--backend transformerlens \
		--n-samples 200 \
		--device $(DEVICE)

# Experiment 1: Does π work?
# Requires: e00 output (phi direction .npy file)
# Output: results/e01_results_*.json
e01:
	$(eval PHI_PATH := $(shell ls experiments/e00_find_phi/results/phi_direction_*.npy 2>/dev/null | head -1))
	$(eval PHI_LAYER := $(shell python -c "import json; d=json.load(open('experiments/e00_find_phi/results/e00_results_$(MODEL).json')); print(d['best_layer'])" 2>/dev/null || echo 16))
	cd experiments/e01_test_projection && python test_projection.py \
		--model $(MODEL) \
		--phi-path $(PHI_PATH) \
		--phi-layer $(PHI_LAYER) \
		--n-problems 100 \
		--device $(DEVICE)

# Experiment 2: Can we build ψ?
# Output: results/e02_results_*.json
e02:
	cd experiments/e02_probe_psi && python build_probe.py \
		--model $(MODEL) \
		--n-traces 300 \
		--device $(DEVICE)

# Run all in sequence
all: e00 e01 e02

clean:
	rm -rf experiments/*/results/
