# TLoT Experiment Log
## Model: Qwen/Qwen2.5-1.5B (28 layers, d_model=1536)

---

## E00: Find Phi (Hallucination Direction)
**Status:** Complete
**Question:** Does a linear direction Phi exist in residual stream that separates hallucinating from correct states?
**Method:** PCA/LDA on residual stream activations from hallucinating vs correct completions.
**Results:**
- Phi exists with d=15-22 (cosine similarity between PCA and LDA directions)
- Best layers: mid-to-upper (L12-L23)
- Cross-validated accuracy ~75-85%
**Verdict:** Phi direction exists as a statistical separator, but this is correlational, not causal.

---

## E01: Test Projection (Phi as Control)
**Status:** Complete
**Question:** Can we project OUT Phi from the residual stream to prevent hallucination?
**Method:** Linear projection: h' = h - (h . phi) * phi, applied during generation.
**Results:**
- Projection does NOT prevent hallucination
- Model compensates: removing Phi from one layer, model rebuilds it in later layers
- Replication with Qwen2.5-1.5B confirms: projection accuracy ~80% but no causal effect
**Verdict:** Phi is a READOUT direction, not a causal direction. Model's computation is distributed, not localized in one linear subspace.

---

## E02: Causal Tracing
**Status:** Complete
**Question:** Which layers/positions are causally important for hallucination?
**Method:** Activation patching (corrupt-then-restore) at individual layers and positions.
**Results:**
- Causal effect distributed across many layers
- No single "hallucination layer" found
- Mid layers (L10-L18) show slightly more effect
**Verdict:** Hallucination is a distributed phenomenon, not localized.

---

## E03: Trajectory Control (Phi-guided Steering)
**Status:** Complete (negative result)
**Question:** Can we steer generation by adding/subtracting Phi during decoding?
**Results:**
- Steering shifts logits but model output doesn't reliably change
- Confirms E01: linear intervention on Phi doesn't work
**Verdict:** Linear steering is insufficient. Need nonlinear or multi-point intervention.

---

## E04: Logit Intervention
**Status:** Complete
**Question:** Direct logit modification as alternative to activation steering.
**Results:**
- Logit bias can force correct tokens but is trivial/circular
- Doesn't reveal mechanism, just overrides output
**Verdict:** Not informative for TLoT theory.

---

## E05: Find Hallucination (Systematic)
**Status:** Complete
**Question:** Build a proper dataset of genuine hallucinations for this model.
**Method:** 61 prompts across categories: factual, false_premise, confabulation, leading, multi_hop, math.
**Results:**
- 32 genuine hallucinations found
- 20 correct control cases
- Categories: confabulation most reliable for hallucination, false_premise mixed
**Verdict:** Clean dataset for downstream experiments.

---

## E06: Multi-token TLoT
**Status:** Complete
**Question:** Does Phi signal evolve during multi-token generation?
**Method:** Track Phi projection across generation steps.
**Results:**
- Phi projection grows during hallucinating generation
- But this is still correlational
- Phi tensor saved for reference
**Verdict:** Phi tracks hallucination state across generation but doesn't prove causation.

---

## E07: Trajectory Analysis (REVISED) -- Temperature Bifurcation
**Status:** Complete
**Question:** Do hallucinating and correct trajectories genuinely diverge, controlling for prompt effects?

### E07a: Bifurcation Discovery
**Method:** Same prompt, 20 samples at T=0.7. Find prompts where model produces BOTH correct and wrong outputs.
**Key Design Decision:** Previous E07 compared different prompts (hall vs correct) which trivially gives "ALWAYS SEPARATED" -- a methodological artifact. This version uses same-prompt bifurcation to isolate trajectory effects from prompt effects.

**Results:**
```
Total prompts:         61
Bifurcating:           27  (same prompt -> both correct and wrong)
Always hallucinating:  13
Always correct:         4
Near-bifurcating:       6
```

**Category breakdown:**
| Category | Bifurcating | Deterministic Hall | Pattern |
|----------|------------|-------------------|---------|
| factual | 4/14 | ~5 | Mostly deterministic |
| false_premise | 13/14 | 1 | Almost all bifurcating |
| confabulation | 8/22 | 9 | Mixed |
| leading | 1/3 | 0 | Mostly correct |
| multi_hop | 0/4 | 0 | Always correct |
| math | 2/4 | 0 | Mostly correct |

**KL Divergence:**
- Step 0: KL = 0.00 for ALL bifurcating prompts (same input -> same logits, as expected)
- Step 1: KL jumps to 1-19 (immediate divergence after first different token)
- Mean onset: step 1.1
- This confirms: prompt encoding is identical, divergence is purely trajectory-driven

**Heatmaps (Cohen's d):**
- Step 0: black (d=0) across all layers -- identical states
- Step 1+: immediate separation, growing monotonically
- Upper layers (L20-L27) separate slightly before lower layers
- By step 5+: massive separation (d > 100)

**Verdict:** Bifurcation is REAL. Same prompt, same initial state, different outcomes. Trajectory commitment happens at step 1 and is immediate. No gradual drift -- it's a sharp fork.

### E07b: Activation Patching (Causal Test)
**Method:** For bifurcating prompts, collect correct and hallucinated runs with full hidden state caches. Then:
- H->C patch: Replace hall run's activation with correct run's activation
- C->H patch: Replace correct run's activation with hall run's activation
- Controls: random clean patch, wrong-to-wrong patch, baseline

**Results -- Layer Sweep (step=1, all 28 layers):**
```
Best H->C (fix hallucination): L24 = 33.3%
Best C->H (induce hallucination): L20 = 87.5%
Random clean control: 12.5% (= baseline noise)
Wrong-to-wrong control: 12.5% (= baseline noise)
Baseline (no patch): 10.4%
```

**Critical finding: ASYMMETRY**
- Corrupting a correct run is EASY: single-layer patch at L20 -> 87.5% corruption
- Fixing a hallucinated run is HARD: best single-layer patch at L24 -> 33.3% fix
- But 33.3% >> 10.4% baseline, and >> 12.5% random control -> effect is REAL and SPECIFIC

**Step Sweep (L20):**
```
Step 0: H->C 20.8%  C->H 66.7%
Step 1: H->C 29.2%  C->H 62.5%  <- peak fix rate
Step 2: H->C 25.0%  C->H 58.3%
Step 3: H->C 12.5%  C->H 66.7%
Step 4: H->C 20.8%  C->H 66.7%
```
- Commitment window: steps 0-2, not a single "decision point"
- C->H corruption works at ANY step (~60-67%) -- correct trajectory is always vulnerable

**Window Patching (L20):**
```
Window [1]:   H->C 12.5%  C->H 58.3%
Window [1-2]: H->C 12.5%  C->H 45.8%
Window [1-3]: H->C 20.8%  C->H 66.7%
Window [1-4]: H->C 33.3%  C->H 75.0%
```
- Fixing requires sustained multi-step intervention (window 4 = 33.3%)
- Single-step patch insufficient for correction (12.5% = noise)
- Corruption only needs one-shot

**Verdict:**
1. CAUSAL CONTROL CONFIRMED: Activation patching at L20-L24 causally affects hallucination trajectory
2. The effect is SPECIFIC: random/wrong-to-wrong patches don't work
3. ASYMMETRY is the key finding:
   - Hallucination acts as an ATTRACTOR BASIN -- easy to fall in, hard to escape
   - Correct trajectory is FRAGILE -- single perturbation can corrupt it
   - Fixing requires multi-step, multi-layer sustained intervention
4. This explains why E01/E03 linear projection failed: single-direction single-layer intervention can't escape the attractor

---

## Cumulative Theory Implications

### What we've established:
1. **Phi exists** (E00) but is NOT causal (E01, E03)
2. **Hallucination has genuine trajectory dynamics** (E07a): same prompt -> different outcomes
3. **Trajectories commit immediately** at step 1, not gradually (E07a)
4. **Commitment is causally verifiable** via activation patching (E07b)
5. **Hallucination is an attractor basin**: easy to enter, hard to exit (E07b asymmetry)
6. **Correction requires sustained multi-point intervention** (E07b window patching)
7. **Different hallucination types have different dynamics** (E07a category analysis):
   - False premise: highly stochastic (model is uncertain)
   - Confabulation: mostly deterministic (model confidently wrong)
   - Factual: mixed

### What this means for TLoT:
- Trajectory-based theory of hallucination is supported by causal evidence
- But control is HARDER than simple linear intervention suggests
- The "physics not punishment" framing maps well: hallucination follows dynamical attractor structure
- Intervention must be multi-point and sustained, not single-shot

### Open questions for next experiments:
- Can multi-layer simultaneous patching (L20 + L24) improve H->C rate beyond 33%?
- Is the attractor structure visible in the loss landscape?
- Does the asymmetry hold for larger models?
- Can we find a NONLINEAR correction that escapes the attractor in one step?

---

## Data Files
```
e00: results/e00_results_pythia-1b.json, phi_all_layers_*.npz
e01: results/e01_results_pythia-410m.json, replication_200_Qwen_*.json
e02: results/causal_tracing_Qwen_*.json
e05: results/multitoken_hallucination_Qwen_*.json
e06: results/multitoken_tlot_Qwen_*.json
e07: results/bifurcation_Qwen_*.json, patching_Qwen_*.json
Figures: e07/results/figures/01-07_*.png
```
