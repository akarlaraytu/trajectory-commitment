"""
E06: Multi-Token TLoT — Φ Computation & Intervention

PREVIOUS FINDINGS:
  - E01-E04: Single-token Φ exists geometrically (Cohen's d ~2.0)
    but projection has ZERO behavioral effect — model compensates
  - E05: 32 genuine multi-token hallucinations found
    (19 confabulation, 7 false premise, 6 factual)
  - Commitment point: avg step 4.3, avg prob 0.460

THIS EXPERIMENT:
  Phase 1: COLLECT hidden states during generation
    - For each hallucinating case: cache h_l at every (layer, step)
    - For each correct case: same
    - Extract states at commitment point and pre-commitment

  Phase 2: COMPUTE multi-token Φ
    - Φ_l = mean(h_halluc[layer,step]) - mean(h_correct[layer,step])
    - Compare: commitment-point Φ vs pre-commitment Φ vs step-0 Φ
    - Cohen's d per layer → where is the separation?

  Phase 3: INTERVENE during generation
    - Hook into generation loop
    - At each step, project out Φ_l from hidden states
    - Compare: does hallucination flip to correct?
    - Ablations: which layers, which steps, how much projection

Usage:
    python multitoken_phi.py [model_name]
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_e05_data():
    """Load E05b results — hallucination and correct cases."""
    e05_path = Path(__file__).parent.parent / "e05_find_hallucination" / "results" / "multitoken_hallucination_Qwen_Qwen2.5-1.5B.json"
    with open(e05_path) as f:
        data = json.load(f)
    return data


def get_prompts_from_e05():
    """
    Re-import prompts from e05 module to get ground_truth and wrong_indicators.
    Returns dict: idx → (prompt, ground_truth, wrong_indicators, category, notes)
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "e05_find_hallucination"))
    from multitoken_hallucination import get_multitoken_prompts
    prompts = get_multitoken_prompts()
    return {i: p for i, p in enumerate(prompts)}


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: COLLECT HIDDEN STATES DURING GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_with_cache(model, prompt, max_new_tokens=30):
    """
    Generate tokens while caching hidden states at every layer.
    Returns: generated_text, per_step_states, per_step_tokens
      per_step_states[step] = tensor of shape [n_layers, d_model]
        (residual stream at last position after each layer)
    """
    tokenizer = model.tokenizer
    input_ids = model.to_tokens(prompt)
    n_layers = model.cfg.n_layers

    generated_ids = []
    per_step_states = []  # list of [n_layers, d_model]
    per_step_tokens = []  # list of (token_str, prob)
    current_ids = input_ids.clone()

    for step in range(max_new_tokens):
        # Run with cache
        with torch.no_grad():
            logits, cache = model.run_with_cache(current_ids)

        # Extract residual stream at last position for each layer
        layer_states = []
        for l in range(n_layers):
            # resid_post = output of layer l
            h = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].float().cpu()
            layer_states.append(h)

        per_step_states.append(torch.stack(layer_states))  # [n_layers, d_model]

        # Greedy decode
        next_logits = logits[0, -1, :].float().cpu()
        probs = torch.softmax(next_logits, dim=-1)
        next_id = probs.argmax().item()
        next_token = tokenizer.decode([next_id])
        next_prob = probs[next_id].item()

        per_step_tokens.append((next_token, next_prob))
        generated_ids.append(next_id)

        # Append to context
        next_tensor = torch.tensor([[next_id]], device=current_ids.device)
        current_ids = torch.cat([current_ids, next_tensor], dim=1)

        # Stop conditions
        if next_id == tokenizer.eos_token_id:
            break
        if len(generated_ids) >= 3:
            recent = tokenizer.decode(generated_ids[-3:])
            if recent.count('\n') >= 2:
                break

        # Clean cache
        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    generated_text = tokenizer.decode(generated_ids)
    return generated_text, per_step_states, per_step_tokens


def generate_with_intervention(model, prompt, phi_per_layer, intervention_config,
                                max_new_tokens=30):
    """
    Generate tokens with Φ projection at specified layers and steps.

    intervention_config: dict with:
      - layers: list of layer indices to intervene
      - start_step: start intervening from this step
      - end_step: stop intervening after this step
      - alpha: projection strength (1.0 = full projection)
      - method: "project" (remove Φ direction) or "steer" (add anti-Φ)
    """
    tokenizer = model.tokenizer
    input_ids = model.to_tokens(prompt)

    layers = intervention_config["layers"]
    start_step = intervention_config.get("start_step", 0)
    end_step = intervention_config.get("end_step", max_new_tokens)
    alpha = intervention_config.get("alpha", 1.0)
    method = intervention_config.get("method", "project")

    generated_ids = []
    per_step_tokens = []
    current_ids = input_ids.clone()

    for step in range(max_new_tokens):
        # Set up hooks for this step
        hooks = []
        if start_step <= step <= end_step:
            for l in layers:
                phi = phi_per_layer[l].to(current_ids.device)
                phi_norm = phi / (phi.norm() + 1e-8)

                if method == "project":
                    def hook_fn(value, hook, phi_n=phi_norm, a=alpha):
                        # Project out Φ from last position
                        h = value[0, -1, :]
                        proj = torch.dot(h.float(), phi_n.float()) * phi_n.to(h.dtype)
                        value[0, -1, :] = h - a * proj.to(h.dtype)
                        return value
                elif method == "steer":
                    def hook_fn(value, hook, phi_n=phi_norm, a=alpha):
                        # Steer AWAY from Φ direction
                        h = value[0, -1, :]
                        dot = torch.dot(h.float(), phi_n.float())
                        if dot > 0:  # only if heading toward hallucination
                            value[0, -1, :] = h - a * dot * phi_n.to(h.dtype)
                        return value

                hooks.append((f"blocks.{l}.hook_resid_post", hook_fn))

        # Run with hooks
        with torch.no_grad():
            if hooks:
                logits = model.run_with_hooks(current_ids, fwd_hooks=hooks)
            else:
                logits = model(current_ids)

        # Greedy decode
        next_logits = logits[0, -1, :].float().cpu()
        probs = torch.softmax(next_logits, dim=-1)
        next_id = probs.argmax().item()
        next_token = tokenizer.decode([next_id])
        next_prob = probs[next_id].item()

        per_step_tokens.append((next_token, next_prob))
        generated_ids.append(next_id)

        next_tensor = torch.tensor([[next_id]], device=current_ids.device)
        current_ids = torch.cat([current_ids, next_tensor], dim=1)

        if next_id == tokenizer.eos_token_id:
            break
        if len(generated_ids) >= 3:
            recent = tokenizer.decode(generated_ids[-3:])
            if recent.count('\n') >= 2:
                break

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    generated_text = tokenizer.decode(generated_ids)
    return generated_text, per_step_tokens


# ═══════════════════════════════════════════════════════════════════
# CLASSIFICATION (reuse from E05)
# ═══════════════════════════════════════════════════════════════════

def classify_output(generated_text, ground_truth, wrong_indicators):
    text_lower = generated_text.lower()
    refusal_phrases = ["i don't know", "i'm not sure", "i cannot", "not certain",
                       "no information", "unable to"]
    for phrase in refusal_phrases:
        if phrase in text_lower:
            return "REFUSAL"
    for gt in ground_truth:
        if gt.lower() in text_lower:
            return "CORRECT"
    for wrong in wrong_indicators:
        if wrong.lower() in text_lower:
            return "HALLUCINATION"
    return "IRRELEVANT"


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"

    print(f"{'='*70}")
    print(f"E06: MULTI-TOKEN TLoT — Φ & INTERVENTION")
    print(f"{'='*70}")
    print(f"Model: {model_name}\n")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    print(f"Loading {model_name}...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"Loaded in {time.time()-t0:.0f}s")
    print(f"  n_layers={n_layers}, d_model={d_model}\n")

    # Load E05 data
    e05_data = load_e05_data()
    prompt_db = get_prompts_from_e05()

    # Separate hallucination vs correct
    hall_indices = [s["idx"] for s in e05_data["hallucination_prompts"]]
    correct_indices = [s["idx"] for s in e05_data["all_samples"]
                       if s["classification"] == "CORRECT"]

    print(f"Hallucination cases: {len(hall_indices)}")
    print(f"Correct cases: {len(correct_indices)}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: COLLECT HIDDEN STATES
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 1: Collecting hidden states during generation")
    print(f"{'='*70}\n")

    # We'll collect states at step 0 (first generated token) for all cases
    # This is the most comparable point across cases
    hall_states_step0 = []   # list of [n_layers, d_model]
    hall_states_step2 = []   # step 2 states
    correct_states_step0 = []
    correct_states_step2 = []

    # Also collect at commitment point for hallucination cases
    hall_states_commit = []
    commit_steps = []

    # Process hallucination cases
    print("  Hallucination cases:")
    for i, idx in enumerate(hall_indices):
        prompt_info = prompt_db[idx]
        prompt = prompt_info[0]
        print(f"    [{i+1}/{len(hall_indices)}] {prompt[:50]}...", end="", flush=True)

        t1 = time.time()
        gen_text, states, tokens = generate_with_cache(model, prompt, max_new_tokens=15)
        dt = time.time() - t1

        # Step 0 state
        hall_states_step0.append(states[0])  # [n_layers, d_model]

        # Step 2 state (if available)
        if len(states) > 2:
            hall_states_step2.append(states[2])

        # Commitment point from E05 data
        e05_entry = next((s for s in e05_data["hallucination_prompts"] if s["idx"] == idx), None)
        if e05_entry and e05_entry.get("commitment_point"):
            cp_step = e05_entry["commitment_point"]["step"]
            if cp_step < len(states):
                hall_states_commit.append(states[cp_step])
                commit_steps.append(cp_step)
            else:
                # Commitment point beyond our generation length, use last
                hall_states_commit.append(states[-1])
                commit_steps.append(len(states) - 1)
        else:
            hall_states_commit.append(states[0])
            commit_steps.append(0)

        gen_short = gen_text[:40].replace('\n', ' ')
        print(f"  ({dt:.1f}s) → {gen_short}")

    # Process correct cases
    print("\n  Correct cases:")
    for i, idx in enumerate(correct_indices):
        prompt_info = prompt_db[idx]
        prompt = prompt_info[0]
        print(f"    [{i+1}/{len(correct_indices)}] {prompt[:50]}...", end="", flush=True)

        t1 = time.time()
        gen_text, states, tokens = generate_with_cache(model, prompt, max_new_tokens=15)
        dt = time.time() - t1

        correct_states_step0.append(states[0])
        if len(states) > 2:
            correct_states_step2.append(states[2])

        gen_short = gen_text[:40].replace('\n', ' ')
        print(f"  ({dt:.1f}s) → {gen_short}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: COMPUTE Φ
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 2: Computing multi-token Φ")
    print(f"{'='*70}\n")

    # Stack: [n_samples, n_layers, d_model]
    H_hall_s0 = torch.stack(hall_states_step0)
    H_corr_s0 = torch.stack(correct_states_step0)

    # Φ at step 0: mean difference per layer
    phi_step0 = []      # [n_layers, d_model]
    cohens_d_s0 = []

    print("  Layer-wise Φ at step 0:")
    print(f"  {'Layer':>5s}  {'||Φ||':>8s}  {'Cohen d':>8s}  {'cos(Φ,mean_h)':>13s}")

    for l in range(n_layers):
        h_hall = H_hall_s0[:, l, :]   # [n_hall, d_model]
        h_corr = H_corr_s0[:, l, :]   # [n_corr, d_model]

        mean_hall = h_hall.mean(dim=0)
        mean_corr = h_corr.mean(dim=0)

        phi_l = mean_hall - mean_corr
        phi_step0.append(phi_l)

        # Cohen's d: ||mean_diff|| / pooled_std
        pooled_std = torch.sqrt(
            (h_hall.var(dim=0) * (len(h_hall)-1) + h_corr.var(dim=0) * (len(h_corr)-1))
            / (len(h_hall) + len(h_corr) - 2)
        ).mean()

        d = phi_l.norm() / (pooled_std + 1e-8)
        cohens_d_s0.append(d.item())

        # Cosine with mean hidden state
        mean_all = torch.cat([h_hall, h_corr]).mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(phi_l.unsqueeze(0), mean_all.unsqueeze(0)).item()

        if l % 4 == 0 or l == n_layers - 1:
            print(f"  {l:5d}  {phi_l.norm().item():8.2f}  {d.item():8.2f}  {cos:13.4f}")

    phi_step0 = torch.stack(phi_step0)  # [n_layers, d_model]

    # Φ at commitment point
    if hall_states_commit:
        H_hall_commit = torch.stack(hall_states_commit)
        phi_commit = []
        cohens_d_commit = []

        print(f"\n  Layer-wise Φ at commitment point (avg step {np.mean(commit_steps):.1f}):")
        print(f"  {'Layer':>5s}  {'||Φ||':>8s}  {'Cohen d':>8s}")

        for l in range(n_layers):
            h_hall = H_hall_commit[:, l, :]
            h_corr = H_corr_s0[:, l, :]  # Compare with correct step 0

            mean_hall = h_hall.mean(dim=0)
            mean_corr = h_corr.mean(dim=0)

            phi_l = mean_hall - mean_corr
            phi_commit.append(phi_l)

            pooled_std = torch.sqrt(
                (h_hall.var(dim=0) * (len(h_hall)-1) + h_corr.var(dim=0) * (len(h_corr)-1))
                / (len(h_hall) + len(h_corr) - 2)
            ).mean()

            d = phi_l.norm() / (pooled_std + 1e-8)
            cohens_d_commit.append(d.item())

            if l % 4 == 0 or l == n_layers - 1:
                print(f"  {l:5d}  {phi_l.norm().item():8.2f}  {d.item():8.2f}")

        phi_commit = torch.stack(phi_commit)

    # Φ at step 2
    if hall_states_step2 and correct_states_step2:
        H_hall_s2 = torch.stack(hall_states_step2)
        H_corr_s2 = torch.stack(correct_states_step2)

        phi_step2 = []
        cohens_d_s2 = []

        print(f"\n  Layer-wise Φ at step 2:")
        print(f"  {'Layer':>5s}  {'||Φ||':>8s}  {'Cohen d':>8s}")

        for l in range(n_layers):
            h_hall = H_hall_s2[:, l, :]
            h_corr = H_corr_s2[:, l, :]

            phi_l = h_hall.mean(dim=0) - h_corr.mean(dim=0)
            phi_step2.append(phi_l)

            pooled_std = torch.sqrt(
                (h_hall.var(dim=0) * (len(h_hall)-1) + h_corr.var(dim=0) * (len(h_corr)-1))
                / (len(h_hall) + len(h_corr) - 2)
            ).mean()

            d = phi_l.norm() / (pooled_std + 1e-8)
            cohens_d_s2.append(d.item())

            if l % 4 == 0 or l == n_layers - 1:
                print(f"  {l:5d}  {phi_l.norm().item():8.2f}  {d.item():8.2f}")

        phi_step2 = torch.stack(phi_step2)

    # Find best layers (top Cohen's d)
    best_layers_s0 = sorted(range(n_layers), key=lambda l: -cohens_d_s0[l])[:8]
    print(f"\n  Top 8 layers by Cohen's d (step 0): {best_layers_s0}")
    print(f"    d values: {[f'{cohens_d_s0[l]:.2f}' for l in best_layers_s0]}")

    if hall_states_commit:
        best_layers_commit = sorted(range(n_layers), key=lambda l: -cohens_d_commit[l])[:8]
        print(f"  Top 8 layers by Cohen's d (commit): {best_layers_commit}")
        print(f"    d values: {[f'{cohens_d_commit[l]:.2f}' for l in best_layers_commit]}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2.5: CROSS-STEP Φ CONSISTENCY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  Cross-step Φ consistency (cosine similarity):")
    if hall_states_commit and hall_states_step2:
        for l in best_layers_s0[:4]:
            cos_s0_s2 = torch.nn.functional.cosine_similarity(
                phi_step0[l].unsqueeze(0), phi_step2[l].unsqueeze(0)).item()
            cos_s0_commit = torch.nn.functional.cosine_similarity(
                phi_step0[l].unsqueeze(0), phi_commit[l].unsqueeze(0)).item()
            print(f"    L{l}: cos(Φ_s0, Φ_s2)={cos_s0_s2:.3f}  cos(Φ_s0, Φ_commit)={cos_s0_commit:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: INTERVENTION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3: Multi-token intervention")
    print(f"{'='*70}\n")

    # Choose which Φ to use for intervention
    # Use step 0 Φ as default — it's the earliest and most general
    phi_for_intervention = phi_step0

    # Define intervention conditions
    top4_layers = best_layers_s0[:4]
    top8_layers = best_layers_s0[:8]
    mid_layers = list(range(10, 18))
    late_layers = list(range(20, 28))
    all_layers = list(range(n_layers))

    conditions = [
        {"name": "baseline", "layers": [], "alpha": 0.0, "method": "project",
         "start_step": 0, "end_step": 30},

        {"name": "top4_project_a1", "layers": top4_layers, "alpha": 1.0,
         "method": "project", "start_step": 0, "end_step": 30},

        {"name": "top4_project_a2", "layers": top4_layers, "alpha": 2.0,
         "method": "project", "start_step": 0, "end_step": 30},

        {"name": "top4_project_a5", "layers": top4_layers, "alpha": 5.0,
         "method": "project", "start_step": 0, "end_step": 30},

        {"name": "top8_project_a1", "layers": top8_layers, "alpha": 1.0,
         "method": "project", "start_step": 0, "end_step": 30},

        {"name": "top4_steer_a1", "layers": top4_layers, "alpha": 1.0,
         "method": "steer", "start_step": 0, "end_step": 30},

        {"name": "top4_steer_a3", "layers": top4_layers, "alpha": 3.0,
         "method": "steer", "start_step": 0, "end_step": 30},

        {"name": "mid_project_a1", "layers": mid_layers, "alpha": 1.0,
         "method": "project", "start_step": 0, "end_step": 30},

        {"name": "late_project_a1", "layers": late_layers, "alpha": 1.0,
         "method": "project", "start_step": 0, "end_step": 30},

        {"name": "all_project_a1", "layers": all_layers, "alpha": 1.0,
         "method": "project", "start_step": 0, "end_step": 30},

        {"name": "top4_early_only", "layers": top4_layers, "alpha": 2.0,
         "method": "project", "start_step": 0, "end_step": 3},

        {"name": "top4_late_only", "layers": top4_layers, "alpha": 2.0,
         "method": "project", "start_step": 4, "end_step": 30},
    ]

    # If commit Φ exists, also test it
    if hall_states_commit:
        conditions.append(
            {"name": "top4_commitPhi_a1", "layers": top4_layers, "alpha": 1.0,
             "method": "project", "start_step": 0, "end_step": 30,
             "use_commit_phi": True}
        )

    results = {}

    for cond in conditions:
        cname = cond["name"]
        print(f"\n  Condition: {cname}")
        print(f"    layers={cond['layers'][:6]}{'...' if len(cond['layers'])>6 else ''} "
              f"α={cond['alpha']} method={cond['method']} "
              f"steps={cond['start_step']}-{cond['end_step']}")

        # Choose Φ
        if cond.get("use_commit_phi"):
            phi_use = phi_commit
        else:
            phi_use = phi_for_intervention

        hall_fixed = 0
        hall_broken_to_irrelevant = 0
        hall_stayed = 0
        correct_maintained = 0
        correct_broken = 0

        all_outputs = []

        # Test on hallucination cases
        for idx in hall_indices:
            prompt_info = prompt_db[idx]
            prompt, ground_truth, wrong_indicators = prompt_info[0], prompt_info[1], prompt_info[2]

            gen_text, tokens = generate_with_intervention(
                model, prompt, phi_use, cond, max_new_tokens=30)
            cls = classify_output(gen_text, ground_truth, wrong_indicators)

            if cls == "CORRECT":
                hall_fixed += 1
            elif cls == "HALLUCINATION":
                hall_stayed += 1
            else:
                hall_broken_to_irrelevant += 1

            all_outputs.append({
                "idx": idx, "type": "hallucination", "prompt": prompt[:60],
                "output": gen_text[:100], "class": cls
            })

        # Test on correct cases
        for idx in correct_indices:
            prompt_info = prompt_db[idx]
            prompt, ground_truth, wrong_indicators = prompt_info[0], prompt_info[1], prompt_info[2]

            gen_text, tokens = generate_with_intervention(
                model, prompt, phi_use, cond, max_new_tokens=30)
            cls = classify_output(gen_text, ground_truth, wrong_indicators)

            if cls == "CORRECT":
                correct_maintained += 1
            else:
                correct_broken += 1

            all_outputs.append({
                "idx": idx, "type": "correct", "prompt": prompt[:60],
                "output": gen_text[:100], "class": cls
            })

        n_hall = len(hall_indices)
        n_corr = len(correct_indices)
        fix_rate = hall_fixed / n_hall if n_hall > 0 else 0
        break_rate = correct_broken / n_corr if n_corr > 0 else 0
        net_rate = fix_rate - break_rate

        results[cname] = {
            "config": {k: v for k, v in cond.items() if k != "layers" or len(v) <= 10},
            "n_layers_used": len(cond["layers"]),
            "hall_fixed": hall_fixed,
            "hall_stayed": hall_stayed,
            "hall_irrelevant": hall_broken_to_irrelevant,
            "correct_maintained": correct_maintained,
            "correct_broken": correct_broken,
            "fix_rate": fix_rate,
            "break_rate": break_rate,
            "net_rate": net_rate,
            "outputs": all_outputs,
        }

        print(f"    HALL: fixed={hall_fixed}/{n_hall} ({fix_rate:.1%}), "
              f"stayed={hall_stayed}, irrelevant={hall_broken_to_irrelevant}")
        print(f"    CORR: maintained={correct_maintained}/{n_corr} ({correct_maintained/n_corr:.1%}), "
              f"broken={correct_broken}")
        print(f"    NET: {net_rate:+.1%} (fix_rate - break_rate)")

    # ═══════════════════════════════════════════════════════════════
    # RESULTS SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}\n")

    print(f"  {'Condition':<25s} {'Fixed':>6s} {'Stayed':>7s} {'Irrelev':>8s} "
          f"{'Maint':>6s} {'Broken':>7s} {'Net':>7s}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*7} {'-'*7}")

    best_net = -999
    best_cond = None

    for cname, r in results.items():
        print(f"  {cname:<25s} {r['hall_fixed']:>6d} {r['hall_stayed']:>7d} "
              f"{r['hall_irrelevant']:>8d} {r['correct_maintained']:>6d} "
              f"{r['correct_broken']:>7d} {r['net_rate']:>+6.1%}")
        if r["net_rate"] > best_net:
            best_net = r["net_rate"]
            best_cond = cname

    print(f"\n  Best condition: {best_cond} (net={best_net:+.1%})")

    # Show flipped examples for best condition
    if best_cond and results[best_cond]["hall_fixed"] > 0:
        print(f"\n  Fixed examples ({best_cond}):")
        for out in results[best_cond]["outputs"]:
            if out["type"] == "hallucination" and out["class"] == "CORRECT":
                print(f"    [{out['idx']:3d}] {out['prompt']}")
                print(f"         → {out['output'][:80]}")

    # Show broken examples for best condition
    if best_cond and results[best_cond]["correct_broken"] > 0:
        print(f"\n  Broken examples ({best_cond}):")
        for out in results[best_cond]["outputs"]:
            if out["type"] == "correct" and out["class"] != "CORRECT":
                print(f"    [{out['idx']:3d}] {out['prompt']}")
                print(f"         → {out['output'][:80]}")

    # ═══════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════

    fig_dir = Path(__file__).parent / "results" / "figures"

    # Plot 1: Cohen's d by layer
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(range(n_layers), cohens_d_s0, 'b-o', markersize=4, label='Step 0 Φ')
    if hall_states_commit:
        ax.plot(range(n_layers), cohens_d_commit, 'r-s', markersize=4, label='Commitment Φ')
    if hall_states_step2:
        ax.plot(range(n_layers), cohens_d_s2, 'g-^', markersize=4, label='Step 2 Φ')
    ax.set_xlabel('Layer')
    ax.set_ylabel("Cohen's d")
    ax.set_title('Multi-Token Φ: Hallucination vs Correct Separation by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='d=0.8 (large)')
    plt.tight_layout()
    plt.savefig(fig_dir / "01_cohens_d_by_layer.png", dpi=150)
    plt.close()

    # Plot 2: Intervention results
    cond_names = list(results.keys())
    fix_rates = [results[c]["fix_rate"] for c in cond_names]
    break_rates = [results[c]["break_rate"] for c in cond_names]
    net_rates = [results[c]["net_rate"] for c in cond_names]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    x = np.arange(len(cond_names))
    width = 0.25
    ax.bar(x - width, fix_rates, width, label='Fix rate', color='green', alpha=0.7)
    ax.bar(x, break_rates, width, label='Break rate', color='red', alpha=0.7)
    ax.bar(x + width, net_rates, width, label='Net rate', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Rate')
    ax.set_title('Multi-Token TLoT Intervention Results')
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / "02_intervention_results.png", dpi=150)
    plt.close()

    # Plot 3: Cross-step Φ consistency
    if hall_states_commit and hall_states_step2:
        cos_matrix = np.zeros((3, n_layers))
        for l in range(n_layers):
            cos_matrix[0, l] = torch.nn.functional.cosine_similarity(
                phi_step0[l].unsqueeze(0), phi_step2[l].unsqueeze(0)).item()
            cos_matrix[1, l] = torch.nn.functional.cosine_similarity(
                phi_step0[l].unsqueeze(0), phi_commit[l].unsqueeze(0)).item()
            cos_matrix[2, l] = torch.nn.functional.cosine_similarity(
                phi_step2[l].unsqueeze(0), phi_commit[l].unsqueeze(0)).item()

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(range(n_layers), cos_matrix[0], 'b-o', markersize=3, label='cos(Φ_s0, Φ_s2)')
        ax.plot(range(n_layers), cos_matrix[1], 'r-s', markersize=3, label='cos(Φ_s0, Φ_commit)')
        ax.plot(range(n_layers), cos_matrix[2], 'g-^', markersize=3, label='cos(Φ_s2, Φ_commit)')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cross-Step Φ Consistency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.1)
        plt.tight_layout()
        plt.savefig(fig_dir / "03_phi_consistency.png", dpi=150)
        plt.close()

    print(f"\n  Figures saved to {fig_dir}/")

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    save_data = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_hallucination": len(hall_indices),
        "n_correct": len(correct_indices),
        "cohens_d_step0": cohens_d_s0,
        "cohens_d_commit": cohens_d_commit if hall_states_commit else None,
        "cohens_d_step2": cohens_d_s2 if hall_states_step2 else None,
        "best_layers_step0": best_layers_s0,
        "intervention_results": {
            cname: {k: v for k, v in r.items() if k != "outputs"}
            for cname, r in results.items()
        },
        "intervention_outputs": {
            cname: r["outputs"] for cname, r in results.items()
        },
        "best_condition": best_cond,
        "best_net_rate": best_net,
    }

    out_dir = Path(__file__).parent / "results"
    out_path = out_dir / f"multitoken_tlot_{model_name.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {out_path}")

    # Save Φ tensors for future use
    phi_path = out_dir / f"phi_tensors_{model_name.replace('/', '_')}.pt"
    phi_save = {"phi_step0": phi_step0}
    if hall_states_commit:
        phi_save["phi_commit"] = phi_commit
    if hall_states_step2:
        phi_save["phi_step2"] = phi_step2
    torch.save(phi_save, phi_path)
    print(f"  Φ tensors saved to {phi_path}")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
