"""
E07b: Activation Patching — Causal Test for Trajectory Commitment

BUILDS ON E07 bifurcation results:
  - 27 bifurcating prompts found (same prompt → both correct and hallucinated)
  - KL divergence onset at step 1 (commitment point)
  - Now: can we CAUSE trajectory change by patching activations?

DESIGN:
  A) LAYER SWEEP: step=1 fixed, patch each layer L0-L27
     → Which layers are causally relevant?

  B) STEP SWEEP: best layer fixed, patch step 0/1/2/3
     → Is step 1 really the critical window?

  C) SYMMETRIC TEST:
     - Hall → Correct patch: does wrong run flip to correct?
     - Correct → Hall patch: does correct run flip to wrong?
     → If both work: step-1 state IS the branching cause

  D) CONTROLS:
     - Random clean patch: activation from DIFFERENT prompt's correct run
     - Wrong-to-wrong patch: activation from different hall run of SAME prompt
     → Rules out brute-force override

METRICS:
  1. Flip rate: wrong→correct (or correct→wrong)
  2. Abstain rate: output becomes uncertain/hedging
  3. Preservation: unchanged outputs

Usage:
    python trajectory_patching.py [model_name]
"""

import os
import sys
import json
import time
import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


def get_prompts():
    sys.path.insert(0, str(Path(__file__).parent.parent / "e05_find_hallucination"))
    from multitoken_hallucination import get_multitoken_prompts
    return get_multitoken_prompts()


def classify_output(generated_text, ground_truth, wrong_indicators):
    text_lower = generated_text.lower()
    for gt in ground_truth:
        if gt.lower() in text_lower:
            return "CORRECT"
    for wrong in wrong_indicators:
        if wrong.lower() in text_lower:
            return "HALLUCINATION"
    return "OTHER"


def generate_with_cache(model, prompt, temperature=0.7, max_new_tokens=15):
    """Generate one sample with full hidden state cache at every (layer, step)."""
    tokenizer = model.tokenizer
    input_ids = model.to_tokens(prompt)
    n_layers = model.cfg.n_layers

    generated_ids = []
    all_states = []       # [step] → [n_layers, d_model]
    all_final_probs = []
    all_tokens = []
    current_ids = input_ids.clone()

    for step in range(max_new_tokens):
        with torch.no_grad():
            logits, cache = model.run_with_cache(current_ids)

        final_logits = logits[0, -1, :].float().cpu()
        final_probs = torch.softmax(final_logits, dim=-1)
        all_final_probs.append(final_probs)

        layer_states = []
        for l in range(n_layers):
            h = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].float().cpu()
            layer_states.append(h)
        all_states.append(torch.stack(layer_states))

        tempered_probs = torch.softmax(final_logits / temperature, dim=-1)
        next_id = torch.multinomial(tempered_probs, 1).item()
        next_token = tokenizer.decode([next_id])
        next_prob = final_probs[next_id].item()
        all_tokens.append((next_token, next_prob, next_id))
        generated_ids.append(next_id)

        next_tensor = torch.tensor([[next_id]], device=current_ids.device)
        current_ids = torch.cat([current_ids, next_tensor], dim=1)

        if next_id == tokenizer.eos_token_id:
            break
        if len(generated_ids) >= 3:
            recent = tokenizer.decode(generated_ids[-3:])
            if recent.count('\n') >= 2:
                break

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return {
        "text": tokenizer.decode(generated_ids),
        "tokens": all_tokens,
        "states": all_states,           # [step][layer] = d_model tensor
        "final_probs": all_final_probs,
        "generated_ids": generated_ids,
        "n_steps": len(all_tokens),
    }


def generate_with_patch(model, prompt, source_run, patch_layer, patch_step,
                        temperature=0.7, max_new_tokens=15):
    """
    Generate tokens, but at (patch_layer, patch_step), replace the residual
    stream activation with the one from source_run.

    Returns the same format as generate_with_cache but without caching
    (for speed — we only need the output text).
    """
    tokenizer = model.tokenizer
    input_ids = model.to_tokens(prompt)
    device = input_ids.device
    n_layers = model.cfg.n_layers

    # Get the patch vector (move to device)
    patch_vector = source_run["states"][patch_step][patch_layer].to(device)

    generated_ids = []
    current_ids = input_ids.clone()

    for step in range(max_new_tokens):
        if step == patch_step:
            # This is the step where we patch
            def patch_hook(value, hook):
                # value shape: [batch, seq_len, d_model]
                # Replace only the last token position
                value[0, -1, :] = patch_vector
                return value

            hook_name = f"blocks.{patch_layer}.hook_resid_post"
            with torch.no_grad():
                logits = model.run_with_hooks(
                    current_ids,
                    fwd_hooks=[(hook_name, patch_hook)]
                )
        else:
            with torch.no_grad():
                logits = model(current_ids)

        final_logits = logits[0, -1, :].float().cpu()
        tempered_probs = torch.softmax(final_logits / temperature, dim=-1)
        next_id = torch.multinomial(tempered_probs, 1).item()
        generated_ids.append(next_id)

        next_tensor = torch.tensor([[next_id]], device=device)
        current_ids = torch.cat([current_ids, next_tensor], dim=1)

        if next_id == tokenizer.eos_token_id:
            break
        if len(generated_ids) >= 3:
            recent = tokenizer.decode(generated_ids[-3:])
            if recent.count('\n') >= 2:
                break

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "text": tokenizer.decode(generated_ids),
        "generated_ids": generated_ids,
    }


def generate_with_window_patch(model, prompt, source_run, patch_layer,
                                patch_steps, temperature=0.7, max_new_tokens=15):
    """
    Like generate_with_patch but patches a WINDOW of steps (e.g., steps 1-3).
    """
    tokenizer = model.tokenizer
    input_ids = model.to_tokens(prompt)
    device = input_ids.device

    generated_ids = []
    current_ids = input_ids.clone()

    for step in range(max_new_tokens):
        if step in patch_steps and step < source_run["n_steps"]:
            patch_vector = source_run["states"][step][patch_layer].to(device)

            def make_hook(pv):
                def patch_hook(value, hook):
                    value[0, -1, :] = pv
                    return value
                return patch_hook

            hook_name = f"blocks.{patch_layer}.hook_resid_post"
            with torch.no_grad():
                logits = model.run_with_hooks(
                    current_ids,
                    fwd_hooks=[(hook_name, make_hook(patch_vector))]
                )
        else:
            with torch.no_grad():
                logits = model(current_ids)

        final_logits = logits[0, -1, :].float().cpu()
        tempered_probs = torch.softmax(final_logits / temperature, dim=-1)
        next_id = torch.multinomial(tempered_probs, 1).item()
        generated_ids.append(next_id)

        next_tensor = torch.tensor([[next_id]], device=device)
        current_ids = torch.cat([current_ids, next_tensor], dim=1)

        if next_id == tokenizer.eos_token_id:
            break
        if len(generated_ids) >= 3:
            recent = tokenizer.decode(generated_ids[-3:])
            if recent.count('\n') >= 2:
                break

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "text": tokenizer.decode(generated_ids),
        "generated_ids": generated_ids,
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
    N_RUNS_PER_CLASS = 3     # cached runs per class for source activations
    N_PATCH_TRIALS = 3       # repeat each patch to average over sampling noise
    TEMPERATURE = 0.7
    MAX_GEN = 15

    print(f"{'='*70}")
    print(f"E07b: ACTIVATION PATCHING — CAUSAL TEST")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Runs per class: {N_RUNS_PER_CLASS}")
    print(f"Patch trials: {N_PATCH_TRIALS}")
    print(f"Temperature: {TEMPERATURE}\n")

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"

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

    # Load bifurcating prompts from E07
    e07_path = Path(__file__).parent / "results" / f"bifurcation_{model_name.replace('/', '_')}.json"
    with open(e07_path) as f:
        e07_data = json.load(f)

    bifurcating_indices = [p["idx"] for p in e07_data["bifurcating_prompts"]]
    prompts = get_prompts()

    # Select best bifurcating prompts (highest min(C,H) rate)
    bif_sorted = sorted(e07_data["bifurcating_prompts"],
                        key=lambda p: min(p["counts"]["CORRECT"], p["counts"]["HALLUCINATION"]),
                        reverse=True)

    # Take top 8 most balanced
    selected = bif_sorted[:8]
    print(f"Selected {len(selected)} most balanced bifurcating prompts:")
    for p in selected:
        print(f"  [{p['idx']:2d}] [{p['category']:15s}] "
              f"C={p['counts']['CORRECT']:2d} H={p['counts']['HALLUCINATION']:2d}  "
              f"| {p['prompt'][:55]}")

    # ═══════════════════════════════════════════════════════════════
    # COLLECT SOURCE RUNS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"PHASE 1: Collecting source runs")
    print(f"{'='*70}\n")

    prompt_data = {}  # idx → {correct_runs, hall_runs, prompt_info}

    for sp in selected:
        idx = sp["idx"]
        prompt, ground_truth, wrong_indicators, category, notes = prompts[idx]
        print(f"  Prompt [{idx}]: {prompt[:55]}...")

        correct_runs = []
        hall_runs = []
        attempts = 0
        max_attempts = 100

        while (len(correct_runs) < N_RUNS_PER_CLASS or
               len(hall_runs) < N_RUNS_PER_CLASS) and attempts < max_attempts:
            attempts += 1
            data = generate_with_cache(model, prompt, temperature=TEMPERATURE,
                                       max_new_tokens=MAX_GEN)
            cls = classify_output(data["text"], ground_truth, wrong_indicators)

            if cls == "CORRECT" and len(correct_runs) < N_RUNS_PER_CLASS:
                correct_runs.append(data)
                print(f"    ✓ correct  [{len(correct_runs)}/{N_RUNS_PER_CLASS}]  → {data['text'][:40]}")
            elif cls == "HALLUCINATION" and len(hall_runs) < N_RUNS_PER_CLASS:
                hall_runs.append(data)
                print(f"    ✗ halluc   [{len(hall_runs)}/{N_RUNS_PER_CLASS}]  → {data['text'][:40]}")

        if len(correct_runs) < 2 or len(hall_runs) < 2:
            print(f"    ⚠ Not enough runs, skipping")
            continue

        prompt_data[idx] = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "wrong_indicators": wrong_indicators,
            "category": category,
            "correct_runs": correct_runs,
            "hall_runs": hall_runs,
        }
        print(f"    Collected: {len(correct_runs)} correct, {len(hall_runs)} hall\n")
        sys.stdout.flush()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"  Total prompts with sufficient runs: {len(prompt_data)}\n")

    if not prompt_data:
        print("No prompts with sufficient runs. Exiting.")
        return

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT A: LAYER SWEEP (step=1 fixed)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"EXPERIMENT A: LAYER SWEEP — patch step=1, vary layer")
    print(f"{'='*70}\n")

    layer_sweep_results = {}  # layer → {h2c_flips, c2h_flips, h2c_total, c2h_total, ...}

    for layer in range(n_layers):
        h2c_flips = 0   # hall→correct
        c2h_flips = 0   # correct→hall
        h2c_total = 0
        c2h_total = 0
        h2c_abstain = 0
        c2h_abstain = 0

        for idx, pd in prompt_data.items():
            prompt = pd["prompt"]
            gt = pd["ground_truth"]
            wrong = pd["wrong_indicators"]

            # Hall → Correct patch: take hall run, patch with correct activation
            h_run = pd["hall_runs"][0]
            c_run = pd["correct_runs"][0]
            if 1 < min(h_run["n_steps"], c_run["n_steps"]):
                for trial in range(N_PATCH_TRIALS):
                    result = generate_with_patch(
                        model, prompt, source_run=c_run,
                        patch_layer=layer, patch_step=1,
                        temperature=TEMPERATURE, max_new_tokens=MAX_GEN)
                    cls = classify_output(result["text"], gt, wrong)
                    h2c_total += 1
                    if cls == "CORRECT":
                        h2c_flips += 1
                    elif cls == "OTHER":
                        h2c_abstain += 1

            # Correct → Hall patch: take correct run, patch with hall activation
            if 1 < min(h_run["n_steps"], c_run["n_steps"]):
                for trial in range(N_PATCH_TRIALS):
                    result = generate_with_patch(
                        model, prompt, source_run=h_run,
                        patch_layer=layer, patch_step=1,
                        temperature=TEMPERATURE, max_new_tokens=MAX_GEN)
                    cls = classify_output(result["text"], gt, wrong)
                    c2h_total += 1
                    if cls == "HALLUCINATION":
                        c2h_flips += 1
                    elif cls == "OTHER":
                        c2h_abstain += 1

        h2c_rate = h2c_flips / max(h2c_total, 1)
        c2h_rate = c2h_flips / max(c2h_total, 1)
        h2c_abs_rate = h2c_abstain / max(h2c_total, 1)
        c2h_abs_rate = c2h_abstain / max(c2h_total, 1)

        layer_sweep_results[layer] = {
            "h2c_flips": h2c_flips, "h2c_total": h2c_total,
            "h2c_rate": h2c_rate, "h2c_abstain_rate": h2c_abs_rate,
            "c2h_flips": c2h_flips, "c2h_total": c2h_total,
            "c2h_rate": c2h_rate, "c2h_abstain_rate": c2h_abs_rate,
        }

        bar_h2c = "█" * int(h2c_rate * 30)
        bar_c2h = "█" * int(c2h_rate * 30)
        print(f"  L{layer:2d}  H→C: {h2c_rate:5.1%} ({h2c_flips:3d}/{h2c_total:3d}) {bar_h2c}")
        print(f"        C→H: {c2h_rate:5.1%} ({c2h_flips:3d}/{c2h_total:3d}) {bar_c2h}")
        sys.stdout.flush()

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Find best layer
    best_h2c_layer = max(layer_sweep_results, key=lambda l: layer_sweep_results[l]["h2c_rate"])
    best_c2h_layer = max(layer_sweep_results, key=lambda l: layer_sweep_results[l]["c2h_rate"])

    print(f"\n  Best H→C layer: L{best_h2c_layer} ({layer_sweep_results[best_h2c_layer]['h2c_rate']:.1%})")
    print(f"  Best C→H layer: L{best_c2h_layer} ({layer_sweep_results[best_c2h_layer]['c2h_rate']:.1%})")

    # Use the layer with highest combined effect
    best_layer = max(layer_sweep_results,
                     key=lambda l: layer_sweep_results[l]["h2c_rate"] + layer_sweep_results[l]["c2h_rate"])
    print(f"  Best combined layer: L{best_layer}")

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT B: STEP SWEEP (best layer fixed)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"EXPERIMENT B: STEP SWEEP — layer=L{best_layer} fixed, vary step")
    print(f"{'='*70}\n")

    step_sweep_results = {}

    for patch_step in range(5):  # steps 0-4
        h2c_flips = 0
        c2h_flips = 0
        h2c_total = 0
        c2h_total = 0

        for idx, pd in prompt_data.items():
            prompt = pd["prompt"]
            gt = pd["ground_truth"]
            wrong = pd["wrong_indicators"]
            h_run = pd["hall_runs"][0]
            c_run = pd["correct_runs"][0]

            if patch_step >= min(h_run["n_steps"], c_run["n_steps"]):
                continue

            for trial in range(N_PATCH_TRIALS):
                result = generate_with_patch(
                    model, prompt, source_run=c_run,
                    patch_layer=best_layer, patch_step=patch_step,
                    temperature=TEMPERATURE, max_new_tokens=MAX_GEN)
                cls = classify_output(result["text"], gt, wrong)
                h2c_total += 1
                if cls == "CORRECT":
                    h2c_flips += 1

            for trial in range(N_PATCH_TRIALS):
                result = generate_with_patch(
                    model, prompt, source_run=h_run,
                    patch_layer=best_layer, patch_step=patch_step,
                    temperature=TEMPERATURE, max_new_tokens=MAX_GEN)
                cls = classify_output(result["text"], gt, wrong)
                c2h_total += 1
                if cls == "HALLUCINATION":
                    c2h_flips += 1

        h2c_rate = h2c_flips / max(h2c_total, 1)
        c2h_rate = c2h_flips / max(c2h_total, 1)
        step_sweep_results[patch_step] = {
            "h2c_rate": h2c_rate, "h2c_flips": h2c_flips, "h2c_total": h2c_total,
            "c2h_rate": c2h_rate, "c2h_flips": c2h_flips, "c2h_total": c2h_total,
        }

        print(f"  Step {patch_step}  H→C: {h2c_rate:5.1%} ({h2c_flips}/{h2c_total})  "
              f"C→H: {c2h_rate:5.1%} ({c2h_flips}/{c2h_total})")
        sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT C: WINDOW PATCHING (best layer, steps 1 to 1+k)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"EXPERIMENT C: WINDOW PATCHING — layer=L{best_layer}")
    print(f"{'='*70}\n")

    window_results = {}

    for window_size in [1, 2, 3, 4]:
        patch_steps = list(range(1, 1 + window_size))  # [1], [1,2], [1,2,3], [1,2,3,4]
        h2c_flips = 0
        c2h_flips = 0
        h2c_total = 0
        c2h_total = 0

        for idx, pd in prompt_data.items():
            prompt = pd["prompt"]
            gt = pd["ground_truth"]
            wrong = pd["wrong_indicators"]
            h_run = pd["hall_runs"][0]
            c_run = pd["correct_runs"][0]
            max_step = max(patch_steps)

            if max_step >= min(h_run["n_steps"], c_run["n_steps"]):
                continue

            for trial in range(N_PATCH_TRIALS):
                result = generate_with_window_patch(
                    model, prompt, source_run=c_run,
                    patch_layer=best_layer, patch_steps=patch_steps,
                    temperature=TEMPERATURE, max_new_tokens=MAX_GEN)
                cls = classify_output(result["text"], gt, wrong)
                h2c_total += 1
                if cls == "CORRECT":
                    h2c_flips += 1

            for trial in range(N_PATCH_TRIALS):
                result = generate_with_window_patch(
                    model, prompt, source_run=h_run,
                    patch_layer=best_layer, patch_steps=patch_steps,
                    temperature=TEMPERATURE, max_new_tokens=MAX_GEN)
                cls = classify_output(result["text"], gt, wrong)
                c2h_total += 1
                if cls == "HALLUCINATION":
                    c2h_flips += 1

        h2c_rate = h2c_flips / max(h2c_total, 1)
        c2h_rate = c2h_flips / max(c2h_total, 1)
        window_results[window_size] = {
            "steps": patch_steps,
            "h2c_rate": h2c_rate, "h2c_flips": h2c_flips, "h2c_total": h2c_total,
            "c2h_rate": c2h_rate, "c2h_flips": c2h_flips, "c2h_total": c2h_total,
        }

        print(f"  Window [1..{1+window_size-1}]  H→C: {h2c_rate:5.1%} ({h2c_flips}/{h2c_total})  "
              f"C→H: {c2h_rate:5.1%} ({c2h_flips}/{c2h_total})")
        sys.stdout.flush()

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT D: CONTROLS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"EXPERIMENT D: CONTROLS — layer=L{best_layer}, step=1")
    print(f"{'='*70}\n")

    control_results = {"random_clean": {}, "wrong_to_wrong": {}, "baseline_nopatch": {}}

    # D1: Random clean patch — activation from DIFFERENT prompt's correct run
    print(f"  D1: RANDOM CLEAN PATCH (different prompt's correct activation)")
    random_flips = 0
    random_total = 0
    prompt_indices = list(prompt_data.keys())

    for idx, pd in prompt_data.items():
        prompt = pd["prompt"]
        gt = pd["ground_truth"]
        wrong = pd["wrong_indicators"]

        # Pick a different prompt's correct run as source
        other_indices = [j for j in prompt_indices if j != idx]
        if not other_indices:
            continue
        other_idx = other_indices[0]
        other_correct = prompt_data[other_idx]["correct_runs"][0]

        for h_run in pd["hall_runs"][:2]:
            if 1 >= min(h_run["n_steps"], other_correct["n_steps"]):
                continue
            for trial in range(N_PATCH_TRIALS):
                result = generate_with_patch(
                    model, prompt, source_run=other_correct,
                    patch_layer=best_layer, patch_step=1,
                    temperature=TEMPERATURE, max_new_tokens=MAX_GEN)
                cls = classify_output(result["text"], gt, wrong)
                random_total += 1
                if cls == "CORRECT":
                    random_flips += 1

    random_rate = random_flips / max(random_total, 1)
    control_results["random_clean"] = {
        "flip_rate": random_rate, "flips": random_flips, "total": random_total
    }
    print(f"      H→C flip rate: {random_rate:.1%} ({random_flips}/{random_total})")

    # D2: Wrong-to-wrong patch — different hall run of SAME prompt
    print(f"\n  D2: WRONG-TO-WRONG PATCH (different hall run, same prompt)")
    w2w_flips = 0
    w2w_total = 0

    for idx, pd in prompt_data.items():
        prompt = pd["prompt"]
        gt = pd["ground_truth"]
        wrong = pd["wrong_indicators"]

        if len(pd["hall_runs"]) < 2:
            continue

        h_run_target = pd["hall_runs"][0]
        h_run_source = pd["hall_runs"][1]

        if 1 >= min(h_run_target["n_steps"], h_run_source["n_steps"]):
            continue

        for trial in range(N_PATCH_TRIALS):
            result = generate_with_patch(
                model, prompt, source_run=h_run_source,
                patch_layer=best_layer, patch_step=1,
                temperature=TEMPERATURE, max_new_tokens=MAX_GEN)
            cls = classify_output(result["text"], gt, wrong)
            w2w_total += 1
            if cls == "CORRECT":
                w2w_flips += 1

    w2w_rate = w2w_flips / max(w2w_total, 1)
    control_results["wrong_to_wrong"] = {
        "flip_rate": w2w_rate, "flips": w2w_flips, "total": w2w_total
    }
    print(f"      H→C flip rate: {w2w_rate:.1%} ({w2w_flips}/{w2w_total})")

    # D3: Baseline — no patch, just normal temperature sampling
    print(f"\n  D3: BASELINE (no patch, just resample)")
    base_flips = 0
    base_total = 0

    for idx, pd in prompt_data.items():
        prompt = pd["prompt"]
        gt = pd["ground_truth"]
        wrong = pd["wrong_indicators"]

        for trial in range(N_PATCH_TRIALS * 2):
            from transformer_lens import HookedTransformer
            tokenizer = model.tokenizer
            input_ids = model.to_tokens(prompt)
            generated_ids = []
            current_ids = input_ids.clone()

            for step in range(MAX_GEN):
                with torch.no_grad():
                    logits = model(current_ids)
                final_logits = logits[0, -1, :].float().cpu()
                tempered_probs = torch.softmax(final_logits / TEMPERATURE, dim=-1)
                next_id = torch.multinomial(tempered_probs, 1).item()
                generated_ids.append(next_id)
                next_tensor = torch.tensor([[next_id]], device=current_ids.device)
                current_ids = torch.cat([current_ids, next_tensor], dim=1)
                if next_id == tokenizer.eos_token_id:
                    break

            text = tokenizer.decode(generated_ids)
            cls = classify_output(text, gt, wrong)
            base_total += 1
            if cls == "CORRECT":
                base_flips += 1

    base_rate = base_flips / max(base_total, 1)
    control_results["baseline_nopatch"] = {
        "correct_rate": base_rate, "correct": base_flips, "total": base_total
    }
    print(f"      Baseline correct rate: {base_rate:.1%} ({base_flips}/{base_total})")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    best_h2c = layer_sweep_results[best_h2c_layer]["h2c_rate"]
    best_c2h = layer_sweep_results[best_c2h_layer]["c2h_rate"]

    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  LAYER SWEEP (step=1)                                       │")
    print(f"  │    Best H→C:  L{best_h2c_layer:2d}  {best_h2c:5.1%}                              │")
    print(f"  │    Best C→H:  L{best_c2h_layer:2d}  {best_c2h:5.1%}                              │")
    print(f"  │                                                              │")
    print(f"  │  STEP SWEEP (L{best_layer})                                       │")
    for s, sr in step_sweep_results.items():
        print(f"  │    Step {s}: H→C {sr['h2c_rate']:5.1%}  C→H {sr['c2h_rate']:5.1%}                     │")
    print(f"  │                                                              │")
    print(f"  │  WINDOW PATCHING (L{best_layer})                                  │")
    for w, wr in window_results.items():
        print(f"  │    [1..{w}]: H→C {wr['h2c_rate']:5.1%}  C→H {wr['c2h_rate']:5.1%}                     │")
    print(f"  │                                                              │")
    print(f"  │  CONTROLS                                                    │")
    print(f"  │    Random clean patch:  {random_rate:5.1%}                           │")
    print(f"  │    Wrong-to-wrong:      {w2w_rate:5.1%}                           │")
    print(f"  │    Baseline (no patch): {base_rate:5.1%}                           │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    # Interpret
    print(f"\n  INTERPRETATION:")
    if best_h2c > 0.3 and best_c2h > 0.2:
        if random_rate < best_h2c * 0.5:
            print(f"  ✓ CAUSAL CONTROL CONFIRMED")
            print(f"    → Patching correct activation into hall run FLIPS output ({best_h2c:.0%})")
            print(f"    → Patching hall activation into correct run CORRUPTS output ({best_c2h:.0%})")
            print(f"    → Random patch does NOT flip ({random_rate:.0%}) → not brute-force")
            print(f"    → Step-1 L{best_layer} activation IS the branching cause")
        else:
            print(f"  ⚠ PATCH WORKS BUT SO DOES RANDOM")
            print(f"    → Patching disrupts the model, not specifically corrects it")
    elif best_h2c > 0.1:
        print(f"  ~ PARTIAL EFFECT")
        print(f"    → Some causal influence but not dominant")
        print(f"    → Real cause may be distributed across layers/steps")
    else:
        print(f"  ✗ NO CAUSAL EFFECT")
        print(f"    → Step-1 activation is a readout, not a cause")
        print(f"    → The trajectory may be determined by something else")
        print(f"      (attention patterns, token embeddings, earlier processing)")

    print(f"\n{'='*70}")
    print("Done.")

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    def to_serializable(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_serializable(v) for v in obj]
        return obj

    save_data = to_serializable({
        "model": model_name,
        "n_prompts": len(prompt_data),
        "best_layer": best_layer,
        "best_h2c_layer": best_h2c_layer,
        "best_c2h_layer": best_c2h_layer,
        "layer_sweep": layer_sweep_results,
        "step_sweep": step_sweep_results,
        "window_results": window_results,
        "controls": control_results,
    })

    out_dir = Path(__file__).parent / "results"
    out_path = out_dir / f"patching_{model_name.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    # ═══════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════
    fig_dir = Path(__file__).parent / "results" / "figures"

    # Plot A: Layer sweep
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    layers = sorted(layer_sweep_results.keys())
    h2c_rates = [layer_sweep_results[l]["h2c_rate"] for l in layers]
    c2h_rates = [layer_sweep_results[l]["c2h_rate"] for l in layers]

    ax1.bar(layers, h2c_rates, color='green', alpha=0.7, label='H→C flip rate')
    ax1.axhline(y=random_rate, color='gray', linestyle='--', alpha=0.5, label=f'Random ctrl ({random_rate:.0%})')
    ax1.axhline(y=base_rate, color='blue', linestyle=':', alpha=0.5, label=f'Baseline ({base_rate:.0%})')
    ax1.set_ylabel('Flip Rate')
    ax1.set_title('Hall → Correct (patching correct activation into wrong run)')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    ax2.bar(layers, c2h_rates, color='red', alpha=0.7, label='C→H flip rate')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Flip Rate')
    ax2.set_title('Correct → Hall (patching wrong activation into correct run)')
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.suptitle(f'Activation Patching Layer Sweep (step=1, {len(prompt_data)} prompts)', fontsize=13)
    plt.tight_layout()
    plt.savefig(fig_dir / "05_patching_layer_sweep.png", dpi=150)
    plt.close()

    # Plot B: Step sweep
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    steps_list = sorted(step_sweep_results.keys())
    h2c_s = [step_sweep_results[s]["h2c_rate"] for s in steps_list]
    c2h_s = [step_sweep_results[s]["c2h_rate"] for s in steps_list]

    ax.plot(steps_list, h2c_s, 'g-o', linewidth=2, markersize=8, label='H→C')
    ax.plot(steps_list, c2h_s, 'r-o', linewidth=2, markersize=8, label='C→H')
    ax.axhline(y=base_rate, color='blue', linestyle=':', alpha=0.5, label=f'Baseline ({base_rate:.0%})')
    ax.set_xlabel('Patch Step')
    ax.set_ylabel('Flip Rate')
    ax.set_title(f'Step Sweep at L{best_layer}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "06_patching_step_sweep.png", dpi=150)
    plt.close()

    # Plot C: Window
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    wins = sorted(window_results.keys())
    h2c_w = [window_results[w]["h2c_rate"] for w in wins]
    c2h_w = [window_results[w]["c2h_rate"] for w in wins]

    ax.plot(wins, h2c_w, 'g-s', linewidth=2, markersize=8, label='H→C')
    ax.plot(wins, c2h_w, 'r-s', linewidth=2, markersize=8, label='C→H')
    ax.set_xlabel('Window Size (steps 1..N)')
    ax.set_ylabel('Flip Rate')
    ax.set_title(f'Window Patching at L{best_layer}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "07_patching_window.png", dpi=150)
    plt.close()

    print(f"  Plots saved to {fig_dir}/")


if __name__ == "__main__":
    main()
