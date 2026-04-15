"""
E07 (REVISED): Temperature Bifurcation — True Trajectory Analysis

PROBLEM WITH PREVIOUS E07:
  Compared different prompts (hall vs correct) → trivially different states.
  "ALWAYS SEPARATED" was a methodological artifact, not a finding.

THIS EXPERIMENT:
  Same prompt, multiple samples, sometimes correct sometimes wrong.
  THIS is the true test: same input → different trajectories.

DESIGN:
  Phase 1: DISCOVERY
    - Take all 61 prompts from E05
    - Run each 40 times with temperature=0.7
    - Classify each run as CORRECT / HALLUCINATION / OTHER
    - Find BIFURCATING prompts: ones that produce BOTH correct and wrong

  Phase 2: TRAJECTORY ANALYSIS (only on bifurcating prompts)
    - For each bifurcating prompt:
      - Collect hidden states from correct runs
      - Collect hidden states from hallucinated runs
      - Same prompt, same initial encoding, different outcomes
    - KL divergence: at which (layer, step) do they first diverge?
    - PCA: trajectory shape — bifurcation or gradual drift?

  Phase 3: FALSE-PREMISE CONTROL
    - Within false_premise category: 7 hall vs 5 correct
    - These share prompt structure → secondary validation

Usage:
    python trajectory_bifurcation.py [model_name]
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
from sklearn.decomposition import PCA

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


def generate_sample(model, prompt, temperature=0.7, max_new_tokens=30):
    """Generate one sample with temperature sampling. No cache (fast)."""
    tokenizer = model.tokenizer
    input_ids = model.to_tokens(prompt)
    generated_ids = []
    current_ids = input_ids.clone()

    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = model(current_ids)
        next_logits = logits[0, -1, :].float().cpu()
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
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
    return tokenizer.decode(generated_ids)


def generate_with_cache(model, prompt, temperature=0.7, max_new_tokens=20):
    """Generate one sample with full hidden state cache."""
    tokenizer = model.tokenizer
    input_ids = model.to_tokens(prompt)
    n_layers = model.cfg.n_layers

    generated_ids = []
    all_states = []       # [step] → [n_layers, d_model]
    all_final_probs = []  # [step] → [d_vocab]
    all_tokens = []
    current_ids = input_ids.clone()

    for step in range(max_new_tokens):
        with torch.no_grad():
            logits, cache = model.run_with_cache(current_ids)

        final_logits = logits[0, -1, :].float().cpu()
        # Store UNTEMPERED probs for KL comparison (temperature only affects sampling)
        final_probs = torch.softmax(final_logits, dim=-1)
        all_final_probs.append(final_probs)

        # Per-layer states
        layer_states = []
        for l in range(n_layers):
            h = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].float().cpu()
            layer_states.append(h)
        all_states.append(torch.stack(layer_states))

        # Sample with temperature
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
        "states": all_states,
        "final_probs": all_final_probs,
        "n_steps": len(all_tokens),
    }


def compute_kl(p, q, eps=1e-10):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum().item()


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
    N_SAMPLES = 20       # samples per prompt for discovery (reduced for MPS stability)
    N_CACHED = 6         # cached runs per class for trajectory analysis
    MAX_GEN = 30         # max tokens for discovery
    MAX_GEN_CACHED = 15  # max tokens for cached runs
    TEMPERATURE = 0.7

    print(f"{'='*70}")
    print(f"E07 (REVISED): TEMPERATURE BIFURCATION")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Samples per prompt: {N_SAMPLES}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Cached runs per class: {N_CACHED}\n")

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

    prompts = get_prompts()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: DISCOVERY — find bifurcating prompts
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*70}")
    print(f"PHASE 1: Discovery — sampling {N_SAMPLES}x each prompt")
    print(f"{'='*70}\n")

    prompt_results = []

    for i, (prompt, ground_truth, wrong_indicators, category, notes) in enumerate(prompts):
        counts = {"CORRECT": 0, "HALLUCINATION": 0, "OTHER": 0}

        for s in range(N_SAMPLES):
            gen_text = generate_sample(model, prompt, temperature=TEMPERATURE,
                                       max_new_tokens=MAX_GEN)
            cls = classify_output(gen_text, ground_truth, wrong_indicators)
            counts[cls] += 1

        total = N_SAMPLES
        c_rate = counts["CORRECT"] / total
        h_rate = counts["HALLUCINATION"] / total
        is_bifurcating = counts["CORRECT"] >= 2 and counts["HALLUCINATION"] >= 2

        prompt_results.append({
            "idx": i,
            "prompt": prompt,
            "category": category,
            "notes": notes,
            "ground_truth": ground_truth,
            "wrong_indicators": wrong_indicators,
            "counts": counts,
            "correct_rate": c_rate,
            "hallucination_rate": h_rate,
            "is_bifurcating": is_bifurcating,
        })

        tag = "★ BIFURCATING" if is_bifurcating else ""
        print(f"  [{i+1:2d}/{len(prompts)}] [{category:15s}] "
              f"C={counts['CORRECT']:3d} H={counts['HALLUCINATION']:3d} "
              f"O={counts['OTHER']:3d}  ({c_rate:.0%}/{h_rate:.0%})  {tag}")
        sys.stdout.flush()
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Summary
    bifurcating = [p for p in prompt_results if p["is_bifurcating"]]
    always_correct = [p for p in prompt_results if p["correct_rate"] > 0.9]
    always_hall = [p for p in prompt_results if p["hallucination_rate"] > 0.9]

    print(f"\n  Summary:")
    print(f"    Total prompts:     {len(prompts)}")
    print(f"    Bifurcating:       {len(bifurcating)}  ← GOLD")
    print(f"    Always correct:    {len(always_correct)}")
    print(f"    Always halluc:     {len(always_hall)}")
    print(f"    Mixed/other:       {len(prompts) - len(bifurcating) - len(always_correct) - len(always_hall)}")

    if bifurcating:
        print(f"\n  Bifurcating prompts:")
        for p in bifurcating:
            print(f"    [{p['idx']:2d}] [{p['category']:15s}] "
                  f"C={p['counts']['CORRECT']:2d} H={p['counts']['HALLUCINATION']:2d}  "
                  f"| {p['prompt'][:60]}")

    if not bifurcating:
        print(f"\n  ⚠ NO BIFURCATING PROMPTS FOUND!")
        print(f"  This means: model's hallucination is deterministic, not stochastic.")
        print(f"  Temperature sampling only affects surface — the trajectory is fixed.")
        print(f"\n  This is itself a major finding:")
        print(f"  → Hallucination is decided at prompt encoding, NOT during generation.")

        # Still save results and do what we can
        # Fall back to within-category analysis

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: TRAJECTORY ANALYSIS on bifurcating prompts
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"PHASE 2: Trajectory Analysis")
    print(f"{'='*70}\n")

    trajectory_data = {}

    if bifurcating:
        print(f"  Collecting cached runs for {len(bifurcating)} bifurcating prompts...\n")

        for bp in bifurcating:
            prompt = bp["prompt"]
            gt = bp["ground_truth"]
            wrong = bp["wrong_indicators"]
            idx = bp["idx"]

            print(f"  Prompt [{idx}]: {prompt[:55]}...")

            correct_runs = []
            hall_runs = []
            attempts = 0
            max_attempts = N_SAMPLES * 3  # safety limit

            while (len(correct_runs) < N_CACHED or len(hall_runs) < N_CACHED) and attempts < max_attempts:
                attempts += 1
                data = generate_with_cache(model, prompt, temperature=TEMPERATURE,
                                           max_new_tokens=MAX_GEN_CACHED)
                cls = classify_output(data["text"], gt, wrong)

                if cls == "CORRECT" and len(correct_runs) < N_CACHED:
                    correct_runs.append(data)
                    print(f"    ✓ correct  [{len(correct_runs)}/{N_CACHED}]  → {data['text'][:40]}")
                elif cls == "HALLUCINATION" and len(hall_runs) < N_CACHED:
                    hall_runs.append(data)
                    print(f"    ✗ halluc   [{len(hall_runs)}/{N_CACHED}]  → {data['text'][:40]}")

            if not correct_runs or not hall_runs:
                print(f"    ⚠ Could not collect enough runs, skipping")
                continue

            # Compute trajectory metrics
            n_steps = min(
                min(d["n_steps"] for d in correct_runs),
                min(d["n_steps"] for d in hall_runs),
                MAX_GEN_CACHED
            )

            # KL divergence per (layer, step)
            kl_per_step = []
            kl_matrix = np.zeros((n_layers, n_steps))

            for step in range(n_steps):
                # Mean probs
                corr_probs = torch.stack([d["final_probs"][step] for d in correct_runs]).mean(dim=0)
                hall_probs = torch.stack([d["final_probs"][step] for d in hall_runs]).mean(dim=0)
                kl = compute_kl(hall_probs, corr_probs)
                kl_per_step.append(kl)

                # Per-layer: compare hidden states projected through logit lens would
                # be ideal but expensive. Instead compare hidden state distances.
                for l in range(n_layers):
                    corr_h = torch.stack([d["states"][step][l] for d in correct_runs])
                    hall_h = torch.stack([d["states"][step][l] for d in hall_runs])
                    # Cohen's d as separation metric
                    mean_diff = (hall_h.mean(dim=0) - corr_h.mean(dim=0)).norm()
                    pooled_std = torch.sqrt(
                        (hall_h.var(dim=0).mean() * (len(hall_h)-1) +
                         corr_h.var(dim=0).mean() * (len(corr_h)-1)) /
                        (len(hall_h) + len(corr_h) - 2)
                    )
                    kl_matrix[l, step] = (mean_diff / (pooled_std + 1e-8)).item()

            # Find divergence onset
            onset = next((s for s in range(n_steps) if kl_per_step[s] > 0.5), None)

            # PCA at key layers
            pca_results = {}
            for layer in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
                all_pts = []
                all_labels = []
                all_steps = []

                for d in correct_runs:
                    for s in range(n_steps):
                        all_pts.append(d["states"][s][layer].numpy())
                        all_labels.append("correct")
                        all_steps.append(s)
                for d in hall_runs:
                    for s in range(n_steps):
                        all_pts.append(d["states"][s][layer].numpy())
                        all_labels.append("hall")
                        all_steps.append(s)

                X = np.array(all_pts)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                # Step-wise separation
                step_dists = []
                for s in range(n_steps):
                    c_pts = X_pca[(np.array(all_labels) == "correct") & (np.array(all_steps) == s)]
                    h_pts = X_pca[(np.array(all_labels) == "hall") & (np.array(all_steps) == s)]
                    if len(c_pts) > 0 and len(h_pts) > 0:
                        step_dists.append(float(np.linalg.norm(h_pts.mean(0) - c_pts.mean(0))))
                    else:
                        step_dists.append(0.0)

                pca_results[layer] = {
                    "X_pca": X_pca,
                    "labels": all_labels,
                    "steps": all_steps,
                    "step_dists": step_dists,
                    "explained_var": pca.explained_variance_ratio_.tolist(),
                }

            trajectory_data[idx] = {
                "prompt": prompt,
                "category": bp["category"],
                "n_correct": len(correct_runs),
                "n_hall": len(hall_runs),
                "n_steps": n_steps,
                "kl_per_step": kl_per_step,
                "kl_matrix": kl_matrix,
                "onset_step": onset,
                "pca_results": pca_results,
            }

            print(f"    KL per step: {' '.join([f'{kl:.2f}' for kl in kl_per_step[:8]])}")
            print(f"    Onset (KL>0.5): {'step '+str(onset) if onset is not None else 'never'}")
            print()

    else:
        print("  No bifurcating prompts — skipping trajectory analysis.\n")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2b: NEAR-BIFURCATION ANALYSIS
    # For prompts that are >80% one class but have some variance
    # ═══════════════════════════════════════════════════════════════

    # Find prompts with at least SOME variance (>=1 correct AND >=1 hall)
    near_bifurcating = [p for p in prompt_results
                        if p["counts"]["CORRECT"] >= 1 and p["counts"]["HALLUCINATION"] >= 1
                        and not p["is_bifurcating"]]

    if near_bifurcating:
        print(f"  Near-bifurcating prompts ({len(near_bifurcating)}):")
        for p in near_bifurcating:
            minority = min(p["counts"]["CORRECT"], p["counts"]["HALLUCINATION"])
            majority_class = "CORRECT" if p["correct_rate"] > p["hallucination_rate"] else "HALL"
            print(f"    [{p['idx']:2d}] [{p['category']:15s}] "
                  f"C={p['counts']['CORRECT']:2d} H={p['counts']['HALLUCINATION']:2d} "
                  f"(mostly {majority_class}, minority={minority})  "
                  f"| {p['prompt'][:55]}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: RESULTS & PLOTS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}\n")

    fig_dir = Path(__file__).parent / "results" / "figures"
    os.makedirs(fig_dir, exist_ok=True)

    if trajectory_data:
        # Plot per-prompt KL curves
        fig, axes = plt.subplots(1, min(len(trajectory_data), 4),
                                  figsize=(5*min(len(trajectory_data), 4), 5))
        if len(trajectory_data) == 1:
            axes = [axes]

        for ax, (idx, td) in zip(axes, list(trajectory_data.items())[:4]):
            ax.plot(range(td["n_steps"]), td["kl_per_step"][:td["n_steps"]],
                    'b-o', linewidth=2, markersize=5)
            if td["onset_step"] is not None:
                ax.axvline(x=td["onset_step"], color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Generation Step')
            ax.set_ylabel('KL(P_hall || P_correct)')
            ax.set_title(f'Prompt [{idx}]\n{td["prompt"][:40]}...', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / "01_bifurcation_kl.png", dpi=150)
        plt.close()

        # Plot PCA trajectories
        for idx, td in trajectory_data.items():
            n_pca_layers = len(td["pca_results"])
            fig, axes = plt.subplots(1, n_pca_layers, figsize=(5*n_pca_layers, 5))
            if n_pca_layers == 1:
                axes = [axes]

            for ax, (layer, pr) in zip(axes, td["pca_results"].items()):
                X = pr["X_pca"]
                labels = np.array(pr["labels"])
                steps = np.array(pr["steps"])
                n_steps = td["n_steps"]

                # Draw individual trajectories
                n_corr = td["n_correct"]
                n_hall = td["n_hall"]

                for run_i in range(n_corr):
                    mask = (labels == "correct") & (steps >= 0)
                    pts_all = X[labels == "correct"]
                    pts = pts_all[run_i*n_steps:(run_i+1)*n_steps]
                    if len(pts) > 1:
                        ax.plot(pts[:, 0], pts[:, 1], '-', color='green', alpha=0.3, linewidth=1)
                        ax.scatter(pts[0, 0], pts[0, 1], c='green', s=20, alpha=0.4, zorder=3)

                for run_i in range(n_hall):
                    pts_all = X[labels == "hall"]
                    pts = pts_all[run_i*n_steps:(run_i+1)*n_steps]
                    if len(pts) > 1:
                        ax.plot(pts[:, 0], pts[:, 1], '-', color='red', alpha=0.3, linewidth=1)
                        ax.scatter(pts[0, 0], pts[0, 1], c='red', s=20, alpha=0.4, zorder=3)

                # Mean trajectories
                corr_mean = []
                hall_mean = []
                for s in range(n_steps):
                    c = X[(labels == "correct") & (steps == s)]
                    h = X[(labels == "hall") & (steps == s)]
                    if len(c) > 0: corr_mean.append(c.mean(0))
                    if len(h) > 0: hall_mean.append(h.mean(0))

                if corr_mean:
                    cm = np.array(corr_mean)
                    ax.plot(cm[:, 0], cm[:, 1], '-o', color='darkgreen', linewidth=3,
                            markersize=6, label='correct mean', zorder=5)
                if hall_mean:
                    hm = np.array(hall_mean)
                    ax.plot(hm[:, 0], hm[:, 1], '-o', color='darkred', linewidth=3,
                            markersize=6, label='hall mean', zorder=5)

                ev = pr["explained_var"]
                ax.set_xlabel(f'PC1 ({ev[0]:.1%})')
                ax.set_ylabel(f'PC2 ({ev[1]:.1%})')
                ax.set_title(f'L{layer}', fontsize=10)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.2)

            plt.suptitle(f'Prompt [{idx}]: {td["prompt"][:50]}...', fontsize=10)
            plt.tight_layout()
            plt.savefig(fig_dir / f"02_trajectory_prompt_{idx}.png", dpi=150)
            plt.close()

        # Plot separation heatmap (layer x step) for each bifurcating prompt
        for idx, td in trajectory_data.items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            im = ax.imshow(td["kl_matrix"][:, :td["n_steps"]], aspect='auto',
                           cmap='hot', interpolation='nearest', origin='lower')
            ax.set_xlabel('Generation Step')
            ax.set_ylabel('Layer')
            ax.set_title(f'Hidden State Separation (Cohen d)\nPrompt [{idx}]: {td["prompt"][:50]}...')
            plt.colorbar(im, ax=ax, label="Cohen's d")
            plt.tight_layout()
            plt.savefig(fig_dir / f"03_separation_heatmap_{idx}.png", dpi=150)
            plt.close()

    # Summary plot: bifurcation rates
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    indices = range(len(prompt_results))
    c_rates = [p["correct_rate"] for p in prompt_results]
    h_rates = [p["hallucination_rate"] for p in prompt_results]
    cats = [p["category"] for p in prompt_results]

    cat_colors = {"confabulation": "red", "false_premise": "blue", "factual": "green",
                  "leading": "purple", "multi_hop": "orange", "math": "brown"}
    colors = [cat_colors.get(c, "gray") for c in cats]

    ax.bar(indices, c_rates, color='green', alpha=0.5, label='Correct rate')
    ax.bar(indices, [-h for h in h_rates], color='red', alpha=0.5, label='Halluc rate (neg)')
    for i, p in enumerate(prompt_results):
        if p["is_bifurcating"]:
            ax.annotate('★', (i, 0.5), ha='center', fontsize=14, color='gold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Prompt index')
    ax.set_ylabel('Rate')
    ax.set_title(f'Per-Prompt Correct vs Hallucination Rate ({N_SAMPLES} samples each, T={TEMPERATURE})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "04_bifurcation_rates.png", dpi=150)
    plt.close()

    print(f"  Figures saved to {fig_dir}/")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  Total prompts:        {len(prompts):3d}                          │")
    print(f"  │  Bifurcating (★):      {len(bifurcating):3d}  ← true trajectory test │")
    print(f"  │  Near-bifurcating:     {len(near_bifurcating):3d}                          │")
    print(f"  │  Always correct:       {len(always_correct):3d}                          │")
    print(f"  │  Always hallucinating: {len(always_hall):3d}                          │")
    print(f"  └────────────────────────────────────────────────────────┘")

    if bifurcating:
        print(f"\n  BIFURCATION FOUND!")
        print(f"  → Same prompt, same model, different outcomes")
        print(f"  → Hallucination IS stochastic for these prompts")

        if trajectory_data:
            # Find earliest divergence across all prompts
            all_onsets = [td["onset_step"] for td in trajectory_data.values()
                          if td["onset_step"] is not None]
            if all_onsets:
                print(f"\n  Divergence Analysis:")
                print(f"    Earliest onset: step {min(all_onsets)}")
                print(f"    Mean onset: step {np.mean(all_onsets):.1f}")
                print(f"    → Intervention window: steps 0-{min(all_onsets)}")
    else:
        print(f"\n  NO BIFURCATION FOUND.")
        print(f"  → Hallucination is DETERMINISTIC for this model")
        print(f"  → Temperature only adds surface noise, doesn't change trajectory")
        print(f"  → Model decides at prompt encoding time, not during generation")
        print(f"  → This rules out generation-time trajectory control entirely")

    print(f"\n{'='*70}")
    print("Done.")

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    def to_serializable(obj):
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [to_serializable(v) for v in obj]
        return obj

    save_data = to_serializable({
        "model": model_name,
        "n_samples": N_SAMPLES,
        "temperature": TEMPERATURE,
        "n_total": len(prompts),
        "n_bifurcating": len(bifurcating),
        "n_near_bifurcating": len(near_bifurcating),
        "n_always_correct": len(always_correct),
        "n_always_hall": len(always_hall),
        "prompt_results": [{
            "idx": p["idx"], "prompt": p["prompt"], "category": p["category"],
            "counts": p["counts"], "correct_rate": p["correct_rate"],
            "hallucination_rate": p["hallucination_rate"],
            "is_bifurcating": p["is_bifurcating"],
        } for p in prompt_results],
        "bifurcating_prompts": [{
            "idx": p["idx"], "prompt": p["prompt"], "category": p["category"],
            "counts": p["counts"],
        } for p in bifurcating],
        "trajectory_summary": {
            str(idx): {
                "prompt": td["prompt"][:80],
                "kl_per_step": td["kl_per_step"],
                "onset_step": td["onset_step"],
                "n_steps": td["n_steps"],
            } for idx, td in trajectory_data.items()
        } if trajectory_data else {},
    })

    out_dir = Path(__file__).parent / "results"
    out_path = out_dir / f"bifurcation_{model_name.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
