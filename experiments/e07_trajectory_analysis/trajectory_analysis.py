"""
E07: Trajectory Analysis — KL Divergence + Trajectory Clustering

PREVIOUS FINDINGS (E01-E06):
  - Φ exists (d=15-22) but is NOT causal
  - Linear projection doesn't work — model compensates
  - Φ is a readout direction, not a causal direction
  - 32 genuine hallucinations found (confabulation, false premise, factual)

THIS EXPERIMENT answers TWO questions:

  Q1 (KL DIVERGENCE): WHEN does hallucination diverge from correct?
    - At each generation step: KL(P_hall || P_correct) in logit space
    - At each LAYER within each step: logit lens → intermediate predictions
    - Find: the (layer, step) where the model "decides" to hallucinate

  Q2 (TRAJECTORY CLUSTERING): WHAT is the shape of divergence?
    - PCA/t-SNE of hidden states across steps
    - Do trajectories START in the same place and diverge?
    - Or are they separated from step 0?
    - Is it a bifurcation (sudden split) or gradual drift?

  BONUS: Per-category analysis
    - Confabulation vs false_premise vs factual — different dynamics?

Usage:
    python trajectory_analysis.py [model_name]
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
from sklearn.decomposition import PCA

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_e05_data():
    e05_path = Path(__file__).parent.parent / "e05_find_hallucination" / "results" / "multitoken_hallucination_Qwen_Qwen2.5-1.5B.json"
    with open(e05_path) as f:
        return json.load(f)


def get_prompts_from_e05():
    sys.path.insert(0, str(Path(__file__).parent.parent / "e05_find_hallucination"))
    from multitoken_hallucination import get_multitoken_prompts
    prompts = get_multitoken_prompts()
    return {i: p for i, p in enumerate(prompts)}


# ═══════════════════════════════════════════════════════════════════
# GENERATION WITH FULL CACHE
# ═══════════════════════════════════════════════════════════════════

def generate_with_full_cache(model, prompt, max_new_tokens=20):
    """
    Generate tokens, caching:
      - Residual stream at every (layer, step)
      - Logits at every step
      - Logit lens (W_U applied to intermediate layers) at every (layer, step)
    """
    tokenizer = model.tokenizer
    input_ids = model.to_tokens(prompt)
    n_layers = model.cfg.n_layers
    W_U = model.W_U  # [d_model, d_vocab]

    generated_ids = []
    all_states = []       # [step][layer] = [d_model] tensor
    all_logit_lens = []   # [step][layer] = [d_vocab] tensor (top-k probs)
    all_final_probs = []  # [step] = [d_vocab] probs
    all_tokens = []       # [step] = (token_str, prob)
    current_ids = input_ids.clone()

    for step in range(max_new_tokens):
        with torch.no_grad():
            logits, cache = model.run_with_cache(current_ids)

        # Final logits → probs
        final_logits = logits[0, -1, :].float().cpu()
        final_probs = torch.softmax(final_logits, dim=-1)
        all_final_probs.append(final_probs)

        # Per-layer states and logit lens
        step_states = []
        step_logit_lens = []

        for l in range(n_layers):
            h = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].float()

            # Logit lens: apply unembedding to intermediate state
            # intermediate_logits = h @ W_U
            with torch.no_grad():
                intermediate_logits = (h.to(W_U.device).to(W_U.dtype) @ W_U).float().cpu()
            intermediate_probs = torch.softmax(intermediate_logits, dim=-1)

            step_states.append(h.cpu())
            step_logit_lens.append(intermediate_probs)

        all_states.append(step_states)
        all_logit_lens.append(step_logit_lens)

        # Greedy decode
        next_id = final_probs.argmax().item()
        next_token = tokenizer.decode([next_id])
        next_prob = final_probs[next_id].item()
        all_tokens.append((next_token, next_prob))
        generated_ids.append(next_id)

        # Append
        next_tensor = torch.tensor([[next_id]], device=current_ids.device)
        current_ids = torch.cat([current_ids, next_tensor], dim=1)

        # Stop conditions
        if next_id == tokenizer.eos_token_id:
            break
        if len(generated_ids) >= 3:
            recent = tokenizer.decode(generated_ids[-3:])
            if recent.count('\n') >= 2:
                break

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    generated_text = tokenizer.decode(generated_ids)
    return {
        "text": generated_text,
        "tokens": all_tokens,
        "states": all_states,         # [step][layer] → [d_model]
        "logit_lens": all_logit_lens,  # [step][layer] → [d_vocab]
        "final_probs": all_final_probs, # [step] → [d_vocab]
        "n_steps": len(all_tokens),
    }


# ═══════════════════════════════════════════════════════════════════
# KL DIVERGENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def compute_kl(p, q, eps=1e-10):
    """KL(P || Q) — how much P diverges from Q."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum().item()


def analyze_kl_divergence(hall_data_list, correct_data_list, n_layers, max_steps=15):
    """
    Compute KL divergence matrices:
      1. Final logit KL: per step, KL(mean_hall_probs || mean_correct_probs)
      2. Logit lens KL: per (layer, step)
    """
    # Align to common step count
    n_steps = min(max_steps,
                  min(d["n_steps"] for d in hall_data_list),
                  min(d["n_steps"] for d in correct_data_list))

    # === Final logit KL per step ===
    kl_per_step = []
    for step in range(n_steps):
        # Mean probability distribution at this step
        hall_probs = torch.stack([d["final_probs"][step] for d in hall_data_list]).mean(dim=0)
        corr_probs = torch.stack([d["final_probs"][step] for d in correct_data_list]).mean(dim=0)
        kl = compute_kl(hall_probs, corr_probs)
        kl_per_step.append(kl)

    # === Logit lens KL per (layer, step) ===
    kl_matrix = np.zeros((n_layers, n_steps))  # [layer, step]
    for step in range(n_steps):
        for l in range(n_layers):
            hall_probs = torch.stack([d["logit_lens"][step][l] for d in hall_data_list]).mean(dim=0)
            corr_probs = torch.stack([d["logit_lens"][step][l] for d in correct_data_list]).mean(dim=0)
            kl_matrix[l, step] = compute_kl(hall_probs, corr_probs)

    return kl_per_step, kl_matrix, n_steps


def analyze_per_sample_divergence(hall_data_list, correct_data_list, n_layers, max_steps=15):
    """
    For each hallucination sample, compute KL against the mean correct distribution.
    Returns per-sample KL curves.
    """
    n_steps = min(max_steps,
                  min(d["n_steps"] for d in hall_data_list),
                  min(d["n_steps"] for d in correct_data_list))

    # Mean correct probs per step
    corr_mean_probs = []
    for step in range(n_steps):
        corr_probs = torch.stack([d["final_probs"][step] for d in correct_data_list]).mean(dim=0)
        corr_mean_probs.append(corr_probs)

    # Per-sample KL
    per_sample_kl = []  # [n_hall, n_steps]
    for d in hall_data_list:
        sample_kl = []
        for step in range(n_steps):
            kl = compute_kl(d["final_probs"][step], corr_mean_probs[step])
            sample_kl.append(kl)
        per_sample_kl.append(sample_kl)

    return per_sample_kl, n_steps


# ═══════════════════════════════════════════════════════════════════
# TRAJECTORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_trajectories(hall_data_list, correct_data_list, hall_categories,
                         correct_categories, n_layers, target_layers, max_steps=15):
    """
    PCA analysis of hidden state trajectories at selected layers.
    Returns PCA projections for visualization.
    """
    n_steps = min(max_steps,
                  min(d["n_steps"] for d in hall_data_list),
                  min(d["n_steps"] for d in correct_data_list))

    trajectory_results = {}

    for layer in target_layers:
        # Collect all states at this layer: [n_samples * n_steps, d_model]
        all_points = []
        labels = []    # "hall" or "correct"
        steps = []
        categories = []
        sample_ids = []

        for i, d in enumerate(hall_data_list):
            for step in range(n_steps):
                h = d["states"][step][layer]
                all_points.append(h.numpy())
                labels.append("hall")
                steps.append(step)
                categories.append(hall_categories[i])
                sample_ids.append(i)

        for i, d in enumerate(correct_data_list):
            for step in range(n_steps):
                h = d["states"][step][layer]
                all_points.append(h.numpy())
                labels.append("correct")
                steps.append(step)
                categories.append(correct_categories[i])
                sample_ids.append(len(hall_data_list) + i)

        X = np.array(all_points)  # [n_total, d_model]

        # PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)

        # Compute trajectory-level metrics
        # 1. Initial separation (step 0)
        hall_step0 = X_pca[(np.array(labels) == "hall") & (np.array(steps) == 0)]
        corr_step0 = X_pca[(np.array(labels) == "correct") & (np.array(steps) == 0)]
        init_dist = np.linalg.norm(hall_step0.mean(axis=0) - corr_step0.mean(axis=0))

        # 2. Final separation (last step)
        last_step = n_steps - 1
        hall_last = X_pca[(np.array(labels) == "hall") & (np.array(steps) == last_step)]
        corr_last = X_pca[(np.array(labels) == "correct") & (np.array(steps) == last_step)]
        final_dist = np.linalg.norm(hall_last.mean(axis=0) - corr_last.mean(axis=0))

        # 3. Per-step separation
        step_dists = []
        for s in range(n_steps):
            h_s = X_pca[(np.array(labels) == "hall") & (np.array(steps) == s)]
            c_s = X_pca[(np.array(labels) == "correct") & (np.array(steps) == s)]
            if len(h_s) > 0 and len(c_s) > 0:
                step_dists.append(np.linalg.norm(h_s.mean(axis=0) - c_s.mean(axis=0)))
            else:
                step_dists.append(0)

        # 4. Within-class spread
        hall_spread = np.mean([np.std(X_pca[(np.array(labels) == "hall") & (np.array(steps) == s)], axis=0).mean()
                               for s in range(n_steps)])
        corr_spread = np.mean([np.std(X_pca[(np.array(labels) == "correct") & (np.array(steps) == s)], axis=0).mean()
                               for s in range(n_steps)])

        trajectory_results[layer] = {
            "X_pca": X_pca,
            "labels": labels,
            "steps": steps,
            "categories": categories,
            "sample_ids": sample_ids,
            "pca_explained": pca.explained_variance_ratio_.tolist(),
            "init_dist": float(init_dist),
            "final_dist": float(final_dist),
            "step_dists": step_dists,
            "hall_spread": float(hall_spread),
            "corr_spread": float(corr_spread),
            "n_steps": n_steps,
            "divergence_ratio": float(final_dist / (init_dist + 1e-8)),
        }

    return trajectory_results


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_kl_heatmap(kl_matrix, kl_per_step, n_steps, fig_dir):
    """Plot KL divergence heatmap (layer x step) + per-step curve."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Heatmap
    ax = axes[0]
    im = ax.imshow(kl_matrix[:, :n_steps], aspect='auto', cmap='hot',
                   interpolation='nearest', origin='lower')
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('Layer')
    ax.set_title('Logit Lens KL Divergence: Hallucination vs Correct\n(per layer × step)')
    plt.colorbar(im, ax=ax, label='KL(P_hall || P_correct)')

    # Annotate max
    max_idx = np.unravel_index(np.argmax(kl_matrix[:, :n_steps]), kl_matrix[:, :n_steps].shape)
    ax.plot(max_idx[1], max_idx[0], 'w*', markersize=15)
    ax.annotate(f'max KL={kl_matrix[max_idx]:.1f}\nL{max_idx[0]}, step {max_idx[1]}',
                xy=(max_idx[1], max_idx[0]), xytext=(max_idx[1]+1, max_idx[0]+3),
                color='white', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='white'))

    # Per-step curve
    ax = axes[1]
    ax.plot(range(n_steps), kl_per_step[:n_steps], 'b-o', markersize=5, linewidth=2)
    ax.set_xlabel('Generation Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Final Logit KL Divergence per Step')
    ax.grid(True, alpha=0.3)

    # Mark max divergence step
    max_step = np.argmax(kl_per_step[:n_steps])
    ax.axvline(x=max_step, color='red', linestyle='--', alpha=0.5)
    ax.annotate(f'max @ step {max_step}', xy=(max_step, kl_per_step[max_step]),
                xytext=(max_step+1, kl_per_step[max_step]*0.8), fontsize=9,
                arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.savefig(fig_dir / "01_kl_divergence_heatmap.png", dpi=150)
    plt.close()


def plot_per_sample_kl(per_sample_kl, hall_categories, n_steps, fig_dir):
    """Plot per-sample KL curves, colored by category."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = {"confabulation": "red", "false_premise": "blue", "factual": "green"}

    for i, kl_curve in enumerate(per_sample_kl):
        cat = hall_categories[i]
        color = colors.get(cat, "gray")
        ax.plot(range(n_steps), kl_curve[:n_steps], '-', color=color,
                alpha=0.3, linewidth=1)

    # Category means
    for cat, color in colors.items():
        cat_curves = [per_sample_kl[i] for i in range(len(per_sample_kl))
                      if hall_categories[i] == cat]
        if cat_curves:
            mean_curve = np.mean(cat_curves, axis=0)[:n_steps]
            ax.plot(range(n_steps), mean_curve, '-o', color=color,
                    linewidth=3, markersize=5, label=f'{cat} (n={len(cat_curves)})')

    ax.set_xlabel('Generation Step')
    ax.set_ylabel('KL(sample || mean_correct)')
    ax.set_title('Per-Sample KL Divergence by Category')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "02_per_sample_kl.png", dpi=150)
    plt.close()


def plot_trajectories(traj_results, target_layers, fig_dir):
    """Plot PCA trajectory projections for selected layers."""
    n_plots = len(target_layers)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for ax, layer in zip(axes, target_layers):
        tr = traj_results[layer]
        X = tr["X_pca"]
        labels = np.array(tr["labels"])
        steps = np.array(tr["steps"])
        n_steps = tr["n_steps"]

        # Plot correct trajectories (gray)
        correct_mask = labels == "correct"
        unique_samples = np.unique(np.array(tr["sample_ids"])[correct_mask])
        for sid in unique_samples:
            smask = (np.array(tr["sample_ids"]) == sid)
            pts = X[smask]
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], '-', color='gray', alpha=0.2, linewidth=1)
                ax.scatter(pts[0, 0], pts[0, 1], c='gray', s=20, alpha=0.3, zorder=3)

        # Plot hallucination trajectories (colored by category)
        cat_colors = {"confabulation": "red", "false_premise": "blue", "factual": "green"}
        hall_mask = labels == "hall"
        unique_hall = np.unique(np.array(tr["sample_ids"])[hall_mask])
        for sid in unique_hall:
            smask = (np.array(tr["sample_ids"]) == sid)
            cat = np.array(tr["categories"])[smask][0]
            color = cat_colors.get(cat, "orange")
            pts = X[smask]
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], '-', color=color, alpha=0.4, linewidth=1)
                ax.scatter(pts[0, 0], pts[0, 1], c=color, s=30, alpha=0.5, zorder=3)
                ax.scatter(pts[-1, 0], pts[-1, 1], c=color, s=50, marker='x',
                           alpha=0.7, zorder=4)

        # Mean trajectories
        hall_mean = []
        corr_mean = []
        for s in range(n_steps):
            h_pts = X[(labels == "hall") & (steps == s)]
            c_pts = X[(labels == "correct") & (steps == s)]
            if len(h_pts) > 0:
                hall_mean.append(h_pts.mean(axis=0))
            if len(c_pts) > 0:
                corr_mean.append(c_pts.mean(axis=0))

        if hall_mean:
            hall_mean = np.array(hall_mean)
            ax.plot(hall_mean[:, 0], hall_mean[:, 1], '-o', color='darkred',
                    linewidth=3, markersize=6, label='HALL mean', zorder=5)
        if corr_mean:
            corr_mean = np.array(corr_mean)
            ax.plot(corr_mean[:, 0], corr_mean[:, 1], '-o', color='black',
                    linewidth=3, markersize=6, label='CORRECT mean', zorder=5)

        exp_var = tr["pca_explained"]
        ax.set_xlabel(f'PC1 ({exp_var[0]:.1%})')
        ax.set_ylabel(f'PC2 ({exp_var[1]:.1%})')
        ax.set_title(f'Layer {layer}\ninit_d={tr["init_dist"]:.1f} → '
                     f'final_d={tr["final_dist"]:.1f} '
                     f'(ratio={tr["divergence_ratio"]:.2f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(fig_dir / "03_trajectory_pca.png", dpi=150)
    plt.close()


def plot_separation_by_step(traj_results, target_layers, fig_dir):
    """Plot inter-class distance vs step for each layer."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(target_layers)))

    for i, layer in enumerate(target_layers):
        tr = traj_results[layer]
        dists = tr["step_dists"]
        n_steps = tr["n_steps"]
        ax.plot(range(n_steps), dists[:n_steps], '-o', color=colors[i],
                markersize=4, linewidth=2, label=f'L{layer}')

    ax.set_xlabel('Generation Step')
    ax.set_ylabel('PCA Distance (hall mean − correct mean)')
    ax.set_title('Trajectory Separation Over Generation Steps')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "04_separation_by_step.png", dpi=150)
    plt.close()


def plot_kl_by_layer_profile(kl_matrix, n_steps, fig_dir):
    """Plot KL at each layer for step 0, 1, 2, 5."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    n_layers = kl_matrix.shape[0]

    target_steps = [s for s in [0, 1, 2, 3, 5, 8] if s < n_steps]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(target_steps)))

    for i, step in enumerate(target_steps):
        ax.plot(range(n_layers), kl_matrix[:, step], '-o', color=colors[i],
                markersize=3, linewidth=2, label=f'Step {step}')

    ax.set_xlabel('Layer')
    ax.set_ylabel('KL Divergence')
    ax.set_title('Layer-wise KL Divergence at Different Generation Steps\n'
                 '(Where in the network does divergence happen?)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "05_kl_layer_profile.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
    MAX_GEN_STEPS = 15

    print(f"{'='*70}")
    print(f"E07: TRAJECTORY ANALYSIS — KL Divergence + Clustering")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Max generation steps: {MAX_GEN_STEPS}\n")

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

    hall_indices = [s["idx"] for s in e05_data["hallucination_prompts"]]
    correct_indices = [s["idx"] for s in e05_data["all_samples"]
                       if s["classification"] == "CORRECT"]

    print(f"Hallucination cases: {len(hall_indices)}")
    print(f"Correct cases: {len(correct_indices)}\n")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: COLLECT DATA
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*70}")
    print("PHASE 1: Generating with full cache (logit lens + states)")
    print(f"{'='*70}\n")

    hall_data = []
    hall_categories = []
    print("  Hallucination cases:")
    for i, idx in enumerate(hall_indices):
        prompt_info = prompt_db[idx]
        prompt = prompt_info[0]
        category = prompt_info[3]
        print(f"    [{i+1}/{len(hall_indices)}] [{category:15s}] {prompt[:45]}...", end="", flush=True)

        t1 = time.time()
        data = generate_with_full_cache(model, prompt, max_new_tokens=MAX_GEN_STEPS)
        dt = time.time() - t1

        hall_data.append(data)
        hall_categories.append(category)
        print(f"  ({dt:.1f}s) {data['n_steps']} steps")

    correct_data = []
    correct_categories = []
    print("\n  Correct cases:")
    for i, idx in enumerate(correct_indices):
        prompt_info = prompt_db[idx]
        prompt = prompt_info[0]
        category = prompt_info[3]
        print(f"    [{i+1}/{len(correct_indices)}] [{category:15s}] {prompt[:45]}...", end="", flush=True)

        t1 = time.time()
        data = generate_with_full_cache(model, prompt, max_new_tokens=MAX_GEN_STEPS)
        dt = time.time() - t1

        correct_data.append(data)
        correct_categories.append(category)
        print(f"  ({dt:.1f}s) {data['n_steps']} steps")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: KL DIVERGENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 2: KL Divergence Analysis")
    print(f"{'='*70}\n")

    kl_per_step, kl_matrix, n_steps = analyze_kl_divergence(
        hall_data, correct_data, n_layers, max_steps=MAX_GEN_STEPS)

    print(f"  Final logit KL per step:")
    for s in range(n_steps):
        bar = "█" * int(min(kl_per_step[s], 50))
        print(f"    Step {s:2d}: KL={kl_per_step[s]:8.2f}  {bar}")

    # Find peak KL layer per step
    print(f"\n  Peak KL layer per step:")
    for s in range(min(n_steps, 10)):
        peak_layer = np.argmax(kl_matrix[:, s])
        peak_kl = kl_matrix[peak_layer, s]
        print(f"    Step {s:2d}: peak at L{peak_layer:2d} (KL={peak_kl:.2f})")

    # Overall peak
    max_layer, max_step = np.unravel_index(np.argmax(kl_matrix[:, :n_steps]),
                                            kl_matrix[:, :n_steps].shape)
    print(f"\n  GLOBAL PEAK: Layer {max_layer}, Step {max_step} "
          f"(KL={kl_matrix[max_layer, max_step]:.2f})")

    # Onset analysis: when does KL first exceed threshold?
    thresholds = [1.0, 2.0, 5.0, 10.0]
    print(f"\n  KL onset analysis (final logits):")
    for thresh in thresholds:
        onset = next((s for s in range(n_steps) if kl_per_step[s] > thresh), None)
        print(f"    KL > {thresh:4.1f}: {'step ' + str(onset) if onset is not None else 'never'}")

    # Per-sample KL
    per_sample_kl, _ = analyze_per_sample_divergence(
        hall_data, correct_data, n_layers, max_steps=MAX_GEN_STEPS)

    # Category-level analysis
    print(f"\n  KL by category (step 0):")
    for cat in set(hall_categories):
        cat_kls = [per_sample_kl[i][0] for i in range(len(per_sample_kl))
                   if hall_categories[i] == cat]
        if cat_kls:
            print(f"    {cat:15s}: mean={np.mean(cat_kls):.2f}  "
                  f"std={np.std(cat_kls):.2f}  n={len(cat_kls)}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: TRAJECTORY ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 3: Trajectory Clustering (PCA)")
    print(f"{'='*70}\n")

    target_layers = [0, 7, 14, 19, 24, 27]  # early, mid, peak, late
    traj_results = analyze_trajectories(
        hall_data, correct_data, hall_categories, correct_categories,
        n_layers, target_layers, max_steps=MAX_GEN_STEPS)

    print(f"  {'Layer':>5s}  {'Init Dist':>10s}  {'Final Dist':>11s}  "
          f"{'Ratio':>7s}  {'H Spread':>9s}  {'C Spread':>9s}  {'PCA Var':>8s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*11}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*8}")

    for layer in target_layers:
        tr = traj_results[layer]
        pca_var = sum(tr["pca_explained"][:2])
        print(f"  {layer:5d}  {tr['init_dist']:10.2f}  {tr['final_dist']:11.2f}  "
              f"{tr['divergence_ratio']:7.2f}  {tr['hall_spread']:9.2f}  "
              f"{tr['corr_spread']:9.2f}  {pca_var:7.1%}")

    # Bifurcation vs Drift analysis
    print(f"\n  Bifurcation vs Drift Analysis:")
    for layer in target_layers:
        tr = traj_results[layer]
        dists = tr["step_dists"]
        n = tr["n_steps"]
        if n > 3:
            # Is separation increasing, constant, or sudden?
            early_dist = np.mean(dists[:3])
            late_dist = np.mean(dists[-3:])
            mid_dist = np.mean(dists[n//3:2*n//3])

            if late_dist > 2 * early_dist and mid_dist > 1.5 * early_dist:
                pattern = "GRADUAL DRIFT"
            elif late_dist > 2 * early_dist and mid_dist < 1.3 * early_dist:
                pattern = "LATE BIFURCATION"
            elif early_dist > 0.8 * late_dist:
                pattern = "ALWAYS SEPARATED"
            else:
                pattern = "MIXED"

            print(f"    L{layer:2d}: early={early_dist:.2f} mid={mid_dist:.2f} "
                  f"late={late_dist:.2f}  → {pattern}")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    # Q1: When does divergence start?
    onset_step = next((s for s in range(n_steps) if kl_per_step[s] > 1.0), n_steps)
    print(f"\n  Q1: WHEN does divergence start?")
    print(f"    → Final logit KL > 1.0 at step {onset_step}")
    print(f"    → Peak layer at step 0: L{np.argmax(kl_matrix[:, 0])}")
    print(f"    → Global peak: L{max_layer}, step {max_step}")

    # Q2: What shape?
    print(f"\n  Q2: What is the trajectory SHAPE?")
    for layer in [14, 19]:
        if layer in traj_results:
            tr = traj_results[layer]
            print(f"    L{layer}: init_dist={tr['init_dist']:.2f}, "
                  f"final_dist={tr['final_dist']:.2f}, "
                  f"ratio={tr['divergence_ratio']:.2f}")

    print(f"\n{'='*70}")
    print("Done.")

    # ═══════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════
    fig_dir = Path(__file__).parent / "results" / "figures"
    os.makedirs(fig_dir, exist_ok=True)

    plot_kl_heatmap(kl_matrix, kl_per_step, n_steps, fig_dir)
    plot_per_sample_kl(per_sample_kl, hall_categories, n_steps, fig_dir)
    plot_trajectories(traj_results, target_layers, fig_dir)
    plot_separation_by_step(traj_results, target_layers, fig_dir)
    plot_kl_by_layer_profile(kl_matrix, n_steps, fig_dir)
    print(f"  Figures saved to {fig_dir}/")

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    def make_serializable(obj):
        """Convert numpy/torch types to Python native for JSON."""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    save_data = make_serializable({
        "model": model_name,
        "n_layers": n_layers,
        "n_hall": len(hall_indices),
        "n_correct": len(correct_indices),
        "n_steps": n_steps,
        "kl_per_step": kl_per_step,
        "kl_matrix": kl_matrix.tolist(),
        "kl_global_peak": {"layer": int(max_layer), "step": int(max_step),
                           "kl": float(kl_matrix[max_layer, max_step])},
        "per_sample_kl": per_sample_kl,
        "hall_categories": hall_categories,
        "trajectory_summary": {
            str(l): {
                "init_dist": tr["init_dist"],
                "final_dist": tr["final_dist"],
                "divergence_ratio": tr["divergence_ratio"],
                "step_dists": tr["step_dists"],
                "pca_explained": tr["pca_explained"],
            }
            for l, tr in traj_results.items()
        },
    })

    out_dir = Path(__file__).parent / "results"
    out_path = out_dir / f"trajectory_analysis_{model_name.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
