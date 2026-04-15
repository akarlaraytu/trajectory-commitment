"""
Experiment 2: Causal Tracing — WHERE does hallucination emerge?

GOAL: Map the causal structure of hallucination in the model.
Find which (layer, token) positions are causally responsible for
the model's output — not just correlational (Φ), but CAUSAL.

Method (adapted from Meng et al. 2022 / ROME):
  1. Clean run: model produces its natural output
  2. Corrupted run: noise added to subject embeddings → output changes
  3. Patch: restore clean activation at (layer, token) → measure recovery

Key outputs:
  - Heatmap: (layer, token) → causal effect (logit recovery)
  - Window analysis: contiguous layer ranges → causal effect
  - Bidirectional: correct→wrong AND wrong→correct
  - Temporal structure: does hallucination emerge gradually or abruptly?

Usage:
    python causal_tracing.py [model_name] [--n-prompts 40]
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)  # line-buffered


# ─── Prompt Set ───────────────────────────────────────────────────

def get_prompts():
    """Factual prompts with subject token ranges marked.

    Returns list of (prompt, correct_answers, subject_description, category).
    Subject = the key entity tokens (what carries the factual knowledge).
    """
    prompts = [
        # Geography — subject is the country name
        ("Q: What is the capital of France? A:", [" Paris"], "France", "geo"),
        ("Q: What is the capital of Japan? A:", [" Tokyo"], "Japan", "geo"),
        ("Q: What is the capital of Italy? A:", [" Rome"], "Italy", "geo"),
        ("Q: What is the capital of Spain? A:", [" Madrid"], "Spain", "geo"),
        ("Q: What is the capital of Russia? A:", [" Moscow"], "Russia", "geo"),
        ("Q: What is the capital of China? A:", [" Beijing", " Peking"], "China", "geo"),
        ("Q: What is the capital of Egypt? A:", [" Cairo"], "Egypt", "geo"),
        ("Q: What is the capital of Germany? A:", [" Berlin"], "Germany", "geo"),
        ("Q: What is the capital of Turkey? A:", [" Ankara", " Istanbul"], "Turkey", "geo"),
        ("Q: What is the capital of India? A:", [" New", " Delhi"], "India", "geo"),
        ("Q: What is the capital of Brazil? A:", [" Bras"], "Brazil", "geo"),
        ("Q: What is the capital of Australia? A:", [" Canberra"], "Australia", "geo"),
        ("Q: What is the capital of Canada? A:", [" Ottawa"], "Canada", "geo"),
        ("Q: What is the capital of South Korea? A:", [" Seoul"], "South Korea", "geo"),
        ("Q: What is the capital of Poland? A:", [" Warsaw"], "Poland", "geo"),
        ("Q: What is the capital of Sweden? A:", [" Stockholm"], "Sweden", "geo"),
        ("Q: What is the capital of Norway? A:", [" Oslo"], "Norway", "geo"),
        ("Q: What is the capital of Greece? A:", [" Athens"], "Greece", "geo"),
        ("Q: What is the capital of Argentina? A:", [" Buenos"], "Argentina", "geo"),
        ("Q: What is the capital of Thailand? A:", [" Bangkok"], "Thailand", "geo"),

        # Chemistry — subject is the element name
        ("The chemical symbol for iron is", [" Fe"], "iron", "chem"),
        ("The chemical symbol for gold is", [" Au"], "gold", "chem"),
        ("The chemical symbol for silver is", [" Ag"], "silver", "chem"),
        ("The chemical symbol for sodium is", [" Na"], "sodium", "chem"),
        ("The chemical symbol for potassium is", [" K"], "potassium", "chem"),
        ("The chemical symbol for calcium is", [" Ca"], "calcium", "chem"),
        ("The chemical symbol for lead is", [" Pb"], "lead", "chem"),
        ("The chemical symbol for mercury is", [" Hg"], "mercury", "chem"),
        ("The chemical symbol for tin is", [" Sn"], "tin", "chem"),
        ("The chemical symbol for copper is", [" Cu"], "copper", "chem"),

        # Physics — subject varies
        ("The Earth orbits the", [" Sun"], "Earth", "phys"),
        ("The Moon orbits the", [" Earth"], "Moon", "phys"),
        ("Light travels faster than", [" sound"], "Light", "phys"),
        ("Force equals mass times", [" acceleration", " accel"], "Force", "phys"),
        ("Einstein developed the theory of", [" relat"], "Einstein", "phys"),

        # History — subject is the event/entity
        ("World War II ended in", [" 1945", " 19"], "World War II", "hist"),
        ("The Berlin Wall fell in", [" 1989", " 19"], "Berlin Wall", "hist"),
        ("Columbus reached the Americas in", [" 1492", " 14"], "Columbus", "hist"),
        ("Napoleon was defeated at", [" Water"], "Napoleon", "hist"),
        ("The printing press was invented by", [" Gut", " Johannes"], "printing press", "hist"),
    ]
    return prompts


def find_subject_tokens(tokenizer, prompt, subject):
    """Find token indices corresponding to the subject in the prompt."""
    tokens = tokenizer.encode(prompt)
    subject_tokens = tokenizer.encode(subject)

    # Try to find subject tokens as subsequence
    for start in range(len(tokens)):
        match = True
        for j, st in enumerate(subject_tokens):
            if start + j >= len(tokens) or tokens[start + j] != st:
                match = False
                break
        if match:
            return list(range(start, start + len(subject_tokens)))

    # Fallback: try single-token encoding or partial match
    subject_lower = subject.lower()
    for i, tok_id in enumerate(tokens):
        decoded = tokenizer.decode([tok_id]).strip().lower()
        if decoded and subject_lower.startswith(decoded):
            # Found start, extend
            end = i + 1
            built = decoded
            while end < len(tokens) and len(built) < len(subject_lower):
                next_decoded = tokenizer.decode([tokens[end]]).strip().lower()
                built += next_decoded
                end += 1
            return list(range(i, end))

    # Last fallback: middle tokens (heuristic)
    mid = len(tokens) // 2
    return [max(0, mid - 1), mid]


# ─── Core Causal Tracing ─────────────────────────────────────────

def run_causal_trace(model, prompt, correct_answers, subject, tokenizer,
                     noise_scale=3.0, device="mps"):
    """
    Run causal tracing for a single prompt.

    Returns dict with:
      - clean_logit: logit for correct token under clean run
      - corrupted_logit: logit for correct token under corrupted run
      - heatmap: (n_layers, n_tokens) array of recovered logits
      - token_labels: list of decoded tokens for visualization
      - subject_indices: which tokens are the subject
    """
    n_layers = model.cfg.n_layers
    tokens = model.to_tokens(prompt)
    n_tokens = tokens.shape[1]

    # Find subject token positions
    subject_indices = find_subject_tokens(tokenizer, prompt, subject)

    # ─── Step 1: Clean run ───
    with torch.no_grad():
        clean_logits = model(tokens)[0, -1, :]

    clean_probs = torch.softmax(clean_logits.float().cpu(), dim=-1)
    top1_id = clean_probs.argmax().item()
    top1_token = tokenizer.decode([top1_id])

    is_correct = any(
        top1_token.strip().lower().startswith(a.strip().lower())
        for a in correct_answers)

    # Find correct token id (first matching answer)
    correct_token_id = None
    for a in correct_answers:
        a_tokens = tokenizer.encode(a)
        if a_tokens:
            correct_token_id = a_tokens[0]
            break
    if correct_token_id is None:
        correct_token_id = top1_id

    clean_logit = clean_logits[correct_token_id].float().cpu().item()
    clean_prob = clean_probs[correct_token_id].item()

    # ─── Step 2: Clean run WITH cache ───
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(tokens)

    # ─── Step 3: Corrupted run (noise on subject embeddings) ───
    def corrupt_hook(value, hook):
        """Add noise to subject token embeddings."""
        for idx in subject_indices:
            if idx < value.shape[1]:
                noise = torch.randn_like(value[0, idx]) * noise_scale
                value[0, idx] += noise
        return value

    with torch.no_grad():
        corrupted_logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[("hook_embed", corrupt_hook)]
        )[0, -1, :]

    corrupted_logit = corrupted_logits[correct_token_id].float().cpu().item()
    corrupted_prob = torch.softmax(corrupted_logits.float().cpu(), dim=-1)[correct_token_id].item()

    # ─── Step 4: Patch each (layer, token) and measure recovery ───
    heatmap = np.zeros((n_layers, n_tokens))

    for layer in range(n_layers):
        for tok_pos in range(n_tokens):
            hook_name = f"blocks.{layer}.hook_resid_post"
            clean_act = clean_cache[hook_name]

            def patch_hook(value, hook, _layer=layer, _pos=tok_pos, _clean=clean_act):
                value[0, _pos] = _clean[0, _pos]
                return value

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[
                        ("hook_embed", corrupt_hook),
                        (hook_name, patch_hook),
                    ]
                )[0, -1, :]

            patched_logit = patched_logits[correct_token_id].float().cpu().item()

            # Normalized recovery: 0 = no recovery (corrupted), 1 = full recovery (clean)
            total_effect = clean_logit - corrupted_logit
            if abs(total_effect) > 1e-6:
                recovery = (patched_logit - corrupted_logit) / total_effect
            else:
                recovery = 0.0

            heatmap[layer, tok_pos] = recovery

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Token labels for visualization
    token_labels = [tokenizer.decode([tokens[0, i].item()]) for i in range(n_tokens)]

    del clean_cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "clean_logit": clean_logit,
        "corrupted_logit": corrupted_logit,
        "clean_prob": clean_prob,
        "corrupted_prob": corrupted_prob,
        "total_effect": clean_logit - corrupted_logit,
        "is_correct": is_correct,
        "top1_token": top1_token,
        "heatmap": heatmap,
        "token_labels": token_labels,
        "subject_indices": subject_indices,
        "n_layers": n_layers,
        "n_tokens": n_tokens,
    }


# ─── Window Patching ─────────────────────────────────────────────

def run_window_patching(model, prompt, correct_answers, subject, tokenizer,
                        noise_scale=3.0, window_sizes=[1, 3, 5, 7], device="mps"):
    """
    Patch contiguous windows of layers (all tokens at subject positions).

    Returns: dict of window_size → (n_windows,) array of recovery values.
    """
    n_layers = model.cfg.n_layers
    tokens = model.to_tokens(prompt)
    subject_indices = find_subject_tokens(tokenizer, prompt, subject)

    # Clean run + cache
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(tokens)

    correct_token_id = None
    for a in correct_answers:
        a_tokens = tokenizer.encode(a)
        if a_tokens:
            correct_token_id = a_tokens[0]
            break
    if correct_token_id is None:
        correct_token_id = clean_logits[0, -1].argmax().item()

    clean_logit = clean_logits[0, -1, correct_token_id].float().cpu().item()

    # Corrupted run
    def corrupt_hook(value, hook):
        for idx in subject_indices:
            if idx < value.shape[1]:
                value[0, idx] += torch.randn_like(value[0, idx]) * noise_scale
        return value

    with torch.no_grad():
        corrupted_logits = model.run_with_hooks(
            tokens, fwd_hooks=[("hook_embed", corrupt_hook)]
        )
    corrupted_logit = corrupted_logits[0, -1, correct_token_id].float().cpu().item()
    total_effect = clean_logit - corrupted_logit

    results = {}
    for ws in window_sizes:
        recoveries = []
        for start in range(n_layers - ws + 1):
            hooks = [("hook_embed", corrupt_hook)]
            for l in range(start, start + ws):
                hook_name = f"blocks.{l}.hook_resid_post"
                clean_act = clean_cache[hook_name]

                def patch_hook(value, hook, _clean=clean_act):
                    for idx in subject_indices:
                        if idx < value.shape[1]:
                            value[0, idx] = _clean[0, idx]
                    return value

                hooks.append((hook_name, patch_hook))

            with torch.no_grad():
                patched_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            patched_logit = patched_logits[0, -1, correct_token_id].float().cpu().item()

            if abs(total_effect) > 1e-6:
                recovery = (patched_logit - corrupted_logit) / total_effect
            else:
                recovery = 0.0
            recoveries.append(recovery)

        results[ws] = np.array(recoveries)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    del clean_cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results, total_effect


# ─── Visualization ───────────────────────────────────────────────

def plot_heatmap(avg_heatmap, token_labels, subject_indices, save_path, title=""):
    """Plot (layer, token) causal effect heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(12, len(token_labels[0]) * 0.8), 10))

    # Use the first prompt's token labels as representative
    im = ax.imshow(avg_heatmap, aspect='auto', cmap='RdBu_r',
                   vmin=-0.2, vmax=1.0, interpolation='nearest')

    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title or "Causal Tracing Heatmap", fontsize=14)

    plt.colorbar(im, ax=ax, label="Recovery (0=corrupted, 1=clean)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_layer_profile(layer_effects, save_path, title=""):
    """Plot per-layer causal effect (averaged over tokens)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: subject tokens only
    ax = axes[0]
    ax.plot(layer_effects["subject_mean"], 'b-o', markersize=4, label="Subject tokens")
    ax.fill_between(range(len(layer_effects["subject_mean"])),
                    np.array(layer_effects["subject_mean"]) - np.array(layer_effects["subject_std"]),
                    np.array(layer_effects["subject_mean"]) + np.array(layer_effects["subject_std"]),
                    alpha=0.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recovery")
    ax.set_title("Causal Effect at Subject Tokens")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: last token (decision point)
    ax = axes[1]
    ax.plot(layer_effects["last_mean"], 'r-o', markersize=4, label="Last token")
    ax.fill_between(range(len(layer_effects["last_mean"])),
                    np.array(layer_effects["last_mean"]) - np.array(layer_effects["last_std"]),
                    np.array(layer_effects["last_mean"]) + np.array(layer_effects["last_std"]),
                    alpha=0.2, color='red')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recovery")
    ax.set_title("Causal Effect at Decision Token (last)")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title or "Layer-wise Causal Profile", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_window_results(window_results, n_layers, save_path, title=""):
    """Plot window patching results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (ws, recoveries) in enumerate(sorted(window_results.items())):
        starts = np.arange(len(recoveries))
        mid_points = starts + ws / 2
        ax.plot(mid_points, recoveries, '-o', markersize=3,
                color=colors[i % len(colors)], label=f"window={ws}", alpha=0.8)

    ax.set_xlabel("Layer (window center)", fontsize=12)
    ax.set_ylabel("Recovery", fontsize=12)
    ax.set_title(title or "Window Patching: Contiguous Layer Ranges", fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bidirectional(correct_profile, wrong_profile, save_path):
    """Compare causal profiles of correct vs wrong answers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subject tokens
    ax = axes[0]
    ax.plot(correct_profile["subject_mean"], 'g-o', markersize=4, label="Correct answers", alpha=0.8)
    ax.plot(wrong_profile["subject_mean"], 'r-o', markersize=4, label="Wrong answers", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recovery at Subject Tokens")
    ax.set_title("Bidirectional: Subject Token Causality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Last token
    ax = axes[1]
    ax.plot(correct_profile["last_mean"], 'g-o', markersize=4, label="Correct answers", alpha=0.8)
    ax.plot(wrong_profile["last_mean"], 'r-o', markersize=4, label="Wrong answers", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recovery at Decision Token")
    ax.set_title("Bidirectional: Decision Point Causality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Correct vs Wrong: WHERE does the model diverge?", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Analysis ─────────────────────────────────────────────────────

def compute_layer_profiles(all_traces):
    """Compute per-layer profiles from individual traces."""
    n_layers = all_traces[0]["n_layers"]

    subject_effects = [[] for _ in range(n_layers)]
    last_effects = [[] for _ in range(n_layers)]
    all_effects = [[] for _ in range(n_layers)]

    for trace in all_traces:
        hm = trace["heatmap"]
        subj = trace["subject_indices"]

        for layer in range(n_layers):
            # Subject tokens
            subj_vals = [hm[layer, t] for t in subj if t < hm.shape[1]]
            if subj_vals:
                subject_effects[layer].append(np.mean(subj_vals))

            # Last token
            last_effects[layer].append(hm[layer, -1])

            # All tokens
            all_effects[layer].append(np.mean(hm[layer, :]))

    return {
        "subject_mean": [float(np.mean(e)) if e else 0 for e in subject_effects],
        "subject_std": [float(np.std(e)) if e else 0 for e in subject_effects],
        "last_mean": [float(np.mean(e)) if e else 0 for e in last_effects],
        "last_std": [float(np.std(e)) if e else 0 for e in last_effects],
        "all_mean": [float(np.mean(e)) if e else 0 for e in all_effects],
    }


def find_critical_window(profile, key="subject_mean"):
    """Find the contiguous layer range with highest causal effect."""
    values = np.array(profile[key])
    n = len(values)

    best_score = -np.inf
    best_start = 0
    best_end = 0

    # Find the peak region (layers where recovery > threshold)
    threshold = np.mean(values) + 0.5 * np.std(values)

    for start in range(n):
        for end in range(start + 1, min(start + 10, n + 1)):
            window = values[start:end]
            score = np.mean(window)
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    return {
        "start": int(best_start),
        "end": int(best_end),
        "mean_recovery": float(best_score),
        "layers": list(range(best_start, best_end)),
    }


def temporal_structure_analysis(all_traces):
    """
    Analyze whether hallucination emerges gradually or abruptly.

    Computes the "divergence curve": at which layer do correct and wrong
    answers start to differ in their causal structure?
    """
    n_layers = all_traces[0]["n_layers"]

    correct_traces = [t for t in all_traces if t["is_correct"]]
    wrong_traces = [t for t in all_traces if not t["is_correct"]]

    if not correct_traces or not wrong_traces:
        return None

    # Per-layer: mean recovery for correct vs wrong
    divergence = []
    for layer in range(n_layers):
        correct_recovery = [t["heatmap"][layer, -1] for t in correct_traces]
        wrong_recovery = [t["heatmap"][layer, -1] for t in wrong_traces]

        c_mean = np.mean(correct_recovery)
        w_mean = np.mean(wrong_recovery)
        c_std = np.std(correct_recovery) + 1e-8
        w_std = np.std(wrong_recovery) + 1e-8

        # Cohen's d between correct and wrong at this layer
        pooled = np.sqrt((c_std**2 + w_std**2) / 2)
        d = (c_mean - w_mean) / pooled

        divergence.append({
            "layer": layer,
            "correct_mean": float(c_mean),
            "wrong_mean": float(w_mean),
            "cohen_d": float(d),
            "gap": float(c_mean - w_mean),
        })

    # Find first layer where gap becomes significant (|d| > 0.5)
    onset_layer = None
    for d in divergence:
        if abs(d["cohen_d"]) > 0.5:
            onset_layer = d["layer"]
            break

    # Find peak divergence
    peak = max(divergence, key=lambda x: abs(x["cohen_d"]))

    return {
        "divergence_curve": divergence,
        "onset_layer": onset_layer,
        "peak_layer": peak["layer"],
        "peak_cohen_d": peak["cohen_d"],
        "n_correct": len(correct_traces),
        "n_wrong": len(wrong_traces),
        "is_gradual": onset_layer is not None and (peak["layer"] - onset_layer) > 3,
    }


# ─── Main ─────────────────────────────────────────────────────────

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
    n_prompts = 40

    # Parse --n-prompts flag
    for i, arg in enumerate(sys.argv):
        if arg == "--n-prompts" and i + 1 < len(sys.argv):
            n_prompts = int(sys.argv[i + 1])

    print(f"{'='*60}")
    print(f"E02: CAUSAL TRACING")
    print(f"Where does hallucination emerge?")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Prompts: {n_prompts}")
    print()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    print(f"Loading {model_name}...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    tokenizer = model.tokenizer
    print(f"Loaded in {time.time()-t0:.0f}s. Layers: {model.cfg.n_layers}, d={model.cfg.d_model}")

    prompts = get_prompts()[:n_prompts]
    results_dir = Path(__file__).parent / "results"
    figures_dir = results_dir / "figures"

    # ─── Phase 1: Full Causal Tracing ───
    print(f"\n{'='*60}")
    print(f"Phase 1: Full (layer, token) causal tracing")
    print(f"{'='*60}\n")

    all_traces = []
    for i, (prompt, correct_answers, subject, category) in enumerate(prompts):
        t1 = time.time()
        trace = run_causal_trace(
            model, prompt, correct_answers, subject, tokenizer,
            noise_scale=3.0, device=device)
        trace["prompt"] = prompt
        trace["subject"] = subject
        trace["category"] = category
        all_traces.append(trace)

        status = "CORRECT" if trace["is_correct"] else "WRONG"
        dt = time.time() - t1
        print(f"  [{i+1}/{len(prompts)}] {status} ({dt:.1f}s) "
              f"clean={trace['clean_logit']:.2f} corrupt={trace['corrupted_logit']:.2f} "
              f"effect={trace['total_effect']:.2f}  |  {prompt[:50]}...")

    # Separate correct and wrong
    correct_traces = [t for t in all_traces if t["is_correct"]]
    wrong_traces = [t for t in all_traces if not t["is_correct"]]
    print(f"\n  Correct: {len(correct_traces)}, Wrong: {len(wrong_traces)}")

    # ─── Aggregate heatmaps ───
    print(f"\n--- Aggregate Analysis ---")

    # Overall profile
    overall_profile = compute_layer_profiles(all_traces)
    correct_profile = compute_layer_profiles(correct_traces) if correct_traces else None
    wrong_profile = compute_layer_profiles(wrong_traces) if wrong_traces else None

    # Critical window
    crit = find_critical_window(overall_profile, "subject_mean")
    print(f"\n  Critical window (subject tokens): layers {crit['start']}-{crit['end']-1}")
    print(f"  Mean recovery in window: {crit['mean_recovery']:.3f}")

    crit_last = find_critical_window(overall_profile, "last_mean")
    print(f"  Critical window (decision token): layers {crit_last['start']}-{crit_last['end']-1}")
    print(f"  Mean recovery in window: {crit_last['mean_recovery']:.3f}")

    # Layer-by-layer summary
    print(f"\n  Per-layer recovery (subject | last | all):")
    for l in range(model.cfg.n_layers):
        s = overall_profile["subject_mean"][l]
        last = overall_profile["last_mean"][l]
        a = overall_profile["all_mean"][l]
        bar = "█" * int(max(0, s) * 30)
        print(f"    L{l:2d}: subj={s:+.3f}  last={last:+.3f}  all={a:+.3f}  {bar}")

    # ─── Phase 2: Window Patching ───
    print(f"\n{'='*60}")
    print(f"Phase 2: Window patching (contiguous layer ranges)")
    print(f"{'='*60}\n")

    # Use a subset for window patching (it's expensive)
    window_prompts = prompts[:15]
    window_sizes = [1, 3, 5, 7]

    all_window_results = {ws: [] for ws in window_sizes}
    for i, (prompt, correct_answers, subject, category) in enumerate(window_prompts):
        t1 = time.time()
        w_results, w_effect = run_window_patching(
            model, prompt, correct_answers, subject, tokenizer,
            noise_scale=3.0, window_sizes=window_sizes, device=device)

        for ws in window_sizes:
            all_window_results[ws].append(w_results[ws])

        print(f"  [{i+1}/{len(window_prompts)}] ({time.time()-t1:.1f}s) effect={w_effect:.2f}  {prompt[:50]}...")

    # Average window results
    avg_window = {}
    for ws in window_sizes:
        stacked = np.array(all_window_results[ws])
        avg_window[ws] = stacked.mean(axis=0)

    # Find minimal intervention region
    print(f"\n  Window patching results:")
    for ws in window_sizes:
        best_idx = int(np.argmax(avg_window[ws]))
        best_val = avg_window[ws][best_idx]
        print(f"    window={ws}: best at layers {best_idx}-{best_idx+ws-1}, recovery={best_val:.3f}")

    # ─── Phase 3: Temporal Structure ───
    print(f"\n{'='*60}")
    print(f"Phase 3: Temporal structure — gradual or abrupt?")
    print(f"{'='*60}\n")

    temporal = temporal_structure_analysis(all_traces)
    if temporal:
        print(f"  Correct prompts: {temporal['n_correct']}")
        print(f"  Wrong prompts: {temporal['n_wrong']}")
        print(f"  Onset layer (|d|>0.5): {temporal['onset_layer']}")
        print(f"  Peak divergence: layer {temporal['peak_layer']} (d={temporal['peak_cohen_d']:+.2f})")
        print(f"  Structure: {'GRADUAL' if temporal['is_gradual'] else 'ABRUPT'}")

        print(f"\n  Divergence curve (correct vs wrong at decision token):")
        for d in temporal["divergence_curve"]:
            bar_c = "█" * int(max(0, d["correct_mean"]) * 20)
            bar_w = "░" * int(max(0, d["wrong_mean"]) * 20)
            print(f"    L{d['layer']:2d}: C={d['correct_mean']:+.3f} W={d['wrong_mean']:+.3f} "
                  f"d={d['cohen_d']:+.2f}  {bar_c}{bar_w}")
    else:
        print("  (Not enough correct/wrong prompts for analysis)")

    # ─── Phase 4: Bidirectional comparison ───
    print(f"\n{'='*60}")
    print(f"Phase 4: Bidirectional — correct vs wrong causal structure")
    print(f"{'='*60}\n")

    if correct_profile and wrong_profile:
        print("  Subject token profile comparison:")
        for l in range(model.cfg.n_layers):
            cs = correct_profile["subject_mean"][l]
            ws = wrong_profile["subject_mean"][l]
            diff = cs - ws
            marker = " ***" if abs(diff) > 0.1 else ""
            print(f"    L{l:2d}: correct={cs:+.3f}  wrong={ws:+.3f}  gap={diff:+.3f}{marker}")

        # Find divergence point
        gaps = [correct_profile["subject_mean"][l] - wrong_profile["subject_mean"][l]
                for l in range(model.cfg.n_layers)]
        max_gap_layer = int(np.argmax(np.abs(gaps)))
        print(f"\n  Max divergence: layer {max_gap_layer} (gap={gaps[max_gap_layer]:+.3f})")
    else:
        print("  (Not enough correct or wrong prompts)")

    # ─── Generate Plots ───
    print(f"\n{'='*60}")
    print(f"Generating plots...")
    print(f"{'='*60}\n")

    model_short = model_name.replace("/", "_")

    # 1. Average heatmap (use traces with matching token count)
    # Group by token count for proper averaging
    token_counts = {}
    for t in all_traces:
        tc = t["n_tokens"]
        if tc not in token_counts:
            token_counts[tc] = []
        token_counts[tc].append(t)

    # Use the most common token count group
    most_common_tc = max(token_counts, key=lambda k: len(token_counts[k]))
    group = token_counts[most_common_tc]
    if len(group) >= 3:
        avg_hm = np.mean([t["heatmap"] for t in group], axis=0)
        plot_heatmap(avg_hm, [group[0]["token_labels"]], group[0]["subject_indices"],
                     figures_dir / f"01_causal_heatmap_{model_short}.png",
                     f"Causal Tracing Heatmap (avg of {len(group)} prompts, {most_common_tc} tokens)")

    # 2. Layer profile
    plot_layer_profile(overall_profile,
                       figures_dir / f"02_layer_profile_{model_short}.png",
                       f"Layer-wise Causal Profile ({model_short})")

    # 3. Window patching
    plot_window_results(avg_window, model.cfg.n_layers,
                        figures_dir / f"03_window_patching_{model_short}.png",
                        f"Window Patching ({model_short})")

    # 4. Bidirectional
    if correct_profile and wrong_profile:
        plot_bidirectional(correct_profile, wrong_profile,
                           figures_dir / f"04_bidirectional_{model_short}.png")

    # ─── Save Results ───
    results = {
        "model": model_name,
        "n_prompts": len(prompts),
        "n_correct": len(correct_traces),
        "n_wrong": len(wrong_traces),
        "overall_profile": overall_profile,
        "correct_profile": correct_profile,
        "wrong_profile": wrong_profile,
        "critical_window_subject": crit,
        "critical_window_decision": crit_last,
        "window_patching": {str(k): v.tolist() for k, v in avg_window.items()},
        "temporal_structure": temporal,
        "per_prompt": [{
            "prompt": t["prompt"],
            "subject": t["subject"],
            "category": t["category"],
            "is_correct": t["is_correct"],
            "top1_token": t["top1_token"],
            "clean_logit": t["clean_logit"],
            "corrupted_logit": t["corrupted_logit"],
            "total_effect": t["total_effect"],
            "clean_prob": t["clean_prob"],
            "corrupted_prob": t["corrupted_prob"],
        } for t in all_traces],
    }

    out_path = results_dir / f"causal_tracing_{model_short}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results: {out_path}")

    # ─── VERDICT ───
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    print(f"\n  1. Critical causal region (subject): layers {crit['start']}-{crit['end']-1}")
    print(f"     Recovery: {crit['mean_recovery']:.3f}")
    print(f"\n  2. Critical causal region (decision): layers {crit_last['start']}-{crit_last['end']-1}")
    print(f"     Recovery: {crit_last['mean_recovery']:.3f}")

    if temporal:
        print(f"\n  3. Temporal structure: {'GRADUAL' if temporal['is_gradual'] else 'ABRUPT'}")
        print(f"     Onset: layer {temporal['onset_layer']}, Peak: layer {temporal['peak_layer']}")

    if correct_profile and wrong_profile:
        gaps = [correct_profile["subject_mean"][l] - wrong_profile["subject_mean"][l]
                for l in range(model.cfg.n_layers)]
        max_gap_layer = int(np.argmax(np.abs(gaps)))
        print(f"\n  4. Max correct/wrong divergence: layer {max_gap_layer}")
        print(f"     Gap: {gaps[max_gap_layer]:+.3f}")

    print(f"\n  5. TLoT implication:")
    print(f"     If GRADUAL → multi-layer trajectory control needed")
    print(f"     If ABRUPT  → single critical point intervention may work")
    print(f"\n{'='*60}")
    print(f"Done.")


if __name__ == "__main__":
    main()
