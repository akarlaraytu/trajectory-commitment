"""
Experiment 0-C: Completion-Based Φ Search

Instead of feeding complete "correct" and "wrong" statements,
we give the model incomplete prompts and sample completions.

The model's OWN decision process generates the labels:
  "The capital of France is" → "Paris" (correct) vs "Lyon" (wrong)

We extract activations at the DECISION POINT (last prompt token, e.g., "is")
BEFORE the model commits to an answer.

This eliminates:
  - Template leakage (each sample is a unique generation event)
  - Exit door bias (we look at the decision point, not the answer token)
  - External labeling bias (model labels itself)

Key insight: If Φ exists, it should appear in MIDDLE-LATE layers (L8-L14),
not the final layer. The final layer is just vocabulary preparation.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import os

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'


# ─── Fact Database ──────────────────────────────────────────────

@dataclass
class FactPrompt:
    """An incomplete factual prompt with known correct answers."""
    prompt: str              # e.g., "The capital of France is"
    correct_answers: list    # e.g., ["Paris", " Paris"]
    category: str            # e.g., "geography"


def get_fact_prompts() -> list[FactPrompt]:
    """Large pool of factual prompts with known correct completions."""
    return [
        # Geography - capitals
        FactPrompt("The capital of France is", [" Paris"], "geography"),
        FactPrompt("The capital of Germany is", [" Berlin"], "geography"),
        FactPrompt("The capital of Japan is", [" Tokyo"], "geography"),
        FactPrompt("The capital of Italy is", [" Rome"], "geography"),
        FactPrompt("The capital of Spain is", [" Madrid"], "geography"),
        FactPrompt("The capital of Australia is", [" Canberra"], "geography"),
        FactPrompt("The capital of Canada is", [" Ottawa"], "geography"),
        FactPrompt("The capital of Brazil is", [" Bras"], "geography"),
        FactPrompt("The capital of Egypt is", [" Cairo"], "geography"),
        FactPrompt("The capital of India is", [" New", " Delhi"], "geography"),
        FactPrompt("The capital of Turkey is", [" Ankara"], "geography"),
        FactPrompt("The capital of Russia is", [" Moscow"], "geography"),
        FactPrompt("The capital of China is", [" Beijing"], "geography"),
        FactPrompt("The capital of South Korea is", [" Seoul"], "geography"),
        FactPrompt("The capital of Mexico is", [" Mexico"], "geography"),
        FactPrompt("The capital of Argentina is", [" Buenos"], "geography"),
        FactPrompt("The capital of Poland is", [" Warsaw"], "geography"),
        FactPrompt("The capital of Sweden is", [" Stockholm"], "geography"),
        FactPrompt("The capital of Norway is", [" Oslo"], "geography"),
        FactPrompt("The capital of Greece is", [" Athens"], "geography"),
        # Science
        FactPrompt("Water boils at 100 degrees", [" Celsius", " C"], "science"),
        FactPrompt("The chemical symbol for gold is", [" Au"], "science"),
        FactPrompt("The chemical symbol for iron is", [" Fe"], "science"),
        FactPrompt("The chemical symbol for silver is", [" Ag"], "science"),
        FactPrompt("The speed of light is approximately", [" 3", " 300"], "science"),
        FactPrompt("The Earth orbits the", [" Sun"], "science"),
        FactPrompt("The Moon orbits the", [" Earth"], "science"),
        FactPrompt("DNA stands for deoxyrib", ["onucle", "on"], "science"),
        FactPrompt("Photosynthesis converts sunlight into", [" chemical", " energy", " glucose"], "science"),
        FactPrompt("The atomic number of oxygen is", [" 8", " eight"], "science"),
        FactPrompt("The atomic number of hydrogen is", [" 1", " one"], "science"),
        FactPrompt("The atomic number of carbon is", [" 6", " six"], "science"),
        FactPrompt("Diamonds are made of", [" carbon"], "science"),
        FactPrompt("The pH of pure water is", [" 7", " seven"], "science"),
        FactPrompt("The closest star to Earth is the", [" Sun"], "science"),
        FactPrompt("Albert Einstein developed the theory of", [" relat"], "science"),
        FactPrompt("Isaac Newton discovered the law of", [" grav"], "science"),
        FactPrompt("Charles Darwin proposed the theory of", [" evol", " natural"], "science"),
        FactPrompt("The periodic table was created by", [" Mend", " Dmitri"], "science"),
        FactPrompt("Electrons have a", [" negative"], "science"),
        # History
        FactPrompt("World War II ended in", [" 1945", " 19"], "history"),
        FactPrompt("The Berlin Wall fell in", [" 1989", " 19"], "history"),
        FactPrompt("The first moon landing was in", [" 1969", " 19"], "history"),
        FactPrompt("Christopher Columbus reached the Americas in", [" 1492", " 14"], "history"),
        FactPrompt("The Declaration of Independence was signed in", [" 1776", " 17"], "history"),
        FactPrompt("The French Revolution began in", [" 1789", " 17"], "history"),
        FactPrompt("The Titanic sank in", [" 1912", " 19", " April"], "history"),
        FactPrompt("Julius Caesar was assassinated in", [" 44"], "history"),
        FactPrompt("The Renaissance began in", [" Italy", " the"], "history"),
        FactPrompt("The printing press was invented by", [" Gut", " Johannes"], "history"),
        FactPrompt("The Industrial Revolution started in", [" Britain", " England", " the"], "history"),
        FactPrompt("Alexander the Great was from", [" Mac", " Greece"], "history"),
        FactPrompt("The Soviet Union dissolved in", [" 1991", " 19"], "history"),
        FactPrompt("Napoleon was defeated at", [" Water"], "history"),
        FactPrompt("The Magna Carta was signed in", [" 12"], "history"),
        FactPrompt("The Wright brothers invented the", [" airplane", " air", " first"], "history"),
        FactPrompt("The Cold War was between the", [" United", " US", " Soviet"], "history"),
        FactPrompt("Shakespeare wrote", [" Hamlet", " Romeo", " Mac", " plays"], "history"),
        FactPrompt("The Great Wall of China was built", [" to", " during", " over"], "history"),
        FactPrompt("Martin Luther King Jr. gave his famous", [" \"", " I", " speech"], "history"),
    ]


# ─── Completion-Based Data Generation ───────────────────────────

@dataclass
class CompletionSample:
    prompt: str
    completion_token: str
    is_correct: bool
    category: str
    layer_activations: dict = field(default_factory=dict)  # layer -> activation vector


def generate_completion_data(
    model,
    fact_prompts: list[FactPrompt],
    n_samples_per_prompt: int = 30,
    temperature: float = 1.5,
    seed: int = 42,
) -> list[CompletionSample]:
    """
    For each fact prompt, sample multiple completions from the model.
    Label each as correct/wrong based on the first generated token.

    Higher temperature = more diverse completions = more wrong answers to learn from.
    """
    torch.manual_seed(seed)
    samples = []
    tokenizer = model.tokenizer

    for fp_idx, fp in enumerate(fact_prompts):
        tokens = model.to_tokens(fp.prompt)  # (1, seq_len)

        # Get logits for next token
        with torch.no_grad():
            logits = model(tokens)[:, -1, :]  # (1, vocab_size)

        # Sample multiple completions (CPU for MPS compatibility)
        probs = torch.softmax(logits[0].float().cpu() / temperature, dim=-1)

        sampled_ids = torch.multinomial(probs, n_samples_per_prompt, replacement=True)

        for sid in sampled_ids:
            token_str = tokenizer.decode([sid.item()])

            # Check if this completion matches any correct answer
            is_correct = any(
                token_str.strip().lower().startswith(ans.strip().lower())
                for ans in fp.correct_answers
            )

            samples.append(CompletionSample(
                prompt=fp.prompt,
                completion_token=token_str,
                is_correct=is_correct,
                category=fp.category,
            ))

        if fp_idx % 10 == 0:
            n_correct = sum(1 for s in samples if s.is_correct)
            print(f"  Prompt {fp_idx+1}/{len(fact_prompts)}: "
                  f"{len(samples)} samples ({n_correct} correct, "
                  f"{len(samples)-n_correct} wrong)")

    n_correct = sum(1 for s in samples if s.is_correct)
    n_wrong = len(samples) - n_correct
    print(f"\nTotal: {len(samples)} samples — {n_correct} correct, {n_wrong} wrong")

    if n_correct < 20 or n_wrong < 20:
        print("WARNING: Too few samples in one class. Adjust temperature.")

    return samples


# ─── Activation Extraction at Decision Point ────────────────────

def extract_decision_point_activations(
    model,
    samples: list[CompletionSample],
    layers: list[int] = None,
) -> dict:
    """
    Extract activations at the DECISION POINT — the last token of the prompt
    (e.g., "is" in "The capital of France is").

    This is WHERE the model decides what to generate next.
    NOT at the generated answer token (that's already the output).
    """
    if layers is None:
        layers = list(range(model.cfg.n_layers))

    device = model.cfg.device
    activations = {l: [] for l in layers}
    labels = []
    categories = []

    # Group samples by prompt to avoid redundant forward passes
    from collections import defaultdict
    prompt_groups = defaultdict(list)
    for i, s in enumerate(samples):
        prompt_groups[s.prompt].append(i)

    print(f"Extracting decision-point activations from {len(prompt_groups)} unique prompts...")

    for pg_idx, (prompt, sample_indices) in enumerate(prompt_groups.items()):
        tokens = model.to_tokens(prompt)  # (1, seq_len)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(
                    f"blocks.{l}.hook_resid_post" in name for l in layers
                ),
            )

        # Extract at LAST prompt token (decision point)
        for l in layers:
            key = f"blocks.{l}.hook_resid_post"
            if key in cache:
                # Last token position = decision point
                act = cache[key][0, -1, :].float().cpu()  # (d_model,)
                # Same activation for all samples from this prompt
                for _ in sample_indices:
                    activations[l].append(act)

        for idx in sample_indices:
            labels.append(samples[idx].is_correct)
            categories.append(samples[idx].category)

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if pg_idx % 10 == 0:
            print(f"  {pg_idx+1}/{len(prompt_groups)} prompts")

    # Stack
    for l in layers:
        activations[l] = torch.stack(activations[l])

    return activations, np.array(labels), np.array(categories)


# ─── Analysis ───────────────────────────────────────────────────

def analyze_layer_completion(
    layer: int,
    act_correct: np.ndarray,
    act_wrong: np.ndarray,
    n_random_baselines: int = 100,
    seed: int = 42,
):
    """Analyze a single layer with proper train/test split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.RandomState(seed)

    # Balance classes
    n = min(len(act_correct), len(act_wrong))
    if n < 10:
        return None

    act_c = act_correct[:n]
    act_w = act_wrong[:n]
    d = act_c.shape[1]

    # 1. Mean-diff direction
    direction = act_w.mean(0) - act_c.mean(0)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # 2. Split-half stability
    indices = rng.permutation(n)
    half = n // 2
    if half < 5:
        return None

    dir_a = act_w[indices[:half]].mean(0) - act_c[indices[:half]].mean(0)
    dir_a = dir_a / (np.linalg.norm(dir_a) + 1e-8)
    dir_b = act_w[indices[half:2*half]].mean(0) - act_c[indices[half:2*half]].mean(0)
    dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-8)
    stability = float(np.dot(dir_a, dir_b))

    # 3. Mean-diff probe accuracy (train on half, test on half)
    train_c, test_c = act_c[indices[:half]], act_c[indices[half:2*half]]
    train_w, test_w = act_w[indices[:half]], act_w[indices[half:2*half]]

    train_dir = train_w.mean(0) - train_c.mean(0)
    train_dir = train_dir / (np.linalg.norm(train_dir) + 1e-8)

    proj_c = test_c @ train_dir
    proj_w = test_w @ train_dir
    threshold = (proj_c.mean() + proj_w.mean()) / 2
    mean_acc = float(((proj_c < threshold).mean() + (proj_w >= threshold).mean()) / 2)

    # 4. LDA (logistic regression) with cross-validation
    X = np.vstack([act_c, act_w])
    y = np.array([0]*n + [1]*n)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Simple train/test for LDA
    X_train = np.vstack([scaler.transform(train_c), scaler.transform(train_w)])
    y_train = np.array([0]*len(train_c) + [1]*len(train_w))
    X_test = np.vstack([scaler.transform(test_c), scaler.transform(test_w)])
    y_test = np.array([0]*len(test_c) + [1]*len(test_w))

    clf = LogisticRegression(max_iter=5000, C=1.0)
    clf.fit(X_train, y_train)
    lda_acc = float(clf.score(X_test, y_test))

    lda_dir = (clf.coef_[0] / scaler.scale_).copy()
    lda_dir = lda_dir / (np.linalg.norm(lda_dir) + 1e-8)

    # Agreement
    mean_lda_cos = float(np.dot(direction, lda_dir))

    # 5. Random baseline
    random_accs = []
    for _ in range(n_random_baselines):
        rand_dir = rng.randn(d).astype(np.float32)
        rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-8)
        rp_c = test_c @ rand_dir
        rp_w = test_w @ rand_dir
        r_thresh = (rp_c.mean() + rp_w.mean()) / 2
        r_acc = float(((rp_c < r_thresh).mean() + (rp_w >= r_thresh).mean()) / 2)
        random_accs.append(r_acc)
    random_acc = float(np.mean(random_accs))

    # 6. Separation gap
    proj_c_all = act_c @ direction
    proj_w_all = act_w @ direction
    gap = float(proj_w_all.mean() - proj_c_all.mean())

    return {
        'layer': layer,
        'stability': stability,
        'mean_acc': mean_acc,
        'lda_acc': lda_acc,
        'random_acc': random_acc,
        'signal_mean': mean_acc - random_acc,
        'signal_lda': lda_acc - random_acc,
        'mean_lda_cos': mean_lda_cos,
        'gap': gap,
        'n_correct': len(act_correct),
        'n_wrong': len(act_wrong),
        'direction': direction,
        'lda_direction': lda_dir,
        'proj_correct': proj_c_all,
        'proj_wrong': proj_w_all,
    }


# ─── Main ───────────────────────────────────────────────────────

def run_experiment(
    model_name: str = "pythia-410m",
    n_samples_per_prompt: int = 30,
    temperature: float = 1.5,
    device: str = "auto",
    output_dir: str = None,
):
    import transformer_lens as tl

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if output_dir is None:
        output_dir = Path(__file__).parent / "results_completion"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 0-C: COMPLETION-BASED Φ SEARCH")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"Samples per prompt: {n_samples_per_prompt}")
    print(f"Temperature: {temperature}")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {model_name}...")
    model = tl.HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )

    # Get fact prompts
    fact_prompts = get_fact_prompts()
    categories = sorted(set(fp.category for fp in fact_prompts))
    print(f"Fact prompts: {len(fact_prompts)} across {categories}")

    # Generate completion data
    print(f"\n--- Generating completion data ---")
    samples = generate_completion_data(
        model, fact_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
        temperature=temperature,
    )

    # Show some examples
    print(f"\nExample completions:")
    shown = 0
    for s in samples:
        if shown >= 15:
            break
        mark = "✓" if s.is_correct else "✗"
        print(f"  {mark} {s.prompt}... → '{s.completion_token.strip()}'")
        shown += 1

    # Extract activations at decision point
    print(f"\n--- Extracting decision-point activations ---")
    n_layers = model.cfg.n_layers
    activations, labels, cats = extract_decision_point_activations(
        model, samples, layers=list(range(n_layers)),
    )

    correct_mask = labels == True
    wrong_mask = labels == False

    print(f"\nSamples: {correct_mask.sum()} correct, {wrong_mask.sum()} wrong")

    # Analyze each layer
    print(f"\n--- Analyzing {n_layers} layers ---")
    print(f"{'Layer':>6s}  {'Stab':>6s}  {'MeanAcc':>8s}  {'LDAAcc':>8s}  "
          f"{'RandAcc':>8s}  {'M↔L cos':>8s}  {'Signal':>8s}")
    print("-" * 72)

    results = []
    for layer_idx in range(n_layers):
        act = activations[layer_idx]
        act_c = act[correct_mask].numpy()
        act_w = act[wrong_mask].numpy()

        r = analyze_layer_completion(layer_idx, act_c, act_w)
        if r is None:
            print(f"  {layer_idx:4d}  SKIPPED (too few samples)")
            continue

        results.append(r)
        signal = max(r['signal_mean'], r['signal_lda'])

        marker = ""
        if r['stability'] > 0.7 and r['mean_acc'] > 0.7:
            marker = " ** STRONG (mean-diff)"
        if r['lda_acc'] > 0.75 and r['signal_lda'] > 0.2:
            marker += " ** STRONG (LDA)"

        print(f"  {layer_idx:4d}  {r['stability']:>6.3f}  "
              f"{r['mean_acc']:>8.3f}  {r['lda_acc']:>8.3f}  "
              f"{r['random_acc']:>8.3f}  "
              f"{r['mean_lda_cos']:>8.3f}  "
              f"{signal:>8.3f}{marker}")

    if not results:
        print("\nNo layers could be analyzed. Check data balance.")
        return

    # Find best layer — prioritize MIDDLE layers (not exit door!)
    # Weight middle-late layers higher
    def layer_score(r):
        layer_frac = r['layer'] / n_layers
        # Penalize first 20% and last 10% of layers
        if layer_frac < 0.2:
            position_weight = 0.5
        elif layer_frac > 0.9:
            position_weight = 0.7  # slight penalty for exit door
        else:
            position_weight = 1.0
        signal = max(r['signal_mean'], r['signal_lda'])
        return signal * position_weight

    best = max(results, key=layer_score)
    best_signal = max(best['signal_mean'], best['signal_lda'])

    print(f"\n--- Best layer: {best['layer']} ---")
    print(f"  Stability:      {best['stability']:.4f}")
    print(f"  Mean-diff acc:  {best['mean_acc']:.4f}")
    print(f"  LDA acc:        {best['lda_acc']:.4f}")
    print(f"  Random acc:     {best['random_acc']:.4f}")
    print(f"  Signal (mean):  {best['signal_mean']:.4f}")
    print(f"  Signal (LDA):   {best['signal_lda']:.4f}")
    print(f"  Mean↔LDA cos:   {best['mean_lda_cos']:.4f}")
    print(f"  N correct:      {best['n_correct']}")
    print(f"  N wrong:        {best['n_wrong']}")

    # Cross-category generalization
    print(f"\n--- Cross-category generalization (layer {best['layer']}) ---")
    best_act = activations[best['layer']]

    for train_cat in categories:
        for test_cat in categories:
            if train_cat == test_cat:
                continue

            train_mask_c = correct_mask & (cats == train_cat)
            train_mask_w = wrong_mask & (cats == train_cat)

            if train_mask_c.sum() < 5 or train_mask_w.sum() < 5:
                continue

            train_c = best_act[train_mask_c].numpy()
            train_w = best_act[train_mask_w].numpy()
            train_dir = train_w.mean(0) - train_c.mean(0)
            train_dir = train_dir.numpy() if isinstance(train_dir, torch.Tensor) else train_dir
            train_dir = train_dir / (np.linalg.norm(train_dir) + 1e-8)

            test_mask_c = correct_mask & (cats == test_cat)
            test_mask_w = wrong_mask & (cats == test_cat)

            if test_mask_c.sum() < 5 or test_mask_w.sum() < 5:
                continue

            test_c = best_act[test_mask_c].numpy()
            test_w = best_act[test_mask_w].numpy()

            proj_c = test_c @ train_dir
            proj_w = test_w @ train_dir
            thresh = (proj_c.mean() + proj_w.mean()) / 2
            acc = float(((proj_c < thresh).mean() + (proj_w >= thresh).mean()) / 2)

            print(f"  {train_cat:12s} → {test_cat:12s}: {acc:.3f}")

    # Diagnostic plots
    print(f"\n--- Generating plots ---")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Plot 1: Layer sensitivity
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Experiment 0-C: Completion-Based Φ — {model_name}",
                     fontsize=14, fontweight="bold")

        layers = [r['layer'] for r in results]

        ax = axes[0, 0]
        ax.bar(layers, [r['stability'] for r in results],
               color=["#e74c3c" if s > 0.7 else "#95a5a6"
                      for s in [r['stability'] for r in results]])
        ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.7)
        ax.set_title("Direction Stability")
        ax.set_xlabel("Layer")

        ax = axes[0, 1]
        x = np.arange(len(layers))
        w = 0.25
        ax.bar(x-w, [r['mean_acc'] for r in results], w, label="Mean-diff", color="#3498db")
        ax.bar(x, [r['lda_acc'] for r in results], w, label="LDA", color="#2ecc71")
        ax.bar(x+w, [r['random_acc'] for r in results], w, label="Random", color="#e74c3c", alpha=0.5)
        ax.axhline(y=0.5, color="gray", linestyle=":")
        ax.set_title("Probe Accuracy by Method")
        ax.set_xlabel("Layer")
        ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)

        ax = axes[1, 0]
        signals_mean = [r['signal_mean'] for r in results]
        signals_lda = [r['signal_lda'] for r in results]
        ax.bar(x-0.15, signals_mean, 0.3, label="Mean-diff signal", color="#3498db")
        ax.bar(x+0.15, signals_lda, 0.3, label="LDA signal", color="#2ecc71")
        ax.axhline(y=0.2, color="green", linestyle="--", alpha=0.5, label="threshold")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title("Signal Above Random")
        ax.set_xlabel("Layer")
        ax.legend(fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)

        # Highlight middle layers
        ax = axes[1, 1]
        combined = [max(r['signal_mean'], r['signal_lda']) for r in results]
        colors = []
        for i, r in enumerate(results):
            frac = r['layer'] / n_layers
            if 0.3 <= frac <= 0.8:
                colors.append("#27ae60" if combined[i] > 0.1 else "#f39c12")
            else:
                colors.append("#95a5a6")
        ax.barh(layers, combined, color=colors)
        ax.set_title("Best Signal (green=middle layers)")
        ax.set_ylabel("Layer")
        ax.axvline(x=0.2, color="green", linestyle="--")

        plt.tight_layout()
        plt.savefig(figures_dir / "01_completion_layer_sensitivity.png", dpi=150)
        plt.close()
        print(f"  Saved: 01_completion_layer_sensitivity.png")

        # Plot 2: Projection distribution at best layer
        if best.get('proj_correct') is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(best['proj_correct'], bins=40, alpha=0.6, color="#2ecc71",
                    label=f"Correct (n={best['n_correct']})", density=True)
            ax.hist(best['proj_wrong'], bins=40, alpha=0.6, color="#e74c3c",
                    label=f"Wrong (n={best['n_wrong']})", density=True)
            ax.set_title(f"Projection Distribution — Layer {best['layer']}\n"
                        f"Mean-diff acc={best['mean_acc']:.3f}, "
                        f"LDA acc={best['lda_acc']:.3f}")
            ax.set_xlabel("Projection onto Φ direction")
            ax.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "02_completion_projection_dist.png", dpi=150)
            plt.close()
            print(f"  Saved: 02_completion_projection_dist.png")

    except ImportError:
        print("  matplotlib not available")

    # Verdict
    print(f"\n{'='*60}")
    print("VERDICT (Completion-Based)")
    print(f"{'='*60}")

    print(f"  Best layer: {best['layer']} (of {n_layers})")
    layer_position = best['layer'] / n_layers
    if layer_position > 0.9:
        print(f"  WARNING: Best layer is exit door ({layer_position:.0%})")
    elif 0.3 <= layer_position <= 0.8:
        print(f"  GOOD: Best layer is in middle-late range ({layer_position:.0%})")

    print(f"  Signal (mean-diff): {best['signal_mean']:.3f}")
    print(f"  Signal (LDA):       {best['signal_lda']:.3f}")
    print(f"  Best signal:        {best_signal:.3f} (threshold: 0.2)")
    print()

    if best_signal > 0.2 and best['stability'] > 0.5 and layer_position <= 0.9:
        print("PHI_EXISTS: Direction found in reasoning layers!")
        print("  → Proceed to Experiment 1")
        verdict = "PHI_EXISTS"
    elif best_signal > 0.2:
        print("PHI_SURFACE: Direction found but may be surface-level")
        if layer_position > 0.9:
            print("  → Signal in exit door only — not deep reasoning")
        verdict = "PHI_SURFACE"
    elif best_signal > 0.1:
        print("PHI_WEAK: Marginal signal detected")
        verdict = "PHI_WEAK"
    else:
        print("PHI_NOT_FOUND: No completion-based direction found")
        verdict = "PHI_NOT_FOUND"

    # Save results
    results_dict = {
        "model": model_name,
        "method": "completion-based",
        "temperature": temperature,
        "n_samples_per_prompt": n_samples_per_prompt,
        "n_prompts": len(fact_prompts),
        "n_correct": int(correct_mask.sum()),
        "n_wrong": int(wrong_mask.sum()),
        "verdict": verdict,
        "best_layer": best['layer'],
        "best_layer_position": layer_position,
        "best_stability": best['stability'],
        "best_mean_acc": best['mean_acc'],
        "best_lda_acc": best['lda_acc'],
        "best_random_acc": best['random_acc'],
        "best_signal_mean": best['signal_mean'],
        "best_signal_lda": best['signal_lda'],
        "best_mean_lda_cos": best['mean_lda_cos'],
        "per_layer": [
            {k: v for k, v in r.items()
             if k not in ('direction', 'lda_direction', 'proj_correct', 'proj_wrong')}
            for r in results
        ],
    }

    results_path = output_dir / f"e00c_results_{model_name.replace('/', '_')}.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save best direction
    if best.get('lda_direction') is not None:
        np.save(
            output_dir / f"phi_completion_lda_layer{best['layer']}_{model_name.replace('/', '_')}.npy",
            best['lda_direction']
        )
    np.save(
        output_dir / f"phi_completion_mean_layer{best['layer']}_{model_name.replace('/', '_')}.npy",
        best['direction']
    )

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 0-C: Completion-based Φ search")
    parser.add_argument("--model", type=str, default="pythia-410m")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Completions per prompt")
    parser.add_argument("--temperature", type=float, default=1.5,
                        help="Sampling temperature (higher = more wrong answers)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        n_samples_per_prompt=args.n_samples,
        temperature=args.temperature,
        device=args.device,
        output_dir=args.output_dir,
    )
