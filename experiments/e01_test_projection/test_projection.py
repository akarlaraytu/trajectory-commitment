"""
Experiment 1: Does π Work?

Prerequisites: Experiment 0 must have found a valid Φ direction.

THE GOLDEN QUESTION:
    "If we project out the hallucination direction during inference,
     does accuracy stay the same while hallucination drops?"

    Accuracy stable + hallucination ↓  →  PUBLISH
    Accuracy ↓                         →  projection too aggressive, tune λ
    No change                          →  Φ is wrong, go back to e00

CONTROLS:
    A. Real Φ direction vs RANDOM direction (same λ)
       → If random also works, direction doesn't matter = noise
    B. Log-scale λ sweep (0.01 → 1.0)
       → Catch subtle effects at small λ
    C. Perplexity measurement (not just accuracy)
       → Detect coherence damage even when accuracy holds

Usage:
    python test_projection.py \
        --model pythia-410m \
        --phi-path ../e00_find_phi/results/phi_direction_layer12_pythia-410m.npy \
        --phi-layer 12
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'


# ─── Projection Hook ──────────────────────────────────────────────

class TLoTProjectionHook:
    """
    The core TLoT intervention: removes forbidden direction from
    residual stream during forward pass.
    π_λ(h) = normalize(h - λ · (h·φ̂)φ̂)
    """

    def __init__(
        self,
        phi_direction: np.ndarray,
        lam: float = 1.0,
        preserve_norm: bool = True,
    ):
        self.phi = torch.tensor(phi_direction, dtype=torch.float32)
        self.phi = F.normalize(self.phi, dim=0)
        self.lam = lam
        self.preserve_norm = preserve_norm

        self.forbidden_components_before = []
        self.forbidden_components_after = []

    def hook_fn(self, value, hook):
        """
        TransformerLens hook function.
        value shape: (batch, seq_len, d_model)
        """
        device = value.device
        dtype = value.dtype
        phi = self.phi.to(device=device, dtype=dtype)

        h_last = value[:, -1, :]
        component_before = torch.abs(h_last @ phi).mean().item()
        self.forbidden_components_before.append(component_before)

        original_norm = value.norm(dim=-1, keepdim=True)
        projection = (value @ phi).unsqueeze(-1) * phi.unsqueeze(0).unsqueeze(0)
        value_safe = value - self.lam * projection

        if self.preserve_norm:
            safe_norm = value_safe.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            value_safe = value_safe * (original_norm / safe_norm)

        h_last_safe = value_safe[:, -1, :]
        component_after = torch.abs(h_last_safe @ phi).mean().item()
        self.forbidden_components_after.append(component_after)

        return value_safe

    def get_diagnostics(self) -> dict:
        if not self.forbidden_components_before:
            return {"n_calls": 0}
        return {
            "n_calls": len(self.forbidden_components_before),
            "mean_forbidden_before": float(np.mean(self.forbidden_components_before)),
            "mean_forbidden_after": float(np.mean(self.forbidden_components_after)),
            "reduction_ratio": float(
                1.0 - np.mean(self.forbidden_components_after) /
                max(np.mean(self.forbidden_components_before), 1e-8)
            ),
        }

    def reset_diagnostics(self):
        self.forbidden_components_before.clear()
        self.forbidden_components_after.clear()


# ─── Evaluation ───────────────────────────────────────────────────

@dataclass
class DiversityMetrics:
    """Output diversity — catches fake wins where model just becomes conservative."""
    mean_output_length: float  # avg tokens generated
    unique_unigrams: int
    unique_bigrams: int
    output_entropy: float  # Shannon entropy of unigram distribution
    empty_outputs: int  # how many times model refused to answer

    def is_collapsed(self, baseline: 'DiversityMetrics') -> bool:
        """True if diversity dropped significantly vs baseline."""
        if baseline.mean_output_length < 1e-8:
            return False
        length_ratio = self.mean_output_length / baseline.mean_output_length
        unigram_ratio = self.unique_unigrams / max(baseline.unique_unigrams, 1)
        return length_ratio < 0.5 or unigram_ratio < 0.5


@dataclass
class EvalResult:
    benchmark: str
    n_total: int
    n_correct: int
    accuracy: float
    mean_perplexity: float
    diversity: DiversityMetrics
    details: list[dict]


def generate_factual_qa() -> list[dict]:
    """
    Factual Q:A: prompts where Pythia-410M actually knows answers.
    Same prompts validated in causal_v2.py (58%+ baseline accuracy).
    """
    raw = [
        # Geography Q:A format
        ("Q: What is the capital of France? A:", [" Paris"], "geography"),
        ("Q: What is the capital of Japan? A:", [" Tokyo"], "geography"),
        ("Q: What is the capital of Italy? A:", [" Rome"], "geography"),
        ("Q: What is the capital of Spain? A:", [" Madrid"], "geography"),
        ("Q: What is the capital of Russia? A:", [" Moscow"], "geography"),
        ("Q: What is the capital of China? A:", [" Beijing"], "geography"),
        ("Q: What is the capital of Egypt? A:", [" Cairo"], "geography"),
        ("Q: What is the capital of Germany? A:", [" Berlin"], "geography"),
        ("Q: What is the capital of Turkey? A:", [" Ankara", " Istanbul"], "geography"),
        ("Q: What is the capital of India? A:", [" New", " Delhi"], "geography"),
        ("Q: What is the capital of Brazil? A:", [" Bras"], "geography"),
        ("Q: What is the capital of Australia? A:", [" Canberra"], "geography"),
        ("Q: What is the capital of Canada? A:", [" Ottawa"], "geography"),
        ("Q: What is the capital of South Korea? A:", [" Seoul"], "geography"),
        ("Q: What is the capital of Mexico? A:", [" Mexico"], "geography"),
        ("Q: What is the capital of Poland? A:", [" Warsaw"], "geography"),
        ("Q: What is the capital of Sweden? A:", [" Stockholm"], "geography"),
        ("Q: What is the capital of Norway? A:", [" Oslo"], "geography"),
        ("Q: What is the capital of Greece? A:", [" Athens"], "geography"),
        ("Q: What is the capital of Argentina? A:", [" Buenos"], "geography"),
        # Science
        ("The Earth orbits the", [" Sun"], "science"),
        ("The Moon orbits the", [" Earth"], "science"),
        ("The chemical symbol for iron is", [" Fe"], "science"),
        ("The chemical symbol for gold is", [" Au"], "science"),
        ("The chemical symbol for silver is", [" Ag"], "science"),
        ("Diamonds are made of", [" carbon"], "science"),
        ("The speed of light is approximately", [" 3", " 300"], "science"),
        ("The atomic number of oxygen is", [" 8", " eight"], "science"),
        ("The atomic number of hydrogen is", [" 1", " one"], "science"),
        ("Electrons have a", [" negative"], "science"),
        ("Water boils at 100 degrees", [" Celsius", " C"], "science"),
        ("The pH of pure water is", [" 7", " seven"], "science"),
        ("The closest star to Earth is the", [" Sun"], "science"),
        ("Photosynthesis produces", [" oxygen", " glucose"], "science"),
        ("Sound cannot travel through a", [" vacuum"], "science"),
        ("Einstein developed the theory of", [" relat"], "science"),
        ("Newton discovered the law of", [" grav"], "science"),
        ("Darwin proposed the theory of", [" evol", " natural"], "science"),
        ("The periodic table was created by", [" Mend", " Dmitri"], "science"),
        ("Light travels faster than", [" sound"], "science"),
        # History
        ("World War II ended in", [" 1945", " 19"], "history"),
        ("The Berlin Wall fell in", [" 1989", " 19"], "history"),
        ("The first moon landing was in", [" 1969", " 19"], "history"),
        ("Columbus reached the Americas in", [" 1492", " 14"], "history"),
        ("The Declaration of Independence was signed in", [" 1776", " 17"], "history"),
        ("The French Revolution began in", [" 1789", " 17"], "history"),
        ("The Titanic sank in", [" 1912", " 19", " April"], "history"),
        ("Shakespeare wrote", [" Hamlet", " Romeo", " Mac", " plays"], "history"),
        ("Napoleon was defeated at", [" Water"], "history"),
        ("The Soviet Union dissolved in", [" 1991", " 19"], "history"),
        ("The Industrial Revolution started in", [" Britain", " England", " the"], "history"),
        ("Alexander the Great was from", [" Mac", " Greece"], "history"),
        ("The Wright brothers invented the", [" airplane", " air", " first"], "history"),
        ("The printing press was invented by", [" Gut", " Johannes"], "history"),
        ("The Magna Carta was signed in", [" 12"], "history"),
        ("The Cold War was between the", [" United", " US", " Soviet"], "history"),
        ("Julius Caesar was assassinated in", [" 44"], "history"),
        ("The Renaissance began in", [" Italy", " the"], "history"),
        ("Martin Luther King Jr. gave his famous", [" \"", " I", " speech"], "history"),
        ("The Great Wall of China was built", [" to", " during", " over"], "history"),
    ]

    return [{"prompt": p, "answers": a, "category": c} for p, a, c in raw]


def _compute_diversity(generated_texts: list[str]) -> DiversityMetrics:
    """Compute output diversity metrics from generated texts."""
    from collections import Counter

    all_tokens = []
    all_bigrams = []
    lengths = []
    empty_count = 0

    for text in generated_texts:
        text = text.strip()
        if not text:
            empty_count += 1
            lengths.append(0)
            continue

        words = text.split()
        lengths.append(len(words))
        all_tokens.extend(words)

        for i in range(len(words) - 1):
            all_bigrams.append(f"{words[i]}_{words[i+1]}")

    # Entropy of unigram distribution
    if all_tokens:
        counter = Counter(all_tokens)
        total = sum(counter.values())
        probs = np.array([c / total for c in counter.values()])
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    else:
        entropy = 0.0

    return DiversityMetrics(
        mean_output_length=float(np.mean(lengths)) if lengths else 0.0,
        unique_unigrams=len(set(all_tokens)),
        unique_bigrams=len(set(all_bigrams)),
        output_entropy=entropy,
        empty_outputs=empty_count,
    )


def _check_factual_correct(generated: str, answers: list[str]) -> bool:
    """Check if model's first generated token matches any accepted answer."""
    gen = generated.strip()
    if not gen:
        return False
    first_word = gen.split()[0]
    return any(
        first_word.lower().startswith(a.strip().lower())
        for a in answers
    )


def evaluate_model(
    model,
    problems: list[dict],
    max_new_tokens: int = 10,
) -> EvalResult:
    """
    Evaluate model accuracy, perplexity, AND output diversity.
    Diversity catches fake wins where model becomes conservative.
    """
    details = []
    n_correct = 0
    perplexities = []
    generated_texts = []

    for i, prob in enumerate(problems):
        tokens = model.to_tokens(prob["prompt"])
        with torch.no_grad():
            output = model.generate(
                tokens, max_new_tokens=max_new_tokens, temperature=0.0,
            )

            full_logits = model(output)
            generated_len = output.shape[1] - tokens.shape[1]
            if generated_len > 0:
                start = tokens.shape[1] - 1
                end = output.shape[1] - 1
                logits_slice = full_logits[0, start:end, :]
                target_slice = output[0, start+1:end+1]
                log_probs = F.log_softmax(logits_slice.float(), dim=-1)
                token_log_probs = log_probs.gather(1, target_slice.unsqueeze(1)).squeeze(1)
                ppl = torch.exp(-token_log_probs.mean()).item()
                perplexities.append(ppl)

        generated = model.to_string(output[0, tokens.shape[1]:])
        generated_texts.append(generated)

        is_correct = _check_factual_correct(generated, prob["answers"])

        if is_correct:
            n_correct += 1

        details.append({
            "prompt": prob["prompt"][:80],
            "expected": prob["answers"][0],
            "generated": generated.strip()[:50],
            "correct": is_correct,
            "category": prob.get("category", ""),
            "perplexity": perplexities[-1] if perplexities else None,
        })

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i + 1}/{len(problems)} "
                  f"(acc: {n_correct/(i+1):.3f}, "
                  f"ppl: {np.mean(perplexities[-20:]):.1f})")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    diversity = _compute_diversity(generated_texts)

    return EvalResult(
        benchmark="factual_qa",
        n_total=len(problems),
        n_correct=n_correct,
        accuracy=n_correct / len(problems),
        mean_perplexity=float(np.mean(perplexities)) if perplexities else 0.0,
        diversity=diversity,
        details=details,
    )


# ─── Visualization ────────────────────────────────────────────────

def generate_plots(
    sweep_real: list[dict],
    sweep_random: list[dict],
    model_name: str,
    phi_layer: int,
    output_dir: Path,
):
    """Generate all diagnostic plots for experiment 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    lambdas_real = [r["lambda"] for r in sweep_real]
    acc_real = [r["accuracy"] for r in sweep_real]
    ppl_real = [r["perplexity"] for r in sweep_real]
    red_real = [r["diagnostics"].get("reduction_ratio", 0) for r in sweep_real]

    lambdas_rand = [r["lambda"] for r in sweep_random]
    acc_rand = [r["accuracy"] for r in sweep_random]
    ppl_rand = [r["perplexity"] for r in sweep_random]

    baseline_acc = acc_real[0]
    baseline_ppl = ppl_real[0]

    # ─── PLOT 1: The Money Plot — λ Curve ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Experiment 1: Does π Work? — {model_name}, Layer {phi_layer}",
                 fontsize=14, fontweight="bold")

    # 1a. Accuracy vs λ: Real Φ vs Random
    ax = axes[0, 0]
    ax.plot(lambdas_real, acc_real, "g-o", linewidth=2, markersize=6, label="Real Φ")
    ax.plot(lambdas_rand, acc_rand, "r--x", linewidth=1.5, markersize=6,
            label="Random direction", alpha=0.7)
    ax.axhline(y=baseline_acc, color="gray", linestyle=":", alpha=0.5, label="baseline")
    ax.axhline(y=baseline_acc - 0.05, color="orange", linestyle="--", alpha=0.4,
               label="-5% threshold")
    ax.set_xlabel("λ (projection strength)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Projection Strength")
    ax.legend(fontsize=8)
    ax.set_xscale("log") if min(lambdas_real) > 0 else None

    # 1b. Perplexity vs λ: coherence damage check
    ax = axes[0, 1]
    ax.plot(lambdas_real, ppl_real, "g-o", linewidth=2, markersize=6, label="Real Φ")
    ax.plot(lambdas_rand, ppl_rand, "r--x", linewidth=1.5, markersize=6,
            label="Random direction", alpha=0.7)
    ax.axhline(y=baseline_ppl, color="gray", linestyle=":", alpha=0.5, label="baseline")
    ax.set_xlabel("λ (projection strength)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Coherence (Perplexity) vs Projection Strength")
    ax.legend(fontsize=8)

    # 1c. Forbidden component reduction
    ax = axes[1, 0]
    ax.plot(lambdas_real, red_real, "b-o", linewidth=2, markersize=6)
    ax.set_xlabel("λ (projection strength)")
    ax.set_ylabel("Forbidden Component Reduction Ratio")
    ax.set_title("How Much Φ Component is Removed")
    ax.set_ylim(-0.1, 1.1)

    # 1d. THE GOLDEN PLOT: Accuracy delta vs Forbidden reduction
    ax = axes[1, 1]
    acc_delta_real = [a - baseline_acc for a in acc_real]
    for i, (dr, da, lam) in enumerate(zip(red_real, acc_delta_real, lambdas_real)):
        color = "#27ae60" if da >= -0.05 and dr > 0.3 else "#e74c3c" if da < -0.05 else "#f39c12"
        ax.scatter(dr, da, c=color, s=100, zorder=5, edgecolors="black")
        ax.annotate(f"λ={lam:.2f}", (dr, da), fontsize=7,
                    textcoords="offset points", xytext=(5, 5))

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=-0.05, color="orange", linestyle="--", alpha=0.4)
    ax.axvline(x=0.3, color="green", linestyle="--", alpha=0.4)
    ax.set_xlabel("Forbidden Component Reduction")
    ax.set_ylabel("Accuracy Change (from baseline)")
    ax.set_title("THE GOLDEN PLOT: Hallucination Removed vs Accuracy Cost")

    # Green zone = win
    ax.fill_between([0.3, 1.0], [-0.05, -0.05], [0.2, 0.2],
                     alpha=0.08, color="green")
    ax.text(0.6, 0.05, "WIN ZONE", fontsize=12, color="green",
            alpha=0.5, ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(figures_dir / "01_lambda_sweep.png", dpi=150)
    plt.close()
    print(f"  Saved: 01_lambda_sweep.png")

    # ─── PLOT 2: Per-sample analysis at best λ ───
    # Find best λ (highest reduction with acc >= baseline - 0.05)
    best_idx = 0
    for i, r in enumerate(sweep_real):
        if r["lambda"] == 0:
            continue
        red = r["diagnostics"].get("reduction_ratio", 0)
        if red > 0.3 and r["accuracy"] >= baseline_acc - 0.05:
            if best_idx == 0 or r["accuracy"] > sweep_real[best_idx]["accuracy"]:
                best_idx = i

    if best_idx > 0 and "per_problem_forbidden" in sweep_real[best_idx]:
        forbidden_vals = sweep_real[best_idx]["per_problem_forbidden"]
        correctness = [d["correct"] for d in sweep_real[best_idx].get("details", [])]

        if forbidden_vals and correctness and len(forbidden_vals) == len(correctness):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            correct_forbidden = [f for f, c in zip(forbidden_vals, correctness) if c]
            wrong_forbidden = [f for f, c in zip(forbidden_vals, correctness) if not c]

            ax.hist(correct_forbidden, bins=30, alpha=0.6, color="#2ecc71",
                    label=f"Correct (n={len(correct_forbidden)})", density=True)
            ax.hist(wrong_forbidden, bins=30, alpha=0.6, color="#e74c3c",
                    label=f"Wrong (n={len(wrong_forbidden)})", density=True)
            ax.set_xlabel("Forbidden Component Magnitude (after projection)")
            ax.set_ylabel("Density")
            ax.set_title(f"Residual Forbidden Component — λ={sweep_real[best_idx]['lambda']}")
            ax.legend()

            plt.tight_layout()
            plt.savefig(figures_dir / "02_per_sample_forbidden.png", dpi=150)
            plt.close()
            print(f"  Saved: 02_per_sample_forbidden.png")

    # ─── PLOT 3: Accuracy comparison bar chart ───
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Pick representative λ values
    representative = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    real_bars = []
    rand_bars = []
    x_labels = []

    for lam in representative:
        r_real = next((r for r in sweep_real if abs(r["lambda"] - lam) < 0.02), None)
        r_rand = next((r for r in sweep_random if abs(r["lambda"] - lam) < 0.02), None)
        if r_real and r_rand:
            real_bars.append(r_real["accuracy"])
            rand_bars.append(r_rand["accuracy"])
            x_labels.append(f"λ={lam}")

    if real_bars:
        x = np.arange(len(x_labels))
        width = 0.35
        ax.bar(x - width/2, real_bars, width, label="Real Φ", color="#2ecc71")
        ax.bar(x + width/2, rand_bars, width, label="Random", color="#e74c3c", alpha=0.7)
        ax.axhline(y=baseline_acc, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Projection Strength")
        ax.set_ylabel("Accuracy")
        ax.set_title("Real Φ vs Random Direction: Accuracy Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(figures_dir / "03_real_vs_random_bars.png", dpi=150)
        plt.close()
        print(f"  Saved: 03_real_vs_random_bars.png")

    print(f"\n  All figures saved to: {figures_dir}/")


# ─── Main ─────────────────────────────────────────────────────────

def run_experiment(
    model_name: str,
    phi_path: str,
    phi_layer: int,
    lambdas: list[float] = None,
    device: str = "auto",
    output_dir: str = None,
):
    """
    Full Experiment 1 pipeline:
    1. Sweep λ with real Φ direction
    2. Sweep λ with RANDOM direction (control)
    3. Generate diagnostic plots
    4. Verdict
    """
    import transformer_lens as tl

    # Log-scale λ sweep: catch subtle effects
    if lambdas is None:
        lambdas = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load Φ direction
    phi_direction = np.load(phi_path)
    d_model = phi_direction.shape[0]
    print(f"Loaded Φ direction from {phi_path}, d_model={d_model}")

    # Generate RANDOM direction (control)
    rng = np.random.RandomState(42)
    random_direction = rng.randn(d_model).astype(np.float32)
    random_direction = random_direction / np.linalg.norm(random_direction)

    # Load model
    print(f"Loading {model_name} on {device}...")
    model = tl.HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    problems = generate_factual_qa()

    print(f"\n{'='*60}")
    print("EXPERIMENT 1: DOES π WORK?")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Φ layer: {phi_layer}")
    print(f"λ values: {lambdas}")
    print(f"Problems: {len(problems)} factual Q:A prompts")

    # ─── SWEEP A: Real Φ ───
    print(f"\n{'─'*40}")
    print("SWEEP A: Real Φ direction")
    print(f"{'─'*40}")

    sweep_real = []
    for lam in lambdas:
        print(f"\n  λ = {lam:.3f}")

        hook = TLoTProjectionHook(phi_direction=phi_direction, lam=lam, preserve_norm=True)
        hook_name = f"blocks.{phi_layer}.hook_resid_post"

        model.reset_hooks()
        if lam > 0:
            model.add_hook(hook_name, hook.hook_fn)

        eval_result = evaluate_model(model, problems)
        diagnostics = hook.get_diagnostics()

        print(f"    Accuracy: {eval_result.accuracy:.3f}  "
              f"Perplexity: {eval_result.mean_perplexity:.1f}  "
              f"Reduction: {diagnostics.get('reduction_ratio', 0):.3f}")

        sweep_real.append({
            "lambda": lam,
            "accuracy": eval_result.accuracy,
            "perplexity": eval_result.mean_perplexity,
            "n_correct": eval_result.n_correct,
            "n_total": eval_result.n_total,
            "diagnostics": diagnostics,
            "diversity": {
                "mean_length": eval_result.diversity.mean_output_length,
                "unique_unigrams": eval_result.diversity.unique_unigrams,
                "unique_bigrams": eval_result.diversity.unique_bigrams,
                "output_entropy": eval_result.diversity.output_entropy,
                "empty_outputs": eval_result.diversity.empty_outputs,
            },
        })

        model.reset_hooks()
        hook.reset_diagnostics()

    # ─── SWEEP B: Random direction (CONTROL) ───
    print(f"\n{'─'*40}")
    print("SWEEP B: RANDOM direction (control)")
    print(f"{'─'*40}")

    # Only test a subset of λ values for speed
    control_lambdas = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]

    sweep_random = []
    for lam in control_lambdas:
        print(f"\n  λ = {lam:.3f} (random)")

        hook = TLoTProjectionHook(phi_direction=random_direction, lam=lam, preserve_norm=True)
        hook_name = f"blocks.{phi_layer}.hook_resid_post"

        model.reset_hooks()
        if lam > 0:
            model.add_hook(hook_name, hook.hook_fn)

        eval_result = evaluate_model(model, problems)
        diagnostics = hook.get_diagnostics()

        print(f"    Accuracy: {eval_result.accuracy:.3f}  "
              f"Perplexity: {eval_result.mean_perplexity:.1f}")

        sweep_random.append({
            "lambda": lam,
            "accuracy": eval_result.accuracy,
            "perplexity": eval_result.mean_perplexity,
            "n_correct": eval_result.n_correct,
            "n_total": eval_result.n_total,
            "diagnostics": diagnostics,
        })

        model.reset_hooks()
        hook.reset_diagnostics()

    # ─── SWEEP C: SIGN FLIP — CAUSAL TEST ───
    # h' = h + λv  (ADD forbidden direction instead of removing)
    # If accuracy DROPS → direction is CAUSAL
    # If no change → direction is merely correlational
    print(f"\n{'─'*40}")
    print("SWEEP C: SIGN FLIP (h' = h + λv) — CAUSAL TEST")
    print(f"{'─'*40}")

    sign_flip_lambdas = [0.0, 0.1, 0.3, 0.5, 0.8]
    sweep_flip = []

    for lam in sign_flip_lambdas:
        print(f"\n  λ = +{lam:.3f} (ADDING forbidden direction)")

        # Negative lambda = adding instead of removing
        hook = TLoTProjectionHook(phi_direction=phi_direction, lam=-lam, preserve_norm=True)
        hook_name = f"blocks.{phi_layer}.hook_resid_post"

        model.reset_hooks()
        if lam > 0:
            model.add_hook(hook_name, hook.hook_fn)

        eval_result = evaluate_model(model, problems)
        diagnostics = hook.get_diagnostics()

        print(f"    Accuracy: {eval_result.accuracy:.3f}  "
              f"Perplexity: {eval_result.mean_perplexity:.1f}")

        sweep_flip.append({
            "lambda": lam,
            "accuracy": eval_result.accuracy,
            "perplexity": eval_result.mean_perplexity,
            "n_correct": eval_result.n_correct,
            "n_total": eval_result.n_total,
            "diagnostics": diagnostics,
            "diversity": {
                "mean_length": eval_result.diversity.mean_output_length,
                "unique_unigrams": eval_result.diversity.unique_unigrams,
                "unique_bigrams": eval_result.diversity.unique_bigrams,
                "output_entropy": eval_result.diversity.output_entropy,
                "empty_outputs": eval_result.diversity.empty_outputs,
            },
        })

        model.reset_hooks()
        hook.reset_diagnostics()

    # ─── Summary ───
    baseline_acc = sweep_real[0]["accuracy"]
    baseline_ppl = sweep_real[0]["perplexity"]
    baseline_diversity = sweep_real[0].get("diversity", {})

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nBaseline: accuracy={baseline_acc:.3f}, perplexity={baseline_ppl:.1f}")

    # Main table
    print(f"\n{'λ':>8s}  {'Acc(real)':>10s}  {'Acc(rand)':>10s}  "
          f"{'PPL(real)':>10s}  {'Diversity':>10s}  {'Reduction':>10s}")
    print("-" * 70)

    for r in sweep_real:
        lam = r["lambda"]
        r_rand = next((rr for rr in sweep_random if abs(rr["lambda"] - lam) < 0.02), None)

        acc_rand_str = f"{r_rand['accuracy']:.3f}" if r_rand else "—"
        red = r["diagnostics"].get("reduction_ratio", 0)
        div_entropy = r.get("diversity", {}).get("output_entropy", 0)

        delta = r["accuracy"] - baseline_acc
        marker = ""
        if delta >= 0 and red > 0.3:
            marker = " <<< WIN"
        elif delta < -0.05:
            marker = " !!! DROP"

        print(f"  {lam:>6.3f}  {r['accuracy']:>10.3f}  {acc_rand_str:>10s}  "
              f"{r['perplexity']:>10.1f}  {div_entropy:>10.2f}  {red:>10.3f}{marker}")

    # Sign flip table
    print(f"\n  SIGN FLIP (adding Φ direction — expect accuracy to DROP if causal):")
    print(f"  {'λ':>6s}  {'Accuracy':>10s}  {'Delta':>8s}")
    print(f"  " + "-" * 30)
    for r in sweep_flip:
        delta = r["accuracy"] - baseline_acc
        marker = " <<< CAUSAL" if delta < -0.05 else ""
        print(f"  {r['lambda']:>6.3f}  {r['accuracy']:>10.3f}  {delta:>+8.3f}{marker}")

    # ─── Verdict ───
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    # Condition A: real Φ better than random at same λ
    real_better_than_random = False
    for r_real in sweep_real:
        lam = r_real["lambda"]
        if lam == 0:
            continue
        r_rand = next((rr for rr in sweep_random if abs(rr["lambda"] - lam) < 0.02), None)
        if r_rand and r_real["accuracy"] > r_rand["accuracy"] + 0.02:
            real_better_than_random = True
            break

    # Condition B: exists λ where acc stable + reduction significant
    best_lambda = None
    for r in sweep_real:
        if r["lambda"] == 0:
            continue
        red = r["diagnostics"].get("reduction_ratio", 0)
        if red > 0.3 and r["accuracy"] >= baseline_acc - 0.05:
            if best_lambda is None or r["accuracy"] > best_lambda["accuracy"]:
                best_lambda = r

    # Condition C: perplexity stable
    ppl_stable = True
    if best_lambda:
        ppl_increase = best_lambda["perplexity"] / max(baseline_ppl, 1.0)
        if ppl_increase > 1.5:
            ppl_stable = False

    # Condition D: sign flip drops accuracy (CAUSAL)
    sign_flip_causal = False
    for r in sweep_flip:
        if r["lambda"] > 0 and r["accuracy"] < baseline_acc - 0.05:
            sign_flip_causal = True
            break

    # Condition E: diversity not collapsed
    diversity_ok = True
    if best_lambda:
        bl_entropy = baseline_diversity.get("output_entropy", 1.0)
        best_entropy = best_lambda.get("diversity", {}).get("output_entropy", 1.0)
        if bl_entropy > 0 and best_entropy / bl_entropy < 0.5:
            diversity_ok = False
            print(f"  WARNING: Output diversity collapsed at best λ!")
            print(f"    Baseline entropy: {bl_entropy:.2f}")
            print(f"    Best λ entropy:   {best_entropy:.2f}")
            print(f"    → Possible fake win: model became conservative, not smarter")

    print()
    conditions = {
        "A. Real > Random": real_better_than_random,
        "B. Acc stable + reduction": best_lambda is not None,
        "C. Perplexity stable": ppl_stable,
        "D. Sign flip → acc drops (CAUSAL)": sign_flip_causal,
        "E. Diversity preserved": diversity_ok,
    }
    for name, passed in conditions.items():
        icon = "✓" if passed else "✗"
        print(f"  {icon} {name}")

    n_passed = sum(conditions.values())

    if n_passed == 5:
        verdict = "CAUSAL_AND_WORKS"
        print(f"\nVERY STRONG: π IS CAUSAL AND WORKS at λ={best_lambda['lambda']}")
        print(f"  Projection removes Φ → accuracy stable")
        print(f"  Adding Φ → accuracy drops (causal evidence)")
        print(f"  Real Φ > random direction (direction is specific)")
        print(f"  Output diversity preserved (not a fake win)")
        print(f"  → THIS IS PUBLISHABLE. Proceed to full benchmark.")
    elif n_passed >= 3 and best_lambda:
        verdict = "PROJECTION_WORKS"
        print(f"\nSTRONG: π WORKS at λ={best_lambda['lambda']}")
        print(f"  {5-n_passed} condition(s) failed — investigate before publishing")
    elif best_lambda and not real_better_than_random:
        verdict = "DIRECTION_NOT_SPECIAL"
        print("\nNEGATIVE: Random direction works equally well → Φ is not special")
    elif not diversity_ok:
        verdict = "FAKE_WIN"
        print("\nFAKE WIN: Accuracy looks stable but output diversity collapsed")
        print("  → Model became conservative, not smarter. Φ may be wrong.")
    else:
        verdict = "INCONCLUSIVE"
        print("\nINCONCLUSIVE")

    # ─── Generate plots ───
    print(f"\n--- Generating diagnostic plots ---")
    try:
        generate_plots(sweep_real, sweep_random, model_name, phi_layer, output_dir)
    except ImportError:
        print("  matplotlib not available, skipping plots")

    # ─── Save ───
    output = {
        "model": model_name,
        "phi_layer": phi_layer,
        "phi_path": str(phi_path),
        "n_problems": len(problems),
        "verdict": verdict,
        "baseline_accuracy": baseline_acc,
        "baseline_perplexity": baseline_ppl,
        "conditions": {k: v for k, v in conditions.items()},
        "real_better_than_random": real_better_than_random,
        "sign_flip_causal": sign_flip_causal,
        "diversity_ok": diversity_ok,
        "best_lambda": best_lambda["lambda"] if best_lambda else None,
        "sweep_real": sweep_real,
        "sweep_random": sweep_random,
        "sweep_sign_flip": sweep_flip,
    }

    results_path = output_dir / f"e01_results_{model_name.replace('/', '_')}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Does π work?")
    parser.add_argument("--model", type=str, default="pythia-410m")
    parser.add_argument("--phi-path", type=str, required=True,
                        help="Path to .npy file with Φ direction from Experiment 0")
    parser.add_argument("--phi-layer", type=int, required=True,
                        help="Which layer to apply projection at")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=None, help="Custom λ values (default: log-scale sweep)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        phi_path=args.phi_path,
        phi_layer=args.phi_layer,
        lambdas=args.lambdas,
        device=args.device,
        output_dir=args.output_dir,
    )
