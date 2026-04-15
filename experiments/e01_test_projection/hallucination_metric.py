"""
Experiment 1.6: Does projection ACTUALLY reduce hallucination?

CRITICAL QUESTION (reviewer-killer):
    "Reduction ratio going up just means you removed some geometric component.
     Did ACTUAL hallucination decrease?"

HALLUCINATION DEFINITION:
    Hallucination = model is WRONG but CONFIDENT
    Not just wrong — confidently wrong.

METRICS:
    1. hallucination_rate = n(confident_wrong) / n(total)
    2. accuracy = n(correct) / n(total)
    3. abstention_rate = n(uncertain_wrong) / n(total)  (knows it doesn't know)
    4. calibration = correlation(confidence, correctness)

THE GOLDEN TEST:
    Before projection:  accuracy=X, hallucination_rate=Y
    After projection:   accuracy≈X, hallucination_rate<Y

    If hallucination_rate drops while accuracy holds → REAL EFFECT
    If only accuracy drops → you're just breaking the model

Usage:
    python hallucination_metric.py [model_name] [layer]
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'


def get_qa_prompts():
    """Same 60 factual prompts."""
    return [
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


def evaluate_with_hallucination_metrics(model, prompts, tokenizer):
    """
    Evaluate each prompt and classify into quadrants:
      confident_correct  = knows and right
      confident_wrong    = HALLUCINATING
      uncertain_correct  = lucky/weak knowledge
      uncertain_wrong    = knows it doesn't know

    Returns per-sample details and aggregate metrics.
    """
    results = []

    for prompt, correct_answers, category in prompts:
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            logits = model(tokens)

        final_logits = logits[0, -1, :]
        probs = torch.softmax(final_logits.float().cpu(), dim=-1)

        top1_id = probs.argmax().item()
        top1_prob = probs[top1_id].item()
        top1_token = tokenizer.decode([top1_id])

        # Entropy
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()

        # Logit margin (top1 - top2)
        sorted_probs = probs.sort(descending=True).values
        margin = (sorted_probs[0] - sorted_probs[1]).item()

        # Correctness
        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        results.append({
            "prompt": prompt,
            "category": category,
            "top1_token": top1_token,
            "expected": correct_answers[0],
            "is_correct": is_correct,
            "top1_prob": top1_prob,
            "entropy": entropy,
            "margin": margin,
        })

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def compute_metrics(results):
    """
    Compute aggregate metrics from per-sample results.
    Uses soft confidence (top1_prob) with adaptive threshold.
    """
    n = len(results)

    # Adaptive confidence threshold: median top1_prob
    probs = [r["top1_prob"] for r in results]
    conf_threshold = np.median(probs)

    # Classify into quadrants
    confident_correct = 0
    confident_wrong = 0    # HALLUCINATION
    uncertain_correct = 0
    uncertain_wrong = 0

    for r in results:
        is_conf = r["top1_prob"] > conf_threshold
        if is_conf and r["is_correct"]:
            confident_correct += 1
            r["quadrant"] = "confident_correct"
        elif is_conf and not r["is_correct"]:
            confident_wrong += 1
            r["quadrant"] = "confident_wrong"  # HALLUCINATION
        elif not is_conf and r["is_correct"]:
            uncertain_correct += 1
            r["quadrant"] = "uncertain_correct"
        else:
            uncertain_wrong += 1
            r["quadrant"] = "uncertain_wrong"

    accuracy = sum(1 for r in results if r["is_correct"]) / n
    hallucination_rate = confident_wrong / n
    abstention_rate = uncertain_wrong / n  # knows it doesn't know
    confident_accuracy = confident_correct / max(confident_correct + confident_wrong, 1)

    # Calibration: correlation between confidence and correctness
    from scipy.stats import pearsonr
    correctness = [1.0 if r["is_correct"] else 0.0 for r in results]
    try:
        calibration_r, calibration_p = pearsonr(probs, correctness)
    except:
        calibration_r, calibration_p = 0.0, 1.0

    return {
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "abstention_rate": abstention_rate,
        "confident_accuracy": confident_accuracy,
        "calibration_r": calibration_r,
        "calibration_p": calibration_p,
        "confidence_threshold": conf_threshold,
        "quadrants": {
            "confident_correct": confident_correct,
            "confident_wrong": confident_wrong,
            "uncertain_correct": uncertain_correct,
            "uncertain_wrong": uncertain_wrong,
        },
        "n": n,
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
    target_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    print(f"{'='*60}")
    print(f"HALLUCINATION METRIC TEST")
    print(f"Does projection ACTUALLY reduce hallucination?")
    print(f"{'='*60}")
    print(f"Model: {model_name}, Layer: {target_layer}\n")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    tokenizer = model.tokenizer

    prompts = get_qa_prompts()

    # Load directions
    results_dir = Path(__file__).parent / "results"
    safe_name = model_name.replace("/", "_")

    phi_hall_path = results_dir / f"phi_hall_clean_layer{target_layer}_{safe_name}.npy"
    phi_decision_path = results_dir / f".." / ".." / "experiments" / "e00_find_phi" / "results" / f"phi_direction_layer{target_layer}_{safe_name}.npy"

    # Try to load both directions
    directions = {}

    if phi_hall_path.exists():
        directions["Φ_hall_clean"] = np.load(phi_hall_path)
        print(f"  Loaded Φ_hall_clean from {phi_hall_path}")

    # Also load original Φ_decision from e00
    e00_path = Path(__file__).parent.parent / "e00_find_phi" / "results" / f"phi_direction_layer{target_layer}_{safe_name}.npy"
    if e00_path.exists():
        directions["Φ_decision"] = np.load(e00_path)
        print(f"  Loaded Φ_decision from {e00_path}")

    # Random direction for control
    rng = np.random.RandomState(42)
    d_model = list(directions.values())[0].shape[0] if directions else 1024
    random_dir = rng.randn(d_model).astype(np.float32)
    random_dir = random_dir / np.linalg.norm(random_dir)
    directions["Random"] = random_dir

    from test_projection import TLoTProjectionHook

    # ─── TEST: Measure hallucination metrics at each λ for each direction ───
    test_lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    all_results = {}

    for dir_name, direction in directions.items():
        print(f"\n{'─'*50}")
        print(f"  Direction: {dir_name}")
        print(f"{'─'*50}")

        dir_results = []

        for lam in test_lambdas:
            hook = TLoTProjectionHook(phi_direction=direction, lam=lam, preserve_norm=True)
            hook_name = f"blocks.{target_layer}.hook_resid_post"

            model.reset_hooks()
            if lam > 0:
                model.add_hook(hook_name, hook.hook_fn)

            per_sample = evaluate_with_hallucination_metrics(model, prompts, tokenizer)
            metrics = compute_metrics(per_sample)
            diag = hook.get_diagnostics()

            q = metrics["quadrants"]
            print(f"  λ={lam:.1f}  "
                  f"acc={metrics['accuracy']:.3f}  "
                  f"hall={metrics['hallucination_rate']:.3f}  "
                  f"abst={metrics['abstention_rate']:.3f}  "
                  f"cal_r={metrics['calibration_r']:+.3f}  "
                  f"[CC={q['confident_correct']} CW={q['confident_wrong']} "
                  f"UC={q['uncertain_correct']} UW={q['uncertain_wrong']}]")

            dir_results.append({
                "lambda": lam,
                "metrics": metrics,
                "diagnostics": diag,
                "per_sample": [{k: v for k, v in s.items() if k != "activation"}
                               for s in per_sample],
            })

            model.reset_hooks()
            hook.reset_diagnostics()

        all_results[dir_name] = dir_results

    # ─── ANALYSIS: Per-sample tracking ───
    # Track which specific prompts CHANGE from hallucinating to not (or vice versa)
    print(f"\n{'='*60}")
    print(f"PER-SAMPLE HALLUCINATION TRACKING")
    print(f"{'='*60}")

    if "Φ_hall_clean" in all_results:
        baseline_samples = all_results["Φ_hall_clean"][0]["per_sample"]  # λ=0
        best_lambda_idx = None
        best_hall_reduction = 0

        for i, r in enumerate(all_results["Φ_hall_clean"]):
            if r["lambda"] == 0:
                continue
            baseline_hall = all_results["Φ_hall_clean"][0]["metrics"]["hallucination_rate"]
            current_hall = r["metrics"]["hallucination_rate"]
            reduction = baseline_hall - current_hall
            acc_loss = all_results["Φ_hall_clean"][0]["metrics"]["accuracy"] - r["metrics"]["accuracy"]
            if reduction > best_hall_reduction and acc_loss <= 0.05:
                best_hall_reduction = reduction
                best_lambda_idx = i

        if best_lambda_idx is not None:
            best_r = all_results["Φ_hall_clean"][best_lambda_idx]
            best_samples = best_r["per_sample"]
            best_lam = best_r["lambda"]

            print(f"\n  Best λ for hallucination reduction: {best_lam}")
            print(f"\n  Changes from baseline (λ=0) → λ={best_lam}:")

            changes = {"fixed": [], "broken": [], "stable_hall": [], "stable_correct": []}
            for i, (base, proj) in enumerate(zip(baseline_samples, best_samples)):
                was_hall = base.get("quadrant") == "confident_wrong"
                now_hall = proj.get("quadrant") == "confident_wrong"
                was_correct = base["is_correct"]
                now_correct = proj["is_correct"]

                if was_hall and not now_hall:
                    changes["fixed"].append(base["prompt"][:60])
                elif not was_hall and now_hall:
                    changes["broken"].append(base["prompt"][:60])
                elif was_hall and now_hall:
                    changes["stable_hall"].append(base["prompt"][:60])
                elif was_correct and now_correct:
                    changes["stable_correct"].append(None)

            print(f"\n  FIXED hallucinations ({len(changes['fixed'])}):")
            for p in changes["fixed"]:
                print(f"    ✓ {p}")
            print(f"\n  NEW hallucinations ({len(changes['broken'])}):")
            for p in changes["broken"]:
                print(f"    ✗ {p}")
            print(f"\n  PERSISTENT hallucinations ({len(changes['stable_hall'])}):")
            for p in changes["stable_hall"]:
                print(f"    ! {p}")
            print(f"\n  Stable correct: {len(changes['stable_correct'])}")

    # ─── SUMMARY TABLE ───
    print(f"\n{'='*60}")
    print(f"SUMMARY: HALLUCINATION RATE COMPARISON")
    print(f"{'='*60}")

    print(f"\n  {'Direction':<20s}  {'λ':>5s}  {'Acc':>7s}  {'Hall%':>7s}  {'Δ Hall':>7s}  {'Abst%':>7s}  {'Cal r':>7s}")
    print(f"  " + "─" * 65)

    for dir_name, dir_results in all_results.items():
        baseline_hall = dir_results[0]["metrics"]["hallucination_rate"]
        for r in dir_results:
            m = r["metrics"]
            delta_hall = m["hallucination_rate"] - baseline_hall
            marker = ""
            if delta_hall < -0.03 and m["accuracy"] >= dir_results[0]["metrics"]["accuracy"] - 0.05:
                marker = " ← WIN"
            elif delta_hall > 0.03:
                marker = " ← WORSE"

            print(f"  {dir_name:<20s}  {r['lambda']:>5.1f}  "
                  f"{m['accuracy']:>7.3f}  "
                  f"{m['hallucination_rate']:>7.3f}  "
                  f"{delta_hall:>+7.3f}  "
                  f"{m['abstention_rate']:>7.3f}  "
                  f"{m['calibration_r']:>+7.3f}{marker}")
        print()

    # ─── VERDICT ───
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    if "Φ_hall_clean" in all_results:
        baseline = all_results["Φ_hall_clean"][0]["metrics"]
        best_result = None
        best_score = 0

        for r in all_results["Φ_hall_clean"]:
            if r["lambda"] == 0:
                continue
            m = r["metrics"]
            hall_drop = baseline["hallucination_rate"] - m["hallucination_rate"]
            acc_loss = baseline["accuracy"] - m["accuracy"]
            # Score: hall reduction minus accuracy penalty
            score = hall_drop - 2 * max(acc_loss, 0)
            if score > best_score:
                best_score = score
                best_result = r

        if best_result:
            m = best_result["metrics"]
            hall_drop = baseline["hallucination_rate"] - m["hallucination_rate"]
            acc_loss = baseline["accuracy"] - m["accuracy"]

            print(f"\n  Best result: λ={best_result['lambda']}")
            print(f"    Accuracy:         {baseline['accuracy']:.3f} → {m['accuracy']:.3f}  (Δ={-acc_loss:+.3f})")
            print(f"    Hallucination:    {baseline['hallucination_rate']:.3f} → {m['hallucination_rate']:.3f}  (Δ={-hall_drop:+.3f})")
            print(f"    Abstention:       {baseline['abstention_rate']:.3f} → {m['abstention_rate']:.3f}")
            print(f"    Calibration:      {baseline['calibration_r']:+.3f} → {m['calibration_r']:+.3f}")

            if hall_drop > 0.03 and acc_loss <= 0.05:
                print(f"\n  🔥 HALLUCINATION ACTUALLY DECREASED")
                print(f"     accuracy ≈ stable, hallucination ↓ {hall_drop:.1%}")
                print(f"     THIS IS THE RESULT WE NEED.")
            elif hall_drop > 0 and acc_loss <= 0.02:
                print(f"\n  ⚡ Hallucination decreased slightly ({hall_drop:.1%})")
                print(f"     Signal is weak — may need larger model or more data.")
            elif hall_drop <= 0:
                print(f"\n  ⚠️  Hallucination did NOT decrease.")
                print(f"     Projection is geometric denoising, not hallucination removal.")
                print(f"     Need: better Φ, subspace approach, or larger model.")
            else:
                print(f"\n  ⚠️  Hallucination decreased but accuracy dropped too much.")
                print(f"     Need finer λ tuning or better direction.")
        else:
            print(f"\n  No λ improved hallucination without hurting accuracy.")

    # ─── SAVE ───
    output = {
        "model": model_name,
        "layer": target_layer,
        "results": {},
    }
    for dir_name, dir_results in all_results.items():
        output["results"][dir_name] = [
            {
                "lambda": r["lambda"],
                "metrics": r["metrics"],
                "diagnostics": r["diagnostics"],
            }
            for r in dir_results
        ]

    out_path = results_dir / f"hallucination_metrics_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # ─── PLOT ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Hallucination Metrics — {model_name}, Layer {target_layer}",
                     fontsize=14, fontweight="bold")

        colors = {"Φ_hall_clean": "#2ecc71", "Φ_decision": "#3498db", "Random": "#e74c3c"}

        for dir_name, dir_results in all_results.items():
            lambdas = [r["lambda"] for r in dir_results]
            acc = [r["metrics"]["accuracy"] for r in dir_results]
            hall = [r["metrics"]["hallucination_rate"] for r in dir_results]
            cal = [r["metrics"]["calibration_r"] for r in dir_results]
            c = colors.get(dir_name, "#95a5a6")

            axes[0].plot(lambdas, acc, "-o", color=c, label=dir_name, linewidth=2, markersize=6)
            axes[1].plot(lambdas, hall, "-o", color=c, label=dir_name, linewidth=2, markersize=6)
            axes[2].plot(lambdas, cal, "-o", color=c, label=dir_name, linewidth=2, markersize=6)

        axes[0].set_title("Accuracy vs λ")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        axes[1].set_title("Hallucination Rate vs λ (LOWER = BETTER)")
        axes[1].set_ylabel("Hallucination Rate (confident + wrong)")
        axes[1].legend()

        axes[2].set_title("Calibration vs λ (HIGHER = BETTER)")
        axes[2].set_ylabel("Calibration (r)")
        axes[2].legend()

        for ax in axes:
            ax.set_xlabel("λ (projection strength)")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        figures_dir = results_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        plt.savefig(figures_dir / "04_hallucination_metrics.png", dpi=150)
        plt.close()
        print(f"Saved: 04_hallucination_metrics.png")

    except ImportError:
        print("  matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
