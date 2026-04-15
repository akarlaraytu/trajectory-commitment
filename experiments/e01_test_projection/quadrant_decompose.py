"""
Experiment 1.5: Quadrant Decomposition of Φ

PROBLEM:
    Current Φ = mean(wrong) - mean(correct) is a MIXED direction containing:
    - Hallucination signal (confident + wrong)
    - Confidence signal (high entropy vs low entropy)
    - Epistemic signal (knows vs doesn't know)

    Projecting out Φ cuts ALL three → accuracy drops.

SOLUTION:
    Decompose activation space using 2x2 quadrant:

                    Correct         Wrong
    Confident    [knows+right]   [HALLUCINATING]
    Uncertain    [lucky guess]   [knows it doesn't know]

    Extract SEPARATE directions:
    Φ_hall  = mean(confident_wrong) - mean(confident_correct)   → pure hallucination
    Φ_conf  = mean(confident_all) - mean(uncertain_all)         → pure confidence
    Φ_epist = mean(correct_uncertain) - mean(wrong_uncertain)   → epistemic (low-conf only)

    Then ORTHOGONALIZE:
    Φ_hall_clean = Φ_hall - Proj_{Φ_conf}(Φ_hall)  → hallucination WITHOUT confidence
    Φ_epist_clean = Φ_epist - Proj_{Φ_conf}(Φ_epist) - Proj_{Φ_hall_clean}(Φ_epist)

    SELECTIVE PROJECTION:
    π(h) = h - λ·Proj_{Φ_hall_clean}(h)             ← remove hallucination only
              + μ·Proj_{Φ_epist_clean}(h)            ← optionally amplify epistemic

Usage:
    python quadrant_decompose.py [model_name] [layer]
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
    """Same 60 factual prompts used across all experiments."""
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


def normalize(v):
    """Normalize vector to unit length."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def orthogonalize(v, against):
    """Remove component of v along 'against' direction."""
    against_norm = normalize(against)
    return v - np.dot(v, against_norm) * against_norm


def compute_entropy(logits):
    """Compute entropy of output distribution (on CPU)."""
    probs = torch.softmax(logits.float().cpu(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum().item()


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
    target_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    print(f"{'='*60}")
    print(f"QUADRANT DECOMPOSITION OF Φ")
    print(f"{'='*60}")
    print(f"Model: {model_name}, Layer: {target_layer}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    tokenizer = model.tokenizer

    qa_prompts = get_qa_prompts()

    # ─── STEP 1: Collect activations + metadata ───
    print(f"\n--- Step 1: Collecting activations + confidence metadata ---")

    samples = []
    for prompt, correct_answers, category in qa_prompts:
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: f"blocks.{target_layer}.hook_resid_post" in name,
            )

        key = f"blocks.{target_layer}.hook_resid_post"
        act = cache[key][0, -1, :].float().cpu().numpy()

        final_logits = logits[0, -1, :]

        # Correctness
        top1_id = final_logits.float().cpu().argmax().item()
        top1_token = tokenizer.decode([top1_id])
        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        # Confidence metrics
        entropy = compute_entropy(final_logits)
        top1_prob = torch.softmax(final_logits.float().cpu(), dim=-1).max().item()

        # Top-1 logit margin (difference between top-1 and top-2)
        sorted_logits = final_logits.float().cpu().sort(descending=True).values
        logit_margin = (sorted_logits[0] - sorted_logits[1]).item()

        samples.append({
            "prompt": prompt,
            "category": category,
            "activation": act,
            "is_correct": is_correct,
            "entropy": entropy,
            "top1_prob": top1_prob,
            "logit_margin": logit_margin,
            "top1_token": top1_token,
        })

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    n_correct = sum(1 for s in samples if s["is_correct"])
    print(f"  Baseline: {n_correct}/{len(samples)} correct ({n_correct/len(samples):.1%})")

    # ─── STEP 2: Compute SOFT confidence weights ───
    # No hard threshold — every sample gets a continuous confidence weight
    # w_conf(x) = sigmoid(-(entropy - median) / scale)
    #   → high confidence (low entropy) → w ≈ 1
    #   → low confidence (high entropy)  → w ≈ 0
    entropies = np.array([s["entropy"] for s in samples])
    median_entropy = np.median(entropies)
    entropy_std = np.std(entropies) + 1e-8

    for s in samples:
        # Soft confidence weight: sigmoid centered at median, scaled by std
        z = -(s["entropy"] - median_entropy) / entropy_std
        s["w_conf"] = 1.0 / (1.0 + np.exp(-z))  # high confidence → high weight
        s["w_unconf"] = 1.0 - s["w_conf"]        # complement
        # Also keep hard labels for quadrant display
        s["is_confident"] = s["entropy"] < median_entropy

    # ─── STEP 3: Build quadrants (for display) + soft weighted directions ───
    print(f"\n--- Step 2: Building quadrants ---")
    print(f"  Confidence: SOFT weighting (sigmoid centered at median entropy {median_entropy:.2f})")

    quadrants = {
        "confident_correct": [],
        "confident_wrong": [],
        "uncertain_correct": [],
        "uncertain_wrong": [],
    }

    for s in samples:
        if s["is_confident"] and s["is_correct"]:
            quadrants["confident_correct"].append(s)
        elif s["is_confident"] and not s["is_correct"]:
            quadrants["confident_wrong"].append(s)
        elif not s["is_confident"] and s["is_correct"]:
            quadrants["uncertain_correct"].append(s)
        else:
            quadrants["uncertain_wrong"].append(s)

    print(f"\n  Quadrant distribution (hard labels for display):")
    print(f"                    Correct         Wrong")
    print(f"  Confident    {len(quadrants['confident_correct']):>5d}           {len(quadrants['confident_wrong']):>5d}  <- HALLUCINATING")
    print(f"  Uncertain    {len(quadrants['uncertain_correct']):>5d}           {len(quadrants['uncertain_wrong']):>5d}")

    # Check viability
    min_per_quadrant = 3
    for name, group in quadrants.items():
        if len(group) < min_per_quadrant:
            print(f"\n  WARNING: '{name}' has only {len(group)} samples (need {min_per_quadrant})")
            print(f"  Quadrant decomposition may be unstable.")

    # ─── STEP 4: Compute raw directions (SOFT WEIGHTED) ───
    print(f"\n--- Step 3: Computing raw directions (soft weighted) ---")

    all_activations = np.array([s["activation"] for s in samples])

    def soft_weighted_mean(weight_fn):
        """Compute weighted mean of activations using a weight function."""
        weights = np.array([weight_fn(s) for s in samples])
        total = weights.sum()
        if total < 1e-8:
            return np.zeros(all_activations.shape[1])
        return (all_activations * weights[:, None]).sum(axis=0) / total

    # Φ_decision: same as before (all wrong - all correct), no weighting needed
    correct_mask = np.array([s["is_correct"] for s in samples])
    Phi_decision = all_activations[~correct_mask].mean(0) - all_activations[correct_mask].mean(0)

    # Φ_hall: wrong weighted by confidence - correct weighted by confidence
    #   = Σ w_conf(x) · h_wrong / Σw - Σ w_conf(x) · h_correct / Σw
    Phi_hall_raw = soft_weighted_mean(lambda s: s["w_conf"] if not s["is_correct"] else 0) - \
                   soft_weighted_mean(lambda s: s["w_conf"] if s["is_correct"] else 0)

    # Φ_conf: confident activations - uncertain activations (regardless of correctness)
    Phi_conf_raw = soft_weighted_mean(lambda s: s["w_conf"]) - \
                   soft_weighted_mean(lambda s: s["w_unconf"])

    # Φ_epist: among LOW-confidence samples, what separates correct from wrong?
    #   Weighted by uncertainty (w_unconf) to focus on uncertain regime
    epist_correct_weight = sum(s["w_unconf"] for s in samples if s["is_correct"])
    epist_wrong_weight = sum(s["w_unconf"] for s in samples if not s["is_correct"])
    has_epistemic = epist_correct_weight > 1.0 and epist_wrong_weight > 1.0

    if has_epistemic:
        Phi_epist_raw = soft_weighted_mean(lambda s: s["w_unconf"] if s["is_correct"] else 0) - \
                        soft_weighted_mean(lambda s: s["w_unconf"] if not s["is_correct"] else 0)
    else:
        Phi_epist_raw = np.zeros_like(Phi_decision)
        print(f"  WARNING: Not enough uncertain weight for epistemic direction")
        print(f"    correct uncertain weight: {epist_correct_weight:.2f}, wrong: {epist_wrong_weight:.2f}")

    print(f"  Φ_decision  (all wrong - all correct):        norm = {np.linalg.norm(Phi_decision):.4f}")
    print(f"  Φ_hall_raw  (conf_wrong - conf_correct):      norm = {np.linalg.norm(Phi_hall_raw):.4f}")
    print(f"  Φ_conf_raw  (confident - uncertain):          norm = {np.linalg.norm(Phi_conf_raw):.4f}")
    print(f"  Φ_epist_raw (uncert_correct - uncert_wrong):  norm = {np.linalg.norm(Phi_epist_raw):.4f}")

    # ─── STEP 5: Analyze raw direction overlaps ───
    print(f"\n--- Step 4: Direction overlaps (cosine similarity) ---")

    dirs = {
        "Φ_decision": normalize(Phi_decision),
        "Φ_hall": normalize(Phi_hall_raw),
        "Φ_conf": normalize(Phi_conf_raw),
    }
    if has_epistemic:
        dirs["Φ_epist"] = normalize(Phi_epist_raw)

    names = list(dirs.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            cos = np.dot(dirs[n1], dirs[n2])
            print(f"  cos({n1}, {n2}) = {cos:+.4f}")

    # ─── STEP 6: Orthogonalize ───
    print(f"\n--- Step 5: Orthogonal decomposition ---")

    # Step 1: Keep Φ_conf as-is (reference axis)
    Phi_conf = normalize(Phi_conf_raw)

    # Step 2: Remove confidence component from hallucination
    Phi_hall_clean = orthogonalize(Phi_hall_raw, Phi_conf_raw)
    Phi_hall_clean = normalize(Phi_hall_clean)

    # Step 3: Remove both confidence AND hallucination from epistemic
    if has_epistemic:
        Phi_epist_clean = orthogonalize(Phi_epist_raw, Phi_conf_raw)
        Phi_epist_clean = orthogonalize(Phi_epist_clean, Phi_hall_clean)
        Phi_epist_clean = normalize(Phi_epist_clean)
    else:
        Phi_epist_clean = np.zeros_like(Phi_conf)

    # Verify orthogonality
    print(f"  After orthogonalization:")
    print(f"  cos(Φ_hall_clean, Φ_conf)        = {np.dot(Phi_hall_clean, Phi_conf):+.6f}  (should be ~0)")
    if has_epistemic:
        print(f"  cos(Φ_epist_clean, Φ_conf)       = {np.dot(Phi_epist_clean, Phi_conf):+.6f}  (should be ~0)")
        print(f"  cos(Φ_epist_clean, Φ_hall_clean)  = {np.dot(Phi_epist_clean, Phi_hall_clean):+.6f}  (should be ~0)")

    # How much of original Φ_decision was confidence?
    conf_component = np.dot(normalize(Phi_decision), Phi_conf)
    hall_component = np.dot(normalize(Phi_decision), Phi_hall_clean)
    if has_epistemic:
        epist_component = np.dot(normalize(Phi_decision), Phi_epist_clean)
    else:
        epist_component = 0.0

    print(f"\n  Original Φ_decision decomposition:")
    print(f"    Confidence component:  {conf_component:+.4f}  ({abs(conf_component)*100:.1f}%)")
    print(f"    Hallucination component: {hall_component:+.4f}  ({abs(hall_component)*100:.1f}%)")
    if has_epistemic:
        print(f"    Epistemic component:   {epist_component:+.4f}  ({abs(epist_component)*100:.1f}%)")
    residual = 1.0 - conf_component**2 - hall_component**2 - epist_component**2
    print(f"    Residual (unexplained): {residual:.4f}  ({abs(residual)*100:.1f}%)")

    # ─── STEP 7: Test each direction's classification power ───
    print(f"\n--- Step 6: Per-direction classification power ---")

    all_acts = np.array([s["activation"] for s in samples])
    all_correct = np.array([s["is_correct"] for s in samples])
    all_confident = np.array([s["is_confident"] for s in samples])

    for name, direction in [
        ("Φ_decision (original)", normalize(Phi_decision)),
        ("Φ_hall_clean (pure hallucination)", Phi_hall_clean),
        ("Φ_conf (pure confidence)", Phi_conf),
    ] + ([("Φ_epist_clean (pure epistemic)", Phi_epist_clean)] if has_epistemic else []):
        # Project all activations onto this direction
        projections = all_acts @ direction

        # How well does this direction separate correct/wrong?
        proj_correct = projections[all_correct]
        proj_wrong = projections[~all_correct]

        if len(proj_correct) > 0 and len(proj_wrong) > 0:
            # Cohen's d effect size
            pooled_std = np.sqrt((proj_correct.std()**2 + proj_wrong.std()**2) / 2)
            cohens_d = (proj_wrong.mean() - proj_correct.mean()) / max(pooled_std, 1e-8)

            # AUC (using projection as classifier)
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(~all_correct, projections)
            except:
                auc = 0.5

            print(f"  {name}:")
            print(f"    Correct mean: {proj_correct.mean():+.4f}, Wrong mean: {proj_wrong.mean():+.4f}")
            print(f"    Cohen's d: {cohens_d:+.4f}, AUC: {auc:.3f}")

        # Also check confidence separation
        proj_conf = projections[all_confident]
        proj_unconf = projections[~all_confident]
        if len(proj_conf) > 0 and len(proj_unconf) > 0:
            pooled_std_c = np.sqrt((proj_conf.std()**2 + proj_unconf.std()**2) / 2)
            cohens_d_c = (proj_conf.mean() - proj_unconf.mean()) / max(pooled_std_c, 1e-8)
            print(f"    Confidence d: {cohens_d_c:+.4f}  (should be ~0 for clean directions)")

    # ─── STEP 8: Save directions ───
    print(f"\n--- Step 7: Saving directions ---")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    safe_name = model_name.replace("/", "_")

    np.save(results_dir / f"phi_hall_clean_layer{target_layer}_{safe_name}.npy",
            Phi_hall_clean.astype(np.float32))
    np.save(results_dir / f"phi_conf_layer{target_layer}_{safe_name}.npy",
            Phi_conf.astype(np.float32))
    if has_epistemic:
        np.save(results_dir / f"phi_epist_clean_layer{target_layer}_{safe_name}.npy",
                Phi_epist_clean.astype(np.float32))

    print(f"  Saved: phi_hall_clean_layer{target_layer}_{safe_name}.npy")
    print(f"  Saved: phi_conf_layer{target_layer}_{safe_name}.npy")
    if has_epistemic:
        print(f"  Saved: phi_epist_clean_layer{target_layer}_{safe_name}.npy")

    # ─── STEP 9: Quick projection test with clean directions ───
    print(f"\n--- Step 8: Quick projection test (Φ_hall_clean vs Φ_decision) ---")
    print(f"  Testing: does clean hallucination direction preserve accuracy better?\n")

    from test_projection import TLoTProjectionHook, evaluate_model, generate_factual_qa

    problems = generate_factual_qa()
    test_lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    results_table = []

    for direction_name, direction in [
        ("Φ_decision (original)", normalize(Phi_decision)),
        ("Φ_hall_clean", Phi_hall_clean),
    ]:
        print(f"  Direction: {direction_name}")
        for lam in test_lambdas:
            hook = TLoTProjectionHook(phi_direction=direction, lam=lam, preserve_norm=True)
            hook_name = f"blocks.{target_layer}.hook_resid_post"

            model.reset_hooks()
            if lam > 0:
                model.add_hook(hook_name, hook.hook_fn)

            result = evaluate_model(model, problems)
            diag = hook.get_diagnostics()
            red = diag.get("reduction_ratio", 0)

            print(f"    λ={lam:.1f}  acc={result.accuracy:.3f}  ppl={result.mean_perplexity:.1f}  red={red:.3f}")
            results_table.append({
                "direction": direction_name,
                "lambda": lam,
                "accuracy": result.accuracy,
                "perplexity": result.mean_perplexity,
                "reduction": red,
            })

            model.reset_hooks()
            hook.reset_diagnostics()

        print()

    # ─── STEP 10: Selective projection test ───
    # π(h) = h - λ·Proj_{Φ_hall}(h) + μ·Proj_{Φ_epist}(h)
    if has_epistemic:
        print(f"--- Step 9: Selective Projection (remove hall + amplify epistemic) ---\n")

        for lam in [0.3, 0.5, 0.7]:
            for mu in [0.0, 0.1, 0.3]:
                # Custom dual hook
                phi_h = torch.tensor(Phi_hall_clean, dtype=torch.float32)
                phi_h = F.normalize(phi_h, dim=0)
                phi_e = torch.tensor(Phi_epist_clean, dtype=torch.float32)
                phi_e = F.normalize(phi_e, dim=0)

                def make_dual_hook(phi_hall_t, phi_epist_t, lam_val, mu_val):
                    def hook_fn(value, hook):
                        d = value.device
                        dt = value.dtype
                        ph = phi_hall_t.to(device=d, dtype=dt)
                        pe = phi_epist_t.to(device=d, dtype=dt)

                        original_norm = value.norm(dim=-1, keepdim=True)

                        # Remove hallucination
                        proj_h = (value @ ph).unsqueeze(-1) * ph.unsqueeze(0).unsqueeze(0)
                        value = value - lam_val * proj_h

                        # Amplify epistemic
                        if mu_val > 0:
                            proj_e = (value @ pe).unsqueeze(-1) * pe.unsqueeze(0).unsqueeze(0)
                            value = value + mu_val * proj_e

                        # Preserve norm
                        safe_norm = value.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        value = value * (original_norm / safe_norm)
                        return value
                    return hook_fn

                hook_fn = make_dual_hook(phi_h, phi_e, lam, mu)
                hook_name = f"blocks.{target_layer}.hook_resid_post"

                model.reset_hooks()
                model.add_hook(hook_name, hook_fn)

                result = evaluate_model(model, problems)
                print(f"  λ_hall={lam:.1f}, μ_epist={mu:.1f}  →  acc={result.accuracy:.3f}  ppl={result.mean_perplexity:.1f}")

                model.reset_hooks()

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    print(f"\nOriginal Φ was {abs(conf_component)*100:.0f}% confidence, "
          f"{abs(hall_component)*100:.0f}% hallucination"
          f"{f', {abs(epist_component)*100:.0f}% epistemic' if has_epistemic else ''}")

    # Compare: at what λ does accuracy first drop below baseline-5% for each direction?
    baseline_acc = next(r["accuracy"] for r in results_table if r["lambda"] == 0)
    threshold = baseline_acc - 0.05

    for dir_name in ["Φ_decision (original)", "Φ_hall_clean"]:
        dir_results = [r for r in results_table if r["direction"] == dir_name]
        first_drop = next(
            (r["lambda"] for r in dir_results if r["accuracy"] < threshold and r["lambda"] > 0),
            None)
        max_safe_red = max(
            (r["reduction"] for r in dir_results if r["accuracy"] >= threshold),
            default=0)
        print(f"\n  {dir_name}:")
        print(f"    First acc drop below {threshold:.3f}: λ={first_drop}")
        print(f"    Max safe reduction: {max_safe_red:.3f}")

    # Save full results
    output = {
        "model": model_name,
        "layer": target_layer,
        "n_samples": len(samples),
        "baseline_accuracy": n_correct / len(samples),
        "confidence_threshold_entropy": float(median_entropy),
        "quadrants": {k: len(v) for k, v in quadrants.items()},
        "direction_overlaps": {
            "cos(decision, hall)": float(np.dot(normalize(Phi_decision), normalize(Phi_hall_raw))),
            "cos(decision, conf)": float(np.dot(normalize(Phi_decision), normalize(Phi_conf_raw))),
            "cos(hall, conf)": float(np.dot(normalize(Phi_hall_raw), normalize(Phi_conf_raw))),
        },
        "decision_decomposition": {
            "confidence_pct": float(abs(conf_component) * 100),
            "hallucination_pct": float(abs(hall_component) * 100),
            "epistemic_pct": float(abs(epist_component) * 100) if has_epistemic else None,
            "residual_pct": float(abs(residual) * 100),
        },
        "projection_comparison": results_table,
    }

    out_path = results_dir / f"quadrant_decomposition_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
