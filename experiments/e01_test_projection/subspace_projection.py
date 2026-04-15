"""
Experiment 1.7: Conditional Subspace Projection

KEY INSIGHT:
    Hallucination is not a direction — it is a REGION in activation space.
    Single vector projection cuts through knowledge AND hallucination.
    Subspace projection with energy thresholding only cuts when needed.

METHOD:
    1. Collect confident_wrong (hallucinating) activations
    2. PCA → top-k hallucination subspace S
    3. For each h during inference:
       energy = ||Proj_S(h)||
       if energy > τ:  h' = h - λ·Proj_S(h)    ← conditional cut
       else:           h' = h                     ← leave alone

    This preserves correct answers (low energy in S) while
    cutting hallucinating ones (high energy in S).

Usage:
    python subspace_projection.py [model_name] [layer]
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


# ─── Conditional Subspace Projection Hook ───────────────────────

class SubspaceProjectionHook:
    """
    Conditional subspace projection:
    - Projects onto k-dimensional hallucination subspace S
    - Only intervenes when energy in S exceeds threshold τ
    - Preserves norm after projection

    π(h) = h - λ·Proj_S(h)  if ||Proj_S(h)|| > τ
          = h                 otherwise
    """

    def __init__(self, basis_vectors, lam=1.0, tau=0.0, preserve_norm=True):
        """
        basis_vectors: (k, d_model) — orthonormal basis of hallucination subspace
        lam: projection strength
        tau: energy threshold (0 = always project, >0 = conditional)
        """
        self.basis = torch.tensor(basis_vectors, dtype=torch.float32)  # (k, d)
        self.lam = lam
        self.tau = tau
        self.preserve_norm = preserve_norm

        # Diagnostics
        self.energies_before = []
        self.n_projected = 0
        self.n_skipped = 0

    def hook_fn(self, value, hook):
        device = value.device
        dtype = value.dtype
        basis = self.basis.to(device=device, dtype=dtype)  # (k, d)

        original_norm = value.norm(dim=-1, keepdim=True)

        # Projection onto subspace: Proj_S(h) = Σ_i (h·v_i)v_i = V^T V h
        # value: (batch, seq, d), basis: (k, d)
        # coeffs: (batch, seq, k) = value @ basis.T
        coeffs = torch.matmul(value, basis.T)  # (batch, seq, k)
        # projection: (batch, seq, d) = coeffs @ basis
        projection = torch.matmul(coeffs, basis)  # (batch, seq, d)

        # Energy = ||Proj_S(h)|| at last token position
        energy_last = projection[:, -1, :].norm(dim=-1)  # (batch,)
        self.energies_before.extend(energy_last.cpu().tolist())

        if self.tau > 0:
            # Conditional: only project where energy > tau
            mask = (energy_last > self.tau).float()  # (batch,)
            self.n_projected += int(mask.sum().item())
            self.n_skipped += int((1 - mask).sum().item())

            # Expand mask to (batch, 1, 1) for broadcasting
            # Apply to ALL positions (if last token is high energy, whole sequence gets projected)
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            value_safe = value - self.lam * projection * mask
        else:
            # Unconditional: always project
            value_safe = value - self.lam * projection
            self.n_projected += value.shape[0]

        if self.preserve_norm:
            safe_norm = value_safe.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            value_safe = value_safe * (original_norm / safe_norm)

        return value_safe

    def get_diagnostics(self):
        if not self.energies_before:
            return {"n_calls": 0}
        return {
            "n_calls": len(self.energies_before),
            "mean_energy": float(np.mean(self.energies_before)),
            "std_energy": float(np.std(self.energies_before)),
            "max_energy": float(np.max(self.energies_before)),
            "min_energy": float(np.min(self.energies_before)),
            "n_projected": self.n_projected,
            "n_skipped": self.n_skipped,
            "projection_rate": self.n_projected / max(self.n_projected + self.n_skipped, 1),
        }

    def reset(self):
        self.energies_before.clear()
        self.n_projected = 0
        self.n_skipped = 0


# ─── Hallucination evaluation ───────────────────────────────────

def evaluate_hallucination(model, prompts, tokenizer):
    """Evaluate and return per-sample hallucination metrics."""
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

        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        results.append({
            "prompt": prompt,
            "category": category,
            "is_correct": is_correct,
            "top1_prob": top1_prob,
            "entropy": entropy,
            "top1_token": top1_token,
        })

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def compute_hall_metrics(results):
    """Compute hallucination metrics with adaptive confidence threshold."""
    n = len(results)
    probs = [r["top1_prob"] for r in results]
    conf_threshold = np.median(probs)

    cc = cw = uc = uw = 0
    for r in results:
        is_conf = r["top1_prob"] > conf_threshold
        if is_conf and r["is_correct"]:
            cc += 1
        elif is_conf and not r["is_correct"]:
            cw += 1
        elif not is_conf and r["is_correct"]:
            uc += 1
        else:
            uw += 1

    from scipy.stats import pearsonr
    correctness = [1.0 if r["is_correct"] else 0.0 for r in results]
    try:
        cal_r, _ = pearsonr(probs, correctness)
    except:
        cal_r = 0.0

    return {
        "accuracy": sum(1 for r in results if r["is_correct"]) / n,
        "hallucination_rate": cw / n,
        "abstention_rate": uw / n,
        "confident_accuracy": cc / max(cc + cw, 1),
        "calibration_r": cal_r,
        "quadrants": {"CC": cc, "CW": cw, "UC": uc, "UW": uw},
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
    target_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    print(f"{'='*60}")
    print(f"CONDITIONAL SUBSPACE PROJECTION")
    print(f"'Hallucination is not a direction — it is a region'")
    print(f"{'='*60}")
    print(f"Model: {model_name}, Layer: {target_layer}\n")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    tokenizer = model.tokenizer

    prompts = get_qa_prompts()

    # ─── STEP 1: Collect activations with quadrant labels ───
    print(f"--- Step 1: Collecting activations ---")

    samples = []
    for prompt, correct_answers, category in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: f"blocks.{target_layer}.hook_resid_post" in name)

        key = f"blocks.{target_layer}.hook_resid_post"
        act = cache[key][0, -1, :].float().cpu().numpy()

        final_logits = logits[0, -1, :]
        probs = torch.softmax(final_logits.float().cpu(), dim=-1)
        top1_id = probs.argmax().item()
        top1_prob = probs[top1_id].item()
        top1_token = tokenizer.decode([top1_id])
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        samples.append({
            "activation": act,
            "is_correct": is_correct,
            "top1_prob": top1_prob,
            "entropy": entropy,
            "prompt": prompt,
        })

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Classify into quadrants using median confidence
    probs_all = [s["top1_prob"] for s in samples]
    conf_threshold = np.median(probs_all)

    for s in samples:
        is_conf = s["top1_prob"] > conf_threshold
        if is_conf and s["is_correct"]:
            s["quadrant"] = "CC"
        elif is_conf and not s["is_correct"]:
            s["quadrant"] = "CW"  # HALLUCINATING
        elif not is_conf and s["is_correct"]:
            s["quadrant"] = "UC"
        else:
            s["quadrant"] = "UW"

    n_cc = sum(1 for s in samples if s["quadrant"] == "CC")
    n_cw = sum(1 for s in samples if s["quadrant"] == "CW")
    n_uc = sum(1 for s in samples if s["quadrant"] == "UC")
    n_uw = sum(1 for s in samples if s["quadrant"] == "UW")

    print(f"  CC={n_cc}, CW={n_cw} (hallucinating), UC={n_uc}, UW={n_uw}")
    print(f"  Confidence threshold: {conf_threshold:.4f}")

    # ─── STEP 2: Build hallucination subspace via PCA ───
    print(f"\n--- Step 2: PCA on hallucination activations ---")

    # Strategy: PCA on (confident_wrong - mean_confident_correct)
    # This centers the hallucination space relative to "knowing"
    cw_acts = np.array([s["activation"] for s in samples if s["quadrant"] == "CW"])
    cc_acts = np.array([s["activation"] for s in samples if s["quadrant"] == "CC"])

    if len(cw_acts) < 3:
        print(f"  ERROR: Only {len(cw_acts)} confident_wrong samples. Not enough for PCA.")
        print(f"  Need at least 3 hallucinating samples.")
        sys.exit(1)

    # Center relative to confident_correct mean
    cc_mean = cc_acts.mean(axis=0)
    cw_centered = cw_acts - cc_mean

    # Also include confident_correct (centered) for contrast
    cc_centered = cc_acts - cc_mean

    # PCA on the centered hallucination activations
    from sklearn.decomposition import PCA

    max_k = min(len(cw_acts) - 1, 10)  # Can't have more components than samples-1
    pca = PCA(n_components=max_k)
    pca.fit(cw_centered)

    print(f"  PCA on {len(cw_acts)} confident_wrong activations (centered on CC mean)")
    print(f"  Explained variance ratios:")
    cumulative = 0
    for i, ev in enumerate(pca.explained_variance_ratio_):
        cumulative += ev
        print(f"    PC{i+1}: {ev:.4f}  (cumulative: {cumulative:.4f})")

    # ─── STEP 3: Energy analysis — do correct/wrong separate in subspace? ───
    print(f"\n--- Step 3: Energy analysis in subspace ---")

    all_acts = np.array([s["activation"] for s in samples])
    all_centered = all_acts - cc_mean

    for k in [1, 2, 3, 5, max_k]:
        if k > max_k:
            continue
        basis = pca.components_[:k]  # (k, d)
        projections = all_centered @ basis.T  # (n, k)
        energies = np.linalg.norm(projections, axis=1)  # (n,)

        # Energy by quadrant
        energy_cc = [energies[i] for i, s in enumerate(samples) if s["quadrant"] == "CC"]
        energy_cw = [energies[i] for i, s in enumerate(samples) if s["quadrant"] == "CW"]
        energy_uc = [energies[i] for i, s in enumerate(samples) if s["quadrant"] == "UC"]
        energy_uw = [energies[i] for i, s in enumerate(samples) if s["quadrant"] == "UW"]

        print(f"\n  k={k} subspace:")
        print(f"    CC (correct+confident):   mean={np.mean(energy_cc):.3f} ± {np.std(energy_cc):.3f}")
        print(f"    CW (HALLUCINATING):       mean={np.mean(energy_cw):.3f} ± {np.std(energy_cw):.3f}")
        print(f"    UC (correct+uncertain):   mean={np.mean(energy_uc):.3f} ± {np.std(energy_uc):.3f}")
        print(f"    UW (wrong+uncertain):     mean={np.mean(energy_uw):.3f} ± {np.std(energy_uw):.3f}")

        # Separation: CW should have HIGH energy, CC should have LOW
        if len(energy_cw) > 0 and len(energy_cc) > 0:
            pooled = np.sqrt((np.std(energy_cc)**2 + np.std(energy_cw)**2) / 2)
            d = (np.mean(energy_cw) - np.mean(energy_cc)) / max(pooled, 1e-8)
            print(f"    Cohen's d (CW vs CC): {d:+.3f}  {'← GOOD separation' if d > 0.5 else '← weak'}")

    # ─── STEP 4: Sweep k and τ ───
    print(f"\n--- Step 4: Conditional subspace projection sweep ---")
    print(f"  Sweeping k (subspace dim) × τ (energy threshold) × λ (strength)\n")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    all_sweep_results = []

    for k in [1, 2, 3, min(5, max_k)]:
        if k > max_k:
            continue

        basis = pca.components_[:k].astype(np.float32)  # (k, d)

        # Compute energy distribution to set τ adaptively
        projections = all_centered @ basis.T
        energies = np.linalg.norm(projections, axis=1)
        tau_median = float(np.median(energies))
        tau_p75 = float(np.percentile(energies, 75))

        for tau in [0.0, tau_median, tau_p75]:
            for lam in [0.3, 0.5, 0.7, 1.0]:
                hook = SubspaceProjectionHook(basis, lam=lam, tau=tau, preserve_norm=True)
                hook_name = f"blocks.{target_layer}.hook_resid_post"

                model.reset_hooks()
                model.add_hook(hook_name, hook.hook_fn)

                per_sample = evaluate_hallucination(model, prompts, tokenizer)
                metrics = compute_hall_metrics(per_sample)
                diag = hook.get_diagnostics()

                q = metrics["quadrants"]
                tau_label = "none" if tau == 0 else f"med" if tau == tau_median else "p75"

                print(f"  k={k} τ={tau_label:>4s} λ={lam:.1f}  "
                      f"acc={metrics['accuracy']:.3f}  "
                      f"hall={metrics['hallucination_rate']:.3f}  "
                      f"cal={metrics['calibration_r']:+.3f}  "
                      f"proj_rate={diag.get('projection_rate', 1.0):.2f}  "
                      f"[CC={q['CC']} CW={q['CW']} UC={q['UC']} UW={q['UW']}]")

                all_sweep_results.append({
                    "k": k,
                    "tau": tau,
                    "tau_label": tau_label,
                    "lambda": lam,
                    "metrics": metrics,
                    "diagnostics": diag,
                })

                model.reset_hooks()
                hook.reset()

    # ─── STEP 5: Baseline comparison ───
    print(f"\n--- Baseline (no projection) ---")
    model.reset_hooks()
    baseline_samples = evaluate_hallucination(model, prompts, tokenizer)
    baseline_metrics = compute_hall_metrics(baseline_samples)
    q = baseline_metrics["quadrants"]
    print(f"  acc={baseline_metrics['accuracy']:.3f}  "
          f"hall={baseline_metrics['hallucination_rate']:.3f}  "
          f"cal={baseline_metrics['calibration_r']:+.3f}  "
          f"[CC={q['CC']} CW={q['CW']} UC={q['UC']} UW={q['UW']}]")

    # ─── STEP 6: Find best configuration ───
    print(f"\n{'='*60}")
    print(f"RESULTS: BEST CONFIGURATIONS")
    print(f"{'='*60}")

    baseline_hall = baseline_metrics["hallucination_rate"]
    baseline_acc = baseline_metrics["accuracy"]

    # Sort by: hallucination reduction (more negative = better), then accuracy loss (less = better)
    scored = []
    for r in all_sweep_results:
        hall_drop = baseline_hall - r["metrics"]["hallucination_rate"]
        acc_loss = baseline_acc - r["metrics"]["accuracy"]
        # Score: reward hall drop, penalize acc loss
        score = hall_drop - 2 * max(acc_loss, 0)
        scored.append((score, hall_drop, acc_loss, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  Baseline: acc={baseline_acc:.3f}, hall={baseline_hall:.3f}")
    print(f"\n  {'k':>3s}  {'τ':>5s}  {'λ':>5s}  {'Acc':>7s}  {'ΔAcc':>7s}  {'Hall':>7s}  {'ΔHall':>7s}  {'Cal':>7s}  {'ProjR':>6s}")
    print(f"  " + "─" * 65)

    for score, hall_drop, acc_loss, r in scored[:15]:
        m = r["metrics"]
        marker = ""
        if hall_drop > 0.01 and acc_loss <= 0.05:
            marker = " ← WIN"
        elif hall_drop > 0.01 and acc_loss <= 0.02:
            marker = " ← STRONG WIN"

        print(f"  {r['k']:>3d}  {r['tau_label']:>5s}  {r['lambda']:>5.1f}  "
              f"{m['accuracy']:>7.3f}  {-acc_loss:>+7.3f}  "
              f"{m['hallucination_rate']:>7.3f}  {-hall_drop:>+7.3f}  "
              f"{m['calibration_r']:>+7.3f}  "
              f"{r['diagnostics'].get('projection_rate', 1.0):>6.2f}{marker}")

    # ─── VERDICT ───
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    best = scored[0] if scored else None
    if best and best[1] > 0.01 and best[2] <= 0.05:
        r = best[3]
        m = r["metrics"]
        print(f"\n  HALLUCINATION DECREASED:")
        print(f"    Config: k={r['k']}, τ={r['tau_label']}, λ={r['lambda']}")
        print(f"    Accuracy:      {baseline_acc:.3f} → {m['accuracy']:.3f}  (Δ={-best[2]:+.3f})")
        print(f"    Hallucination: {baseline_hall:.3f} → {m['hallucination_rate']:.3f}  (Δ={-best[1]:+.3f})")
        print(f"    Calibration:   {baseline_metrics['calibration_r']:+.3f} → {m['calibration_r']:+.3f}")
        if best[1] > 0.03:
            print(f"\n  THIS IS THE RESULT. Subspace projection reduces hallucination.")
        else:
            print(f"\n  Weak signal. May need larger model to amplify.")
    elif best and best[1] > 0:
        print(f"\n  Hallucination decreased slightly but accuracy cost too high.")
        print(f"  Best: ΔHall={-best[1]:+.3f}, ΔAcc={-best[2]:+.3f}")
    else:
        print(f"\n  Hallucination did NOT decrease with any configuration.")
        print(f"  Pythia-410M may be too small for separable hallucination subspace.")

    # ─── SAVE ───
    output = {
        "model": model_name,
        "layer": target_layer,
        "baseline": baseline_metrics,
        "n_confident_wrong": n_cw,
        "n_confident_correct": n_cc,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "sweep": [
            {
                "k": r["k"],
                "tau": r["tau"],
                "tau_label": r["tau_label"],
                "lambda": r["lambda"],
                "metrics": r["metrics"],
                "diagnostics": r["diagnostics"],
            }
            for _, _, _, r in scored
        ],
    }

    safe_name = model_name.replace("/", "_")
    out_path = results_dir / f"subspace_projection_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Save basis vectors for future use
    for k in [1, 2, 3, min(5, max_k)]:
        if k > max_k:
            continue
        basis_path = results_dir / f"hall_subspace_k{k}_layer{target_layer}_{safe_name}.npy"
        np.save(basis_path, pca.components_[:k].astype(np.float32))
    print(f"Saved subspace basis vectors to results/")


if __name__ == "__main__":
    main()
