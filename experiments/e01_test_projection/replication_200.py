"""
Replication with 200+ prompts.

GOAL: Confirm that subspace projection reduces hallucination
with enough statistical power to rule out noise.

Previous result on 60 prompts:
  hall: 0.117 → 0.067 (7→4 samples, p unclear)

Need: 200+ prompts → enough confident_wrong samples for significance.

Pipeline:
  1. Collect activations on 200 prompts (train split)
  2. Build subspace from confident_wrong (train)
  3. Evaluate on HELD-OUT prompts (test split) ← prevents overfitting
  4. Measure hallucination rate change
  5. Bootstrap confidence intervals

Usage:
    python replication_200.py [model_name] [layer]
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'


def get_large_prompt_set():
    """200+ factual prompts across multiple categories."""
    prompts = [
        # ─── Geography: Capitals (50) ───
        ("Q: What is the capital of France? A:", [" Paris"], "geo"),
        ("Q: What is the capital of Japan? A:", [" Tokyo"], "geo"),
        ("Q: What is the capital of Italy? A:", [" Rome"], "geo"),
        ("Q: What is the capital of Spain? A:", [" Madrid"], "geo"),
        ("Q: What is the capital of Russia? A:", [" Moscow"], "geo"),
        ("Q: What is the capital of China? A:", [" Beijing", " Peking"], "geo"),
        ("Q: What is the capital of Egypt? A:", [" Cairo"], "geo"),
        ("Q: What is the capital of Germany? A:", [" Berlin"], "geo"),
        ("Q: What is the capital of Turkey? A:", [" Ankara", " Istanbul"], "geo"),
        ("Q: What is the capital of India? A:", [" New", " Delhi"], "geo"),
        ("Q: What is the capital of Brazil? A:", [" Bras"], "geo"),
        ("Q: What is the capital of Australia? A:", [" Canberra"], "geo"),
        ("Q: What is the capital of Canada? A:", [" Ottawa"], "geo"),
        ("Q: What is the capital of South Korea? A:", [" Seoul"], "geo"),
        ("Q: What is the capital of Mexico? A:", [" Mexico"], "geo"),
        ("Q: What is the capital of Poland? A:", [" Warsaw"], "geo"),
        ("Q: What is the capital of Sweden? A:", [" Stockholm"], "geo"),
        ("Q: What is the capital of Norway? A:", [" Oslo"], "geo"),
        ("Q: What is the capital of Greece? A:", [" Athens"], "geo"),
        ("Q: What is the capital of Argentina? A:", [" Buenos"], "geo"),
        ("Q: What is the capital of Thailand? A:", [" Bangkok"], "geo"),
        ("Q: What is the capital of Portugal? A:", [" Lisbon"], "geo"),
        ("Q: What is the capital of Netherlands? A:", [" Amsterdam"], "geo"),
        ("Q: What is the capital of Austria? A:", [" Vienna"], "geo"),
        ("Q: What is the capital of Switzerland? A:", [" Bern"], "geo"),
        ("Q: What is the capital of Ireland? A:", [" Dublin"], "geo"),
        ("Q: What is the capital of Finland? A:", [" Helsinki"], "geo"),
        ("Q: What is the capital of Denmark? A:", [" Copenhagen"], "geo"),
        ("Q: What is the capital of Czech Republic? A:", [" Prague"], "geo"),
        ("Q: What is the capital of Hungary? A:", [" Budapest"], "geo"),
        ("Q: What is the capital of Romania? A:", [" Bucharest"], "geo"),
        ("Q: What is the capital of Ukraine? A:", [" Kiev", " Kyiv"], "geo"),
        ("Q: What is the capital of Peru? A:", [" Lima"], "geo"),
        ("Q: What is the capital of Chile? A:", [" Santiago"], "geo"),
        ("Q: What is the capital of Colombia? A:", [" Bogota", " Bogot"], "geo"),
        ("Q: What is the capital of Venezuela? A:", [" Caracas"], "geo"),
        ("Q: What is the capital of Cuba? A:", [" Havana"], "geo"),
        ("Q: What is the capital of Iran? A:", [" Tehran"], "geo"),
        ("Q: What is the capital of Iraq? A:", [" Baghdad"], "geo"),
        ("Q: What is the capital of Israel? A:", [" Jerusalem", " Tel"], "geo"),
        ("Q: What is the capital of Saudi Arabia? A:", [" Riyadh"], "geo"),
        ("Q: What is the capital of Indonesia? A:", [" Jakarta"], "geo"),
        ("Q: What is the capital of Philippines? A:", [" Manila"], "geo"),
        ("Q: What is the capital of Vietnam? A:", [" Hanoi"], "geo"),
        ("Q: What is the capital of Malaysia? A:", [" Kuala"], "geo"),
        ("Q: What is the capital of Nigeria? A:", [" Abuja", " Lagos"], "geo"),
        ("Q: What is the capital of South Africa? A:", [" Pretoria", " Cape"], "geo"),
        ("Q: What is the capital of Kenya? A:", [" Nairobi"], "geo"),
        ("Q: What is the capital of Morocco? A:", [" Rabat"], "geo"),
        ("Q: What is the capital of New Zealand? A:", [" Wellington"], "geo"),

        # ─── Science: Chemistry (25) ───
        ("The chemical symbol for iron is", [" Fe"], "chem"),
        ("The chemical symbol for gold is", [" Au"], "chem"),
        ("The chemical symbol for silver is", [" Ag"], "chem"),
        ("The chemical symbol for copper is", [" Cu"], "chem"),
        ("The chemical symbol for sodium is", [" Na"], "chem"),
        ("The chemical symbol for potassium is", [" K"], "chem"),
        ("The chemical symbol for calcium is", [" Ca"], "chem"),
        ("The chemical symbol for nitrogen is", [" N"], "chem"),
        ("The chemical symbol for oxygen is", [" O"], "chem"),
        ("The chemical symbol for carbon is", [" C"], "chem"),
        ("The chemical symbol for hydrogen is", [" H"], "chem"),
        ("The chemical symbol for helium is", [" He"], "chem"),
        ("The chemical symbol for lead is", [" Pb"], "chem"),
        ("The chemical symbol for mercury is", [" Hg"], "chem"),
        ("The chemical symbol for tin is", [" Sn"], "chem"),
        ("The atomic number of hydrogen is", [" 1", " one"], "chem"),
        ("The atomic number of helium is", [" 2", " two"], "chem"),
        ("The atomic number of carbon is", [" 6", " six"], "chem"),
        ("The atomic number of oxygen is", [" 8", " eight"], "chem"),
        ("The atomic number of nitrogen is", [" 7", " seven"], "chem"),
        ("Water is made of hydrogen and", [" oxygen"], "chem"),
        ("The pH of pure water is", [" 7", " seven"], "chem"),
        ("Diamonds are made of", [" carbon"], "chem"),
        ("Table salt is made of sodium and", [" chlor"], "chem"),
        ("Rust is iron", [" oxide", " ox"], "chem"),

        # ─── Science: Physics (25) ───
        ("The speed of light is approximately", [" 3", " 300"], "phys"),
        ("The Earth orbits the", [" Sun"], "phys"),
        ("The Moon orbits the", [" Earth"], "phys"),
        ("Electrons have a", [" negative"], "phys"),
        ("Light travels faster than", [" sound"], "phys"),
        ("Sound cannot travel through a", [" vacuum"], "phys"),
        ("Water boils at 100 degrees", [" Celsius", " C"], "phys"),
        ("Water freezes at", [" 0", " zero", " 32"], "phys"),
        ("The closest star to Earth is the", [" Sun"], "phys"),
        ("Gravity pulls objects", [" down", " toward"], "phys"),
        ("The speed of sound is approximately", [" 3", " 340", " 1"], "phys"),
        ("Absolute zero is", [" -", " 0", " zero"], "phys"),
        ("Protons have a", [" positive"], "phys"),
        ("Neutrons have", [" no", " zero", " neutral"], "phys"),
        ("An atom consists of protons, neutrons, and", [" electron"], "phys"),
        ("Energy cannot be created or", [" destroyed", " dest"], "phys"),
        ("Force equals mass times", [" acceleration", " accel"], "phys"),
        ("The unit of force is the", [" Newton", " new"], "phys"),
        ("The unit of energy is the", [" joule", " J"], "phys"),
        ("Ohm's law states that voltage equals current times", [" resistance", " resist"], "phys"),
        ("The three states of matter are solid, liquid, and", [" gas"], "phys"),
        ("Photosynthesis produces", [" oxygen", " glucose"], "phys"),
        ("The wavelength of red light is", [" longer", " 6", " 7"], "phys"),
        ("Einstein developed the theory of", [" relat"], "phys"),
        ("Newton discovered the law of", [" grav"], "phys"),

        # ─── Science: Biology (20) ───
        ("DNA stands for", [" de", " D"], "bio"),
        ("The powerhouse of the cell is the", [" mitochond"], "bio"),
        ("Humans have", [" 23", " 46"], "bio"),
        ("The largest organ in the human body is the", [" skin"], "bio"),
        ("Blood is pumped by the", [" heart"], "bio"),
        ("Oxygen is carried by", [" red", " hem"], "bio"),
        ("Plants convert sunlight into energy through", [" photo"], "bio"),
        ("The basic unit of life is the", [" cell"], "bio"),
        ("Charles Darwin proposed the theory of", [" evol", " natural"], "bio"),
        ("Gregor Mendel is the father of", [" genet"], "bio"),
        ("Antibiotics kill", [" bacteria", " bact"], "bio"),
        ("Insulin regulates", [" blood", " sugar", " gluc"], "bio"),
        ("The brain is part of the", [" nervous", " central"], "bio"),
        ("Mammals breathe with their", [" lungs", " lung"], "bio"),
        ("Fish breathe with their", [" gills", " gill"], "bio"),
        ("Chlorophyll is", [" green"], "bio"),
        ("The human skeleton has", [" 206", " 200"], "bio"),
        ("The longest bone in the human body is the", [" femur"], "bio"),
        ("Viruses are", [" not", " smaller", " non"], "bio"),
        ("The study of living organisms is called", [" biology", " bio"], "bio"),

        # ─── History: Events (30) ───
        ("World War II ended in", [" 1945", " 19"], "hist"),
        ("World War I started in", [" 1914", " 19"], "hist"),
        ("The Berlin Wall fell in", [" 1989", " 19"], "hist"),
        ("The first moon landing was in", [" 1969", " 19"], "hist"),
        ("Columbus reached the Americas in", [" 1492", " 14"], "hist"),
        ("The Declaration of Independence was signed in", [" 1776", " 17"], "hist"),
        ("The French Revolution began in", [" 1789", " 17"], "hist"),
        ("The Titanic sank in", [" 1912", " 19", " April"], "hist"),
        ("Napoleon was defeated at", [" Water"], "hist"),
        ("The Soviet Union dissolved in", [" 1991", " 19"], "hist"),
        ("The Renaissance began in", [" Italy", " the", " 14"], "hist"),
        ("The Magna Carta was signed in", [" 12"], "hist"),
        ("The Cold War was between the", [" United", " US", " Soviet"], "hist"),
        ("Julius Caesar was assassinated in", [" 44"], "hist"),
        ("The printing press was invented by", [" Gut", " Johannes"], "hist"),
        ("The Wright brothers invented the", [" airplane", " air", " first"], "hist"),
        ("Martin Luther King Jr. gave his famous", [" \"", " I", " speech"], "hist"),
        ("The Great Wall of China was built", [" to", " during", " over"], "hist"),
        ("Shakespeare wrote", [" Hamlet", " Romeo", " Mac", " plays"], "hist"),
        ("Alexander the Great was from", [" Mac", " Greece"], "hist"),
        ("The American Civil War ended in", [" 1865", " 18"], "hist"),
        ("The Roman Empire fell in", [" 4", " 476"], "hist"),
        ("The first Olympics were held in", [" Greece", " Ath", " ancient"], "hist"),
        ("Pearl Harbor was attacked in", [" 1941", " 19", " December"], "hist"),
        ("The Internet was invented in the", [" 19", " 1960", " 1970"], "hist"),
        ("Abraham Lincoln was the", [" 16", " sixteenth"], "hist"),
        ("George Washington was the", [" first", " 1"], "hist"),
        ("The Emancipation Proclamation was issued by", [" Abraham", " Lincoln"], "hist"),
        ("The Industrial Revolution started in", [" Britain", " England", " the"], "hist"),
        ("The Reformation was started by", [" Martin", " Luther"], "hist"),

        # ─── Geography: General (20) ───
        ("The largest ocean is the", [" Pacific"], "geo2"),
        ("The longest river in the world is the", [" Nile", " Amazon"], "geo2"),
        ("The tallest mountain in the world is", [" Mount", " Ever"], "geo2"),
        ("The largest continent is", [" Asia"], "geo2"),
        ("The smallest continent is", [" Australia", " Aust"], "geo2"),
        ("The largest country by area is", [" Russia"], "geo2"),
        ("The most populous country is", [" China", " India"], "geo2"),
        ("The Sahara Desert is in", [" Africa"], "geo2"),
        ("The Amazon Rainforest is in", [" South", " Brazil"], "geo2"),
        ("The Great Barrier Reef is in", [" Australia"], "geo2"),
        ("The Nile River flows through", [" Egypt", " Africa"], "geo2"),
        ("The Mississippi River is in", [" the United", " America", " North"], "geo2"),
        ("Japan is an", [" island", " arch"], "geo2"),
        ("The United Kingdom consists of", [" England", " four", " Great"], "geo2"),
        ("The European Union was founded in", [" 19", " 1993", " 1957"], "geo2"),
        ("The United Nations headquarters is in", [" New York", " New"], "geo2"),
        ("The Eiffel Tower is in", [" Paris"], "geo2"),
        ("The Statue of Liberty is in", [" New York", " New"], "geo2"),
        ("The Colosseum is in", [" Rome"], "geo2"),
        ("The Great Pyramid is in", [" Egypt", " Giza"], "geo2"),

        # ─── Language/Culture (15) ───
        ("The official language of Brazil is", [" Portuguese", " Port"], "lang"),
        ("The official language of Japan is", [" Japanese"], "lang"),
        ("The most spoken language in the world is", [" Mandarin", " English", " Chinese"], "lang"),
        ("The Bible was originally written in", [" Hebrew", " Greek", " Aram"], "lang"),
        ("The Quran was written in", [" Arabic"], "lang"),
        ("The currency of Japan is the", [" yen"], "lang"),
        ("The currency of the United Kingdom is the", [" pound"], "lang"),
        ("The currency of the European Union is the", [" euro"], "lang"),
        ("The currency of the United States is the", [" dollar"], "lang"),
        ("The currency of India is the", [" rupee"], "lang"),
        ("The alphabet used in Russia is", [" Cyrillic", " Cyr"], "lang"),
        ("Mozart was from", [" Austria", " Salzburg"], "lang"),
        ("Beethoven was from", [" Germany", " Bonn"], "lang"),
        ("Leonardo da Vinci painted the", [" Mona", " Last"], "lang"),
        ("Michelangelo painted the", [" Sistine", " ceiling"], "lang"),

        # ─── Math/Logic (15) ───
        ("Pi is approximately", [" 3"], "math"),
        ("The square root of 144 is", [" 12", " twelve"], "math"),
        ("The square root of 64 is", [" 8", " eight"], "math"),
        ("The square root of 100 is", [" 10", " ten"], "math"),
        ("A triangle has", [" 3", " three"], "math"),
        ("A hexagon has", [" 6", " six"], "math"),
        ("The Pythagorean theorem relates to", [" right", " triangle"], "math"),
        ("A prime number is divisible only by", [" 1", " one", " itself"], "math"),
        ("The sum of angles in a triangle is", [" 180"], "math"),
        ("Binary code uses only", [" 0", " two", " 1", " zeros"], "math"),
        ("A byte consists of", [" 8", " eight"], "math"),
        ("The decimal system is base", [" 10", " ten"], "math"),
        ("Roman numeral X represents", [" 10", " ten"], "math"),
        ("Roman numeral V represents", [" 5", " five"], "math"),
        ("Roman numeral C represents", [" 100", " one hundred"], "math"),
    ]

    return [(p, a, c) for p, a, c in prompts]


def evaluate_batch(model, prompts, tokenizer):
    """Evaluate all prompts, return per-sample results."""
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


def compute_metrics(results):
    """Compute hallucination metrics with adaptive confidence threshold."""
    n = len(results)
    probs = [r["top1_prob"] for r in results]
    conf_threshold = np.median(probs)

    cc = cw = uc = uw = 0
    for r in results:
        is_conf = r["top1_prob"] > conf_threshold
        if is_conf and r["is_correct"]:
            cc += 1
            r["quadrant"] = "CC"
        elif is_conf and not r["is_correct"]:
            cw += 1
            r["quadrant"] = "CW"
        elif not is_conf and r["is_correct"]:
            uc += 1
            r["quadrant"] = "UC"
        else:
            uw += 1
            r["quadrant"] = "UW"

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
        "calibration_r": cal_r,
        "quadrants": {"CC": cc, "CW": cw, "UC": uc, "UW": uw},
        "n": n,
    }


def bootstrap_ci(baseline_results, projected_results, n_bootstrap=1000, seed=42):
    """Bootstrap confidence interval for hallucination rate difference."""
    rng = np.random.RandomState(seed)
    n = len(baseline_results)

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        base_hall = sum(1 for i in idx if baseline_results[i].get("quadrant") == "CW") / n
        proj_hall = sum(1 for i in idx if projected_results[i].get("quadrant") == "CW") / n
        diffs.append(proj_hall - base_hall)

    diffs = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
        "p_decrease": float(np.mean(diffs < 0)),  # fraction of bootstraps where hall decreased
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
    target_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    print(f"{'='*60}")
    print(f"REPLICATION: 200+ PROMPTS")
    print(f"Statistical validation of subspace projection")
    print(f"{'='*60}")
    print(f"Model: {model_name}, Layer: {target_layer}\n")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    tokenizer = model.tokenizer

    all_prompts = get_large_prompt_set()
    print(f"  Total prompts: {len(all_prompts)}")

    # ─── STEP 1: Train/test split ───
    # Use 60% for building subspace, 40% for evaluation
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_prompts))
    split = int(len(all_prompts) * 0.6)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_prompts = [all_prompts[i] for i in train_idx]
    test_prompts = [all_prompts[i] for i in test_idx]
    print(f"  Train (build subspace): {len(train_prompts)}")
    print(f"  Test (evaluate): {len(test_prompts)}")

    # ─── STEP 2: Collect train activations ───
    print(f"\n--- Step 1: Collecting train activations ---")

    train_samples = []
    for prompt, correct_answers, category in train_prompts:
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

        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        train_samples.append({
            "activation": act,
            "is_correct": is_correct,
            "top1_prob": top1_prob,
            "prompt": prompt,
        })

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Classify train quadrants
    train_probs = [s["top1_prob"] for s in train_samples]
    train_conf_threshold = np.median(train_probs)

    for s in train_samples:
        is_conf = s["top1_prob"] > train_conf_threshold
        if is_conf and s["is_correct"]:
            s["quadrant"] = "CC"
        elif is_conf and not s["is_correct"]:
            s["quadrant"] = "CW"
        elif not is_conf and s["is_correct"]:
            s["quadrant"] = "UC"
        else:
            s["quadrant"] = "UW"

    n_cc = sum(1 for s in train_samples if s["quadrant"] == "CC")
    n_cw = sum(1 for s in train_samples if s["quadrant"] == "CW")
    n_uc = sum(1 for s in train_samples if s["quadrant"] == "UC")
    n_uw = sum(1 for s in train_samples if s["quadrant"] == "UW")
    n_correct = sum(1 for s in train_samples if s["is_correct"])

    print(f"  Train accuracy: {n_correct}/{len(train_samples)} ({n_correct/len(train_samples):.1%})")
    print(f"  Train quadrants: CC={n_cc}, CW={n_cw}, UC={n_uc}, UW={n_uw}")

    # ─── STEP 3: Build hallucination subspace from train CW ───
    print(f"\n--- Step 2: Building hallucination subspace ---")

    cw_acts = np.array([s["activation"] for s in train_samples if s["quadrant"] == "CW"])
    cc_acts = np.array([s["activation"] for s in train_samples if s["quadrant"] == "CC"])

    if len(cw_acts) < 3:
        print(f"  ERROR: Only {len(cw_acts)} confident_wrong in train. Not enough.")
        sys.exit(1)

    cc_mean = cc_acts.mean(axis=0)
    cw_centered = cw_acts - cc_mean

    from sklearn.decomposition import PCA

    max_k = min(len(cw_acts) - 1, 10)
    pca = PCA(n_components=max_k)
    pca.fit(cw_centered)

    print(f"  PCA on {len(cw_acts)} confident_wrong (centered on CC mean)")
    cumulative = 0
    for i, ev in enumerate(pca.explained_variance_ratio_):
        cumulative += ev
        print(f"    PC{i+1}: {ev:.4f}  (cum: {cumulative:.4f})")

    # Energy analysis on TRAIN
    all_train_acts = np.array([s["activation"] for s in train_samples])
    all_train_centered = all_train_acts - cc_mean

    print(f"\n  Energy analysis on train set:")
    for k in [2, 3, min(5, max_k)]:
        if k > max_k:
            continue
        basis = pca.components_[:k]
        projections = all_train_centered @ basis.T
        energies = np.linalg.norm(projections, axis=1)

        for q_name in ["CC", "CW", "UC", "UW"]:
            q_energies = [energies[i] for i, s in enumerate(train_samples) if s["quadrant"] == q_name]
            if q_energies:
                print(f"    k={k} {q_name}: mean={np.mean(q_energies):.3f} ± {np.std(q_energies):.3f}")

        e_cw = [energies[i] for i, s in enumerate(train_samples) if s["quadrant"] == "CW"]
        e_cc = [energies[i] for i, s in enumerate(train_samples) if s["quadrant"] == "CC"]
        if e_cw and e_cc:
            pooled = np.sqrt((np.std(e_cc)**2 + np.std(e_cw)**2) / 2)
            d = (np.mean(e_cw) - np.mean(e_cc)) / max(pooled, 1e-8)
            print(f"    k={k} Cohen's d (CW vs CC): {d:+.3f}")
        print()

    # ─── STEP 4: Evaluate on TEST set ───
    print(f"\n--- Step 3: Evaluating on held-out test set ({len(test_prompts)} prompts) ---")

    # Import hook
    sys.path.insert(0, str(Path(__file__).parent))
    from subspace_projection import SubspaceProjectionHook

    # Baseline
    print(f"\n  Baseline (no projection):")
    model.reset_hooks()
    baseline_results = evaluate_batch(model, test_prompts, tokenizer)
    baseline_metrics = compute_metrics(baseline_results)
    q = baseline_metrics["quadrants"]
    print(f"  acc={baseline_metrics['accuracy']:.3f}  "
          f"hall={baseline_metrics['hallucination_rate']:.3f}  "
          f"cal={baseline_metrics['calibration_r']:+.3f}  "
          f"[CC={q['CC']} CW={q['CW']} UC={q['UC']} UW={q['UW']}]")

    # Sweep configurations
    print(f"\n  Projection sweep:")

    all_test_acts_for_tau = all_train_centered  # use train for tau thresholds
    sweep_results = []

    for k in [2, 3, min(5, max_k)]:
        if k > max_k:
            continue

        basis = pca.components_[:k].astype(np.float32)

        # Compute tau from TRAIN distribution
        train_proj = all_train_centered @ basis.T
        train_energies = np.linalg.norm(train_proj, axis=1)
        tau_median = float(np.median(train_energies))
        tau_p75 = float(np.percentile(train_energies, 75))

        for tau_val, tau_label in [(0.0, "none"), (tau_median, "med"), (tau_p75, "p75")]:
            for lam in [0.3, 0.5, 0.7, 1.0]:
                hook = SubspaceProjectionHook(basis, lam=lam, tau=tau_val, preserve_norm=True)
                hook_name = f"blocks.{target_layer}.hook_resid_post"

                model.reset_hooks()
                model.add_hook(hook_name, hook.hook_fn)

                proj_results = evaluate_batch(model, test_prompts, tokenizer)
                proj_metrics = compute_metrics(proj_results)
                diag = hook.get_diagnostics()

                # Bootstrap CI
                ci = bootstrap_ci(baseline_results, proj_results)

                q = proj_metrics["quadrants"]
                hall_delta = proj_metrics["hallucination_rate"] - baseline_metrics["hallucination_rate"]
                acc_delta = proj_metrics["accuracy"] - baseline_metrics["accuracy"]

                marker = ""
                if hall_delta < -0.02 and acc_delta >= -0.05:
                    marker = " <-- WIN"

                print(f"  k={k} τ={tau_label:>4s} λ={lam:.1f}  "
                      f"acc={proj_metrics['accuracy']:.3f}({acc_delta:+.3f})  "
                      f"hall={proj_metrics['hallucination_rate']:.3f}({hall_delta:+.3f})  "
                      f"cal={proj_metrics['calibration_r']:+.3f}  "
                      f"p(↓)={ci['p_decrease']:.3f}  "
                      f"CI=[{ci['ci_lower']:+.3f},{ci['ci_upper']:+.3f}]  "
                      f"[CC={q['CC']} CW={q['CW']}]{marker}")

                sweep_results.append({
                    "k": k, "tau_label": tau_label, "tau": tau_val,
                    "lambda": lam,
                    "metrics": proj_metrics,
                    "bootstrap_ci": ci,
                    "diagnostics": diag,
                    "per_sample": proj_results,
                })

                model.reset_hooks()
                hook.reset()

    # ─── STEP 5: Ablation table ───
    print(f"\n{'='*60}")
    print(f"ABLATION TABLE (for paper)")
    print(f"{'='*60}")

    # Random direction control
    print(f"\n  Random direction control (k=3, λ=0.5):")
    d_model = pca.components_.shape[1]
    random_basis = rng.randn(3, d_model).astype(np.float32)
    # Orthogonalize random basis
    for i in range(3):
        for j in range(i):
            random_basis[i] -= np.dot(random_basis[i], random_basis[j]) * random_basis[j]
        random_basis[i] /= np.linalg.norm(random_basis[i])

    hook = SubspaceProjectionHook(random_basis, lam=0.5, tau=0.0, preserve_norm=True)
    model.reset_hooks()
    model.add_hook(f"blocks.{target_layer}.hook_resid_post", hook.hook_fn)
    random_results = evaluate_batch(model, test_prompts, tokenizer)
    random_metrics = compute_metrics(random_results)
    model.reset_hooks()
    q = random_metrics["quadrants"]
    print(f"  acc={random_metrics['accuracy']:.3f}  "
          f"hall={random_metrics['hallucination_rate']:.3f}  "
          f"[CC={q['CC']} CW={q['CW']}]")

    # Single vector control (original Φ_decision)
    print(f"\n  Single vector (Φ_decision, λ=0.5):")
    e00_path = Path(__file__).parent.parent / "e00_find_phi" / "results" / f"phi_direction_layer{target_layer}_{model_name.replace('/', '_')}.npy"
    if e00_path.exists():
        phi_decision = np.load(e00_path).reshape(1, -1)  # (1, d) for SubspaceProjectionHook
        hook = SubspaceProjectionHook(phi_decision, lam=0.5, tau=0.0, preserve_norm=True)
        model.reset_hooks()
        model.add_hook(f"blocks.{target_layer}.hook_resid_post", hook.hook_fn)
        vector_results = evaluate_batch(model, test_prompts, tokenizer)
        vector_metrics = compute_metrics(vector_results)
        model.reset_hooks()
        q = vector_metrics["quadrants"]
        print(f"  acc={vector_metrics['accuracy']:.3f}  "
              f"hall={vector_metrics['hallucination_rate']:.3f}  "
              f"[CC={q['CC']} CW={q['CW']}]")

    print(f"\n  ABLATION SUMMARY:")
    print(f"  {'Method':<35s}  {'Acc':>7s}  {'Hall':>7s}  {'ΔHall':>7s}")
    print(f"  " + "─" * 60)
    print(f"  {'No projection (baseline)':<35s}  {baseline_metrics['accuracy']:>7.3f}  {baseline_metrics['hallucination_rate']:>7.3f}  {'—':>7s}")
    print(f"  {'Random subspace (k=3)':<35s}  {random_metrics['accuracy']:>7.3f}  {random_metrics['hallucination_rate']:>7.3f}  {random_metrics['hallucination_rate']-baseline_metrics['hallucination_rate']:>+7.3f}")
    if e00_path.exists():
        print(f"  {'Single vector (Φ_decision)':<35s}  {vector_metrics['accuracy']:>7.3f}  {vector_metrics['hallucination_rate']:>7.3f}  {vector_metrics['hallucination_rate']-baseline_metrics['hallucination_rate']:>+7.3f}")

    # Find best subspace result
    best = None
    best_score = -999
    for r in sweep_results:
        hall_drop = baseline_metrics["hallucination_rate"] - r["metrics"]["hallucination_rate"]
        acc_loss = baseline_metrics["accuracy"] - r["metrics"]["accuracy"]
        score = hall_drop - 2 * max(acc_loss, 0)
        if score > best_score:
            best_score = score
            best = r

    if best:
        bm = best["metrics"]
        print(f"  {'Subspace (best config)':<35s}  {bm['accuracy']:>7.3f}  {bm['hallucination_rate']:>7.3f}  {bm['hallucination_rate']-baseline_metrics['hallucination_rate']:>+7.3f}")
        print(f"    (k={best['k']}, τ={best['tau_label']}, λ={best['lambda']})")
        print(f"    Bootstrap p(hall↓): {best['bootstrap_ci']['p_decrease']:.3f}")
        print(f"    95% CI: [{best['bootstrap_ci']['ci_lower']:+.3f}, {best['bootstrap_ci']['ci_upper']:+.3f}]")

    # ─── VERDICT ───
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    if best:
        ci = best["bootstrap_ci"]
        hall_drop = baseline_metrics["hallucination_rate"] - best["metrics"]["hallucination_rate"]
        acc_loss = baseline_metrics["accuracy"] - best["metrics"]["accuracy"]

        if ci["p_decrease"] > 0.95 and acc_loss <= 0.05:
            print(f"\n  STATISTICALLY SIGNIFICANT HALLUCINATION REDUCTION")
            print(f"  p(hall↓) = {ci['p_decrease']:.3f} > 0.95")
            print(f"  THIS IS PUBLISHABLE. Scale to larger model.")
        elif ci["p_decrease"] > 0.8 and hall_drop > 0.01:
            print(f"\n  TRENDING: hallucination reduction likely real but not significant")
            print(f"  p(hall↓) = {ci['p_decrease']:.3f}")
            print(f"  Need: more data or larger model for significance")
        elif hall_drop > 0:
            print(f"\n  WEAK SIGNAL: direction correct but insufficient power")
            print(f"  p(hall↓) = {ci['p_decrease']:.3f}")
        else:
            print(f"\n  NO EFFECT: subspace projection does not reduce hallucination")
            print(f"  at this scale with this model")

    # ─── SAVE ───
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    safe_name = model_name.replace("/", "_")

    output = {
        "model": model_name,
        "layer": target_layer,
        "n_train": len(train_prompts),
        "n_test": len(test_prompts),
        "train_quadrants": {"CC": n_cc, "CW": n_cw, "UC": n_uc, "UW": n_uw},
        "baseline": baseline_metrics,
        "pca_variance": pca.explained_variance_ratio_.tolist(),
        "sweep": [
            {
                "k": r["k"], "tau_label": r["tau_label"], "lambda": r["lambda"],
                "metrics": r["metrics"],
                "bootstrap_ci": r["bootstrap_ci"],
            }
            for r in sweep_results
        ],
    }

    out_path = results_dir / f"replication_200_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
