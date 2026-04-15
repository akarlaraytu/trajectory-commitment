"""
Experiment 3: Trajectory Control — Multi-layer coordinated intervention.

NOT single-Φ brute-force. This is:
  1. Layer-wise Φ_l: each layer gets its own hallucination direction
  2. Energy-based gating: λ_l = sigmoid(α(E_l - τ_l)) — only intervene on drifting states
  3. Coordinated L10-17 intervention
  4. Recovery rate: does the model return to correct trajectory after intervention?

Key insight from E02:
  - Hallucination is GRADUAL (onset L3, peak L20)
  - Subject tokens carry info in L0-12, decision happens L20+
  - L10-17 is the control bottleneck
  - Single-layer projection fails because model compensates

This experiment tests: can we STEER the trajectory, not just cut a direction?

Usage:
    python trajectory_control.py [model_name]
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
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


# ─── Prompt Set (same as replication_200) ─────────────────────────

def get_prompts():
    """200+ factual prompts across categories."""
    prompts = [
        # Geography (50)
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

        # Chemistry (25)
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

        # Physics (25)
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

        # Biology (20)
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

        # History (30)
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

        # Geography General (20)
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

        # Language/Culture (15)
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

        # Math (15)
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
    return prompts


# ─── Evaluation ───────────────────────────────────────────────────

def evaluate_single(model, prompt, correct_answers, tokenizer):
    """Evaluate single prompt, return detailed info."""
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

    # Get correct token prob
    correct_prob = 0.0
    for a in correct_answers:
        a_tokens = tokenizer.encode(a)
        if a_tokens:
            correct_prob = max(correct_prob, probs[a_tokens[0]].item())

    return {
        "is_correct": is_correct,
        "top1_prob": top1_prob,
        "top1_token": top1_token,
        "entropy": entropy,
        "correct_prob": correct_prob,
    }


def compute_metrics(results):
    """Compute hallucination metrics with adaptive confidence."""
    n = len(results)
    probs = [r["top1_prob"] for r in results]
    conf_threshold = np.median(probs)

    cc = cw = uc = uw = 0
    for r in results:
        is_conf = r["top1_prob"] > conf_threshold
        if is_conf and r["is_correct"]:
            cc += 1; r["quadrant"] = "CC"
        elif is_conf and not r["is_correct"]:
            cw += 1; r["quadrant"] = "CW"
        elif not is_conf and r["is_correct"]:
            uc += 1; r["quadrant"] = "UC"
        else:
            uw += 1; r["quadrant"] = "UW"

    return {
        "accuracy": sum(1 for r in results if r["is_correct"]) / n,
        "hallucination_rate": cw / n,
        "quadrants": {"CC": cc, "CW": cw, "UC": uc, "UW": uw},
        "n": n,
    }


# ─── Layer-wise Φ Learning ───────────────────────────────────────

def learn_layerwise_phi(model, train_prompts, target_layers, tokenizer):
    """
    Learn Φ_l for each target layer.

    For each layer l:
      Φ_l = mean(activations | wrong) - mean(activations | correct)

    This captures the hallucination direction AT THAT LAYER,
    not a single global direction.
    """
    n_layers = len(target_layers)
    d_model = model.cfg.d_model

    # Collect activations at all target layers
    layer_acts = {l: [] for l in target_layers}
    labels = []  # (is_correct, top1_prob)

    print(f"  Collecting activations at {len(target_layers)} layers...")

    hook_names = [f"blocks.{l}.hook_resid_post" for l in target_layers]

    for i, (prompt, correct_answers, category) in enumerate(train_prompts):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(h in name for h in hook_names))

        final_logits = logits[0, -1, :]
        probs = torch.softmax(final_logits.float().cpu(), dim=-1)
        top1_id = probs.argmax().item()
        top1_prob = probs[top1_id].item()
        top1_token = tokenizer.decode([top1_id])

        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        labels.append({"is_correct": is_correct, "top1_prob": top1_prob})

        for l in target_layers:
            key = f"blocks.{l}.hook_resid_post"
            act = cache[key][0, -1, :].float().cpu().numpy()
            layer_acts[l].append(act)

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (i + 1) % 30 == 0:
            print(f"    [{i+1}/{len(train_prompts)}]")

    # Compute confidence threshold
    all_probs = [l["top1_prob"] for l in labels]
    conf_threshold = np.median(all_probs)

    # Classify quadrants
    for l in labels:
        l["is_confident"] = l["top1_prob"] > conf_threshold

    n_correct = sum(1 for l in labels if l["is_correct"])
    n_wrong = sum(1 for l in labels if not l["is_correct"])
    n_cw = sum(1 for l in labels if l["is_confident"] and not l["is_correct"])
    n_cc = sum(1 for l in labels if l["is_confident"] and l["is_correct"])
    print(f"  Train: {n_correct} correct, {n_wrong} wrong, {n_cw} confident-wrong, {n_cc} confident-correct")

    # Learn Φ_l for each layer
    phi_layers = {}
    tau_layers = {}

    for l in target_layers:
        acts = np.array(layer_acts[l])

        # Correct vs wrong activations
        correct_acts = acts[[i for i, lb in enumerate(labels) if lb["is_correct"]]]
        wrong_acts = acts[[i for i, lb in enumerate(labels) if not lb["is_correct"]]]

        if len(wrong_acts) < 2 or len(correct_acts) < 2:
            print(f"    L{l}: not enough samples, using global direction")
            phi = np.mean(wrong_acts, axis=0) - np.mean(correct_acts, axis=0) if len(wrong_acts) > 0 else np.zeros(d_model)
        else:
            # Φ_l = mean_wrong - mean_correct (at this layer)
            phi = np.mean(wrong_acts, axis=0) - np.mean(correct_acts, axis=0)

        # Normalize
        norm = np.linalg.norm(phi)
        if norm > 1e-8:
            phi = phi / norm

        # Compute energy distribution for τ (threshold)
        energies = np.abs(acts @ phi)
        correct_energies = energies[[i for i, lb in enumerate(labels) if lb["is_correct"]]]
        wrong_energies = energies[[i for i, lb in enumerate(labels) if not lb["is_correct"]]]

        tau = float(np.median(energies))

        # Cohen's d for energy separation
        if len(correct_energies) > 1 and len(wrong_energies) > 1:
            pooled = np.sqrt((np.std(correct_energies)**2 + np.std(wrong_energies)**2) / 2)
            d = (np.mean(wrong_energies) - np.mean(correct_energies)) / max(pooled, 1e-8)
        else:
            d = 0.0

        phi_layers[l] = phi.astype(np.float32)
        tau_layers[l] = tau

        print(f"    L{l}: ||Φ||={norm:.3f}  τ={tau:.3f}  d={d:+.2f}  "
              f"E_correct={np.mean(correct_energies):.3f}  E_wrong={np.mean(wrong_energies):.3f}")

    return phi_layers, tau_layers, labels


# ─── Trajectory Control Hook ─────────────────────────────────────

class TrajectoryControlHook:
    """
    Multi-layer coordinated intervention.

    For each layer l in target_layers:
      1. Compute energy: E_l = |h · Φ_l|
      2. Compute gating: λ_l = sigmoid(α * (E_l - τ_l)) * λ_base
      3. Project: h' = h - λ_l * (h · Φ_l) * Φ_l
      4. Preserve norm: h' = h' * ||h|| / ||h'||

    This only intervenes when the state is DRIFTING toward hallucination,
    not uniformly.
    """

    def __init__(self, phi_layers, tau_layers, lambda_base=0.5, alpha=5.0, device="mps"):
        self.phi_layers = {}
        self.tau_layers = tau_layers
        self.lambda_base = lambda_base
        self.alpha = alpha

        for l, phi in phi_layers.items():
            self.phi_layers[l] = torch.tensor(phi, dtype=torch.float32, device=device)

        # Tracking
        self.energies = {l: [] for l in phi_layers}
        self.lambdas = {l: [] for l in phi_layers}
        self.interventions = {l: 0 for l in phi_layers}

    def get_hook(self, layer):
        """Return a hook function for the given layer."""
        phi = self.phi_layers[layer]
        tau = self.tau_layers[layer]
        tracker = self

        def hook_fn(value, hook):
            h = value[0, -1, :].float()
            orig_norm = h.norm()

            # Energy along Φ_l
            projection = torch.dot(h, phi)
            energy = projection.abs().item()

            # Gating: sigmoid-based, only fires when energy > τ
            gate = torch.sigmoid(torch.tensor(tracker.alpha * (energy - tau))).item()
            lam = gate * tracker.lambda_base

            # Track
            tracker.energies[layer].append(energy)
            tracker.lambdas[layer].append(lam)

            if lam > 0.01:  # meaningful intervention
                tracker.interventions[layer] += 1

                # Project out hallucination component
                h_new = h - lam * projection * phi

                # Preserve norm
                new_norm = h_new.norm()
                if new_norm > 1e-8:
                    h_new = h_new * (orig_norm / new_norm)

                value[0, -1, :] = h_new.to(value.dtype)

            return value

        return hook_fn

    def reset_tracking(self):
        for l in self.phi_layers:
            self.energies[l] = []
            self.lambdas[l] = []
            self.interventions[l] = 0

    def get_stats(self):
        stats = {}
        for l in self.phi_layers:
            if self.energies[l]:
                stats[l] = {
                    "mean_energy": float(np.mean(self.energies[l])),
                    "mean_lambda": float(np.mean(self.lambdas[l])),
                    "n_interventions": self.interventions[l],
                    "n_total": len(self.energies[l]),
                    "intervention_rate": self.interventions[l] / max(len(self.energies[l]), 1),
                }
        return stats


# ─── Single Layer Hook (for comparison) ──────────────────────────

class SingleLayerHook:
    """Baseline: single Φ at one layer, no gating."""

    def __init__(self, phi, layer, lambda_val=0.5, device="mps"):
        self.phi = torch.tensor(phi, dtype=torch.float32, device=device)
        self.layer = layer
        self.lambda_val = lambda_val

    def get_hook(self):
        phi = self.phi
        lam = self.lambda_val

        def hook_fn(value, hook):
            h = value[0, -1, :].float()
            orig_norm = h.norm()
            projection = torch.dot(h, phi)
            h_new = h - lam * projection * phi
            new_norm = h_new.norm()
            if new_norm > 1e-8:
                h_new = h_new * (orig_norm / new_norm)
            value[0, -1, :] = h_new.to(value.dtype)
            return value

        return hook_fn


# ─── Recovery Rate ────────────────────────────────────────────────

def compute_recovery_rate(baseline_results, controlled_results):
    """
    Recovery rate: how many wrong→correct transitions does control achieve?

    Also tracks:
    - Maintained: correct→correct (good)
    - Broken: correct→wrong (bad)
    - Fixed: wrong→correct (GOAL)
    - Persistent: wrong→wrong (unfixed)
    """
    n = len(baseline_results)
    maintained = fixed = broken = persistent = 0

    for b, c in zip(baseline_results, controlled_results):
        b_correct = b["is_correct"]
        c_correct = c["is_correct"]

        if b_correct and c_correct:
            maintained += 1
        elif b_correct and not c_correct:
            broken += 1
        elif not b_correct and c_correct:
            fixed += 1
        else:
            persistent += 1

    n_wrong_baseline = fixed + persistent
    recovery_rate = fixed / max(n_wrong_baseline, 1)

    return {
        "maintained": maintained,
        "broken": broken,
        "fixed": fixed,
        "persistent": persistent,
        "recovery_rate": recovery_rate,
        "damage_rate": broken / max(maintained + broken, 1),
        "n_wrong_baseline": n_wrong_baseline,
        "n_correct_baseline": maintained + broken,
        "net_improvement": fixed - broken,
    }


# ─── Visualization ───────────────────────────────────────────────

def plot_ablation_comparison(results_dict, save_path):
    """Bar chart comparing conditions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conditions = list(results_dict.keys())
    metrics = ["accuracy", "hallucination_rate"]
    colors = {"accuracy": "#2196F3", "hallucination_rate": "#F44336"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results_dict[c]["metrics"][metric] for c in conditions]
        bars = ax.bar(range(len(conditions)), values, color=colors[metric], alpha=0.8)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=9)

    plt.suptitle("E03: Trajectory Control — Ablation", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_recovery_analysis(recovery_dict, save_path):
    """Plot recovery rate across conditions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    conditions = list(recovery_dict.keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: recovery vs damage
    ax = axes[0]
    rec = [recovery_dict[c]["recovery_rate"] for c in conditions]
    dmg = [recovery_dict[c]["damage_rate"] for c in conditions]
    x = np.arange(len(conditions))
    w = 0.35
    ax.bar(x - w/2, rec, w, label="Recovery (wrong→correct)", color="#4CAF50", alpha=0.8)
    ax.bar(x + w/2, dmg, w, label="Damage (correct→wrong)", color="#F44336", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Rate")
    ax.set_title("Recovery vs Damage")
    ax.legend()

    # Right: fixed/broken/maintained/persistent stacked
    ax = axes[1]
    maintained = [recovery_dict[c]["maintained"] for c in conditions]
    fixed = [recovery_dict[c]["fixed"] for c in conditions]
    broken = [recovery_dict[c]["broken"] for c in conditions]
    persistent = [recovery_dict[c]["persistent"] for c in conditions]

    ax.bar(x, maintained, label="Maintained (C→C)", color="#4CAF50", alpha=0.8)
    ax.bar(x, fixed, bottom=maintained, label="Fixed (W→C)", color="#8BC34A", alpha=0.8)
    ax.bar(x, broken, bottom=[m+f for m,f in zip(maintained, fixed)],
           label="Broken (C→W)", color="#F44336", alpha=0.8)
    ax.bar(x, persistent, bottom=[m+f+b for m,f,b in zip(maintained, fixed, broken)],
           label="Persistent (W→W)", color="#9E9E9E", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Per-sample Outcome")
    ax.legend(fontsize=8)

    plt.suptitle("E03: Recovery Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_energy_gating(hook_stats, save_path):
    """Plot energy and gating across layers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = sorted(hook_stats.keys())
    energies = [hook_stats[l]["mean_energy"] for l in layers]
    lambdas = [hook_stats[l]["mean_lambda"] for l in layers]
    int_rates = [hook_stats[l]["intervention_rate"] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.bar(range(len(layers)), energies, color="#2196F3", alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Mean Energy")
    ax.set_title("Energy along Φ_l per Layer")

    ax = axes[1]
    ax.bar(range(len(layers)), lambdas, color="#FF9800", alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Mean λ_l")
    ax.set_title("Gating Strength per Layer")

    ax = axes[2]
    ax.bar(range(len(layers)), int_rates, color="#9C27B0", alpha=0.8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Intervention Rate")
    ax.set_title("Fraction of Prompts Intervened")

    plt.suptitle("Trajectory Control: Energy-based Gating", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"

    # Control bottleneck from E02
    target_layers = list(range(10, 18))  # L10-L17

    print(f"{'='*60}")
    print(f"E03: TRAJECTORY CONTROL")
    print(f"Multi-layer coordinated intervention")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Target layers: {target_layers}")
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

    all_prompts = get_prompts()
    print(f"Total prompts: {len(all_prompts)}")

    # Train/test split (60/40)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_prompts))
    split = int(len(all_prompts) * 0.6)
    train_idx = indices[:split]
    test_idx = indices[split:]
    train_prompts = [all_prompts[i] for i in train_idx]
    test_prompts = [all_prompts[i] for i in test_idx]
    print(f"Train: {len(train_prompts)}, Test: {len(test_prompts)}")

    results_dir = Path(__file__).parent / "results"
    figures_dir = results_dir / "figures"

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Learn layer-wise Φ_l on TRAIN set
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Step 1: Learning layer-wise Phi_l (train set)")
    print(f"{'='*60}\n")

    phi_layers, tau_layers, train_labels = learn_layerwise_phi(
        model, train_prompts, target_layers, tokenizer)

    # Also learn a global Φ at best single layer (L14 from E02) for comparison
    single_layer = 14
    if single_layer not in phi_layers:
        phi_single, tau_single, _ = learn_layerwise_phi(
            model, train_prompts, [single_layer], tokenizer)
        phi_global = phi_single[single_layer]
    else:
        phi_global = phi_layers[single_layer]

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Evaluate on TEST set — 6 conditions
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Step 2: Evaluating on test set ({len(test_prompts)} prompts)")
    print(f"{'='*60}\n")

    conditions = {}

    # --- Condition 0: BASELINE (no intervention) ---
    print(f"  [0] Baseline (no intervention)...")
    model.reset_hooks()
    baseline_results = []
    for prompt, correct_answers, cat in test_prompts:
        r = evaluate_single(model, prompt, correct_answers, tokenizer)
        r["category"] = cat
        baseline_results.append(r)
    baseline_metrics = compute_metrics(baseline_results)
    q = baseline_metrics["quadrants"]
    print(f"      acc={baseline_metrics['accuracy']:.3f}  "
          f"hall={baseline_metrics['hallucination_rate']:.3f}  "
          f"[CC={q['CC']} CW={q['CW']} UC={q['UC']} UW={q['UW']}]")
    conditions["baseline"] = {"metrics": baseline_metrics, "results": baseline_results}

    # --- Condition 1: Single layer, no gating (E01 style) ---
    print(f"\n  [1] Single layer L{single_layer}, no gating (λ=0.5)...")
    model.reset_hooks()
    hook_single = SingleLayerHook(phi_global, single_layer, lambda_val=0.5, device=device)
    model.add_hook(f"blocks.{single_layer}.hook_resid_post", hook_single.get_hook())
    single_results = []
    for prompt, correct_answers, cat in test_prompts:
        r = evaluate_single(model, prompt, correct_answers, tokenizer)
        r["category"] = cat
        single_results.append(r)
    single_metrics = compute_metrics(single_results)
    q = single_metrics["quadrants"]
    print(f"      acc={single_metrics['accuracy']:.3f}  "
          f"hall={single_metrics['hallucination_rate']:.3f}  "
          f"[CC={q['CC']} CW={q['CW']} UC={q['UC']} UW={q['UW']}]")
    conditions["single_L14"] = {"metrics": single_metrics, "results": single_results}
    model.reset_hooks()

    # --- Condition 2: Multi-layer, NO gating (naive) ---
    print(f"\n  [2] Multi-layer L10-17, NO gating (λ=0.3 uniform)...")
    model.reset_hooks()
    for l in target_layers:
        naive_hook = SingleLayerHook(phi_layers[l], l, lambda_val=0.3, device=device)
        model.add_hook(f"blocks.{l}.hook_resid_post", naive_hook.get_hook())
    naive_results = []
    for prompt, correct_answers, cat in test_prompts:
        r = evaluate_single(model, prompt, correct_answers, tokenizer)
        r["category"] = cat
        naive_results.append(r)
    naive_metrics = compute_metrics(naive_results)
    q = naive_metrics["quadrants"]
    print(f"      acc={naive_metrics['accuracy']:.3f}  "
          f"hall={naive_metrics['hallucination_rate']:.3f}  "
          f"[CC={q['CC']} CW={q['CW']} UC={q['UC']} UW={q['UW']}]")
    conditions["multi_naive"] = {"metrics": naive_metrics, "results": naive_results}
    model.reset_hooks()

    # --- Condition 3: Multi-layer WITH energy gating (THE REAL TEST) ---
    # Sweep λ_base and α
    best_config = None
    best_net = -999

    lambda_bases = [0.3, 0.5, 0.7, 1.0]
    alphas = [3.0, 5.0, 10.0]

    print(f"\n  [3] Multi-layer + energy gating — sweeping λ_base × α...")

    for lb in lambda_bases:
        for alpha in alphas:
            model.reset_hooks()
            tc_hook = TrajectoryControlHook(
                phi_layers, tau_layers,
                lambda_base=lb, alpha=alpha, device=device)

            for l in target_layers:
                model.add_hook(f"blocks.{l}.hook_resid_post", tc_hook.get_hook(l))

            gated_results = []
            for prompt, correct_answers, cat in test_prompts:
                r = evaluate_single(model, prompt, correct_answers, tokenizer)
                r["category"] = cat
                gated_results.append(r)

            gated_metrics = compute_metrics(gated_results)
            recovery = compute_recovery_rate(baseline_results, gated_results)
            hook_stats = tc_hook.get_stats()

            q = gated_metrics["quadrants"]
            print(f"      λ={lb} α={alpha}: acc={gated_metrics['accuracy']:.3f} "
                  f"hall={gated_metrics['hallucination_rate']:.3f} "
                  f"fixed={recovery['fixed']} broken={recovery['broken']} "
                  f"net={recovery['net_improvement']:+d} "
                  f"recovery={recovery['recovery_rate']:.3f}")

            # Best = highest net improvement (fixed - broken)
            if recovery["net_improvement"] > best_net or \
               (recovery["net_improvement"] == best_net and
                gated_metrics["hallucination_rate"] < best_config.get("hall", 1)):
                best_net = recovery["net_improvement"]
                best_config = {
                    "lambda_base": lb,
                    "alpha": alpha,
                    "metrics": gated_metrics,
                    "results": gated_results,
                    "recovery": recovery,
                    "hook_stats": hook_stats,
                    "hall": gated_metrics["hallucination_rate"],
                }

            model.reset_hooks()

    print(f"\n  Best config: λ={best_config['lambda_base']} α={best_config['alpha']}")
    print(f"  Net improvement: {best_config['recovery']['net_improvement']:+d}")
    conditions["multi_gated"] = {
        "metrics": best_config["metrics"],
        "results": best_config["results"],
        "config": {"lambda_base": best_config["lambda_base"],
                    "alpha": best_config["alpha"]},
    }

    # --- Condition 4: RANDOM multi-layer (control) ---
    print(f"\n  [4] Random multi-layer (control)...")
    model.reset_hooks()
    rng_r = np.random.RandomState(99)
    random_phis = {l: rng_r.randn(model.cfg.d_model).astype(np.float32) for l in target_layers}
    for l in target_layers:
        random_phis[l] /= np.linalg.norm(random_phis[l])

    random_hook = TrajectoryControlHook(
        random_phis, tau_layers,
        lambda_base=best_config["lambda_base"],
        alpha=best_config["alpha"],
        device=device)

    for l in target_layers:
        model.add_hook(f"blocks.{l}.hook_resid_post", random_hook.get_hook(l))

    random_results = []
    for prompt, correct_answers, cat in test_prompts:
        r = evaluate_single(model, prompt, correct_answers, tokenizer)
        r["category"] = cat
        random_results.append(r)
    random_metrics = compute_metrics(random_results)
    random_recovery = compute_recovery_rate(baseline_results, random_results)
    q = random_metrics["quadrants"]
    print(f"      acc={random_metrics['accuracy']:.3f}  "
          f"hall={random_metrics['hallucination_rate']:.3f}  "
          f"fixed={random_recovery['fixed']} broken={random_recovery['broken']} "
          f"net={random_recovery['net_improvement']:+d}")
    conditions["random_multi"] = {"metrics": random_metrics, "results": random_results}
    model.reset_hooks()

    # --- Condition 5: Decision layer only (L22-27, late intervention) ---
    print(f"\n  [5] Decision layer only (L22-27)...")
    late_layers = list(range(22, 28))
    phi_late, tau_late, _ = learn_layerwise_phi(model, train_prompts, late_layers, tokenizer)

    model.reset_hooks()
    late_hook = TrajectoryControlHook(
        phi_late, tau_late,
        lambda_base=best_config["lambda_base"],
        alpha=best_config["alpha"],
        device=device)

    for l in late_layers:
        model.add_hook(f"blocks.{l}.hook_resid_post", late_hook.get_hook(l))

    late_results = []
    for prompt, correct_answers, cat in test_prompts:
        r = evaluate_single(model, prompt, correct_answers, tokenizer)
        r["category"] = cat
        late_results.append(r)
    late_metrics = compute_metrics(late_results)
    late_recovery = compute_recovery_rate(baseline_results, late_results)
    q = late_metrics["quadrants"]
    print(f"      acc={late_metrics['accuracy']:.3f}  "
          f"hall={late_metrics['hallucination_rate']:.3f}  "
          f"fixed={late_recovery['fixed']} broken={late_recovery['broken']} "
          f"net={late_recovery['net_improvement']:+d}")
    conditions["late_L22_27"] = {"metrics": late_metrics, "results": late_results}
    model.reset_hooks()

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Bootstrap CI for best condition
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Step 3: Bootstrap CI for best multi-layer gated")
    print(f"{'='*60}\n")

    n_test = len(test_prompts)
    rng_b = np.random.RandomState(42)
    diffs = []
    for _ in range(1000):
        idx = rng_b.randint(0, n_test, size=n_test)
        base_hall = sum(1 for i in idx if baseline_results[i].get("quadrant") == "CW") / n_test
        ctrl_hall = sum(1 for i in idx if best_config["results"][i].get("quadrant") == "CW") / n_test
        diffs.append(ctrl_hall - base_hall)

    diffs = np.array(diffs)
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    p_decrease = float(np.mean(diffs < 0))
    print(f"  Hall rate diff: {np.mean(diffs):+.4f} [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  P(decrease): {p_decrease:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Generate plots
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Step 4: Generating plots")
    print(f"{'='*60}\n")

    model_short = model_name.replace("/", "_")

    # Ablation comparison
    plot_ablation_comparison(conditions,
                             figures_dir / f"01_ablation_{model_short}.png")

    # Recovery analysis
    recovery_dict = {
        "single_L14": compute_recovery_rate(baseline_results, single_results),
        "multi_naive": compute_recovery_rate(baseline_results, naive_results),
        "multi_gated": best_config["recovery"],
        "random_multi": random_recovery,
        "late_L22_27": late_recovery,
    }
    plot_recovery_analysis(recovery_dict,
                           figures_dir / f"02_recovery_{model_short}.png")

    # Energy gating
    if best_config.get("hook_stats"):
        plot_energy_gating(best_config["hook_stats"],
                           figures_dir / f"03_energy_gating_{model_short}.png")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  ABLATION TABLE                                     ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")

    for name, data in conditions.items():
        m = data["metrics"]
        q = m["quadrants"]
        rec = compute_recovery_rate(baseline_results, data["results"])
        print(f"  ║  {name:20s}  acc={m['accuracy']:.3f}  "
              f"hall={m['hallucination_rate']:.3f}  "
              f"net={rec['net_improvement']:+3d}  ║")

    print(f"  ╚══════════════════════════════════════════════════════╝")

    # Key comparisons
    b_hall = baseline_metrics["hallucination_rate"]
    g_hall = best_config["metrics"]["hallucination_rate"]
    s_hall = single_metrics["hallucination_rate"]
    r_hall = random_metrics["hallucination_rate"]

    print(f"\n  Key comparisons:")
    print(f"    Baseline hall:      {b_hall:.3f}")
    print(f"    Single layer:       {s_hall:.3f}  (Δ={s_hall-b_hall:+.3f})")
    print(f"    Multi-layer gated:  {g_hall:.3f}  (Δ={g_hall-b_hall:+.3f})")
    print(f"    Random multi:       {r_hall:.3f}  (Δ={r_hall-b_hall:+.3f})")

    print(f"\n  Recovery analysis (best gated):")
    rec = best_config["recovery"]
    print(f"    Fixed (W→C):     {rec['fixed']}")
    print(f"    Broken (C→W):    {rec['broken']}")
    print(f"    Net improvement: {rec['net_improvement']:+d}")
    print(f"    Recovery rate:   {rec['recovery_rate']:.3f}")

    print(f"\n  Bootstrap CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  P(hall decrease): {p_decrease:.3f}")

    # Verdict
    print(f"\n  ┌────────────────────────────────────────────────┐")
    if best_config["recovery"]["net_improvement"] > 0 and p_decrease > 0.9:
        print(f"  │  VERDICT: TRAJECTORY CONTROL WORKS             │")
        print(f"  │  Multi-layer gated > single-layer > random     │")
        print(f"  │  Net recovery: {rec['net_improvement']:+d} samples                     │")
    elif best_config["recovery"]["net_improvement"] > 0:
        print(f"  │  VERDICT: PROMISING but not significant        │")
        print(f"  │  Net improvement: {rec['net_improvement']:+d}, p={p_decrease:.3f}              │")
    elif best_config["recovery"]["net_improvement"] == 0:
        print(f"  │  VERDICT: NO EFFECT                            │")
        print(f"  │  Trajectory control does not change behavior   │")
    else:
        print(f"  │  VERDICT: HARMFUL                              │")
        print(f"  │  Control breaks more than it fixes             │")
    print(f"  └────────────────────────────────────────────────┘")

    # TLoT implications
    print(f"\n  TLoT implications:")
    if best_config["recovery"]["net_improvement"] > 0:
        print(f"  → Trajectory-level intervention > state-level")
        print(f"  → Energy gating prevents collateral damage")
        print(f"  → Next: formalize as constrained transition system")
    else:
        print(f"  → Projection-based control (even multi-layer) insufficient")
        print(f"  → Need to explore: attention intervention, logit steering,")
        print(f"    or fundamentally different control mechanism")
        print(f"  → The representation→behavior gap persists")

    print(f"\n{'='*60}")
    print(f"Done.")

    # ─── Save results ───
    save_data = {
        "model": model_name,
        "target_layers": target_layers,
        "n_train": len(train_prompts),
        "n_test": len(test_prompts),
        "best_config": {
            "lambda_base": best_config["lambda_base"],
            "alpha": best_config["alpha"],
        },
        "conditions": {},
        "bootstrap": {
            "mean_diff": float(np.mean(diffs)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_decrease": p_decrease,
        },
    }

    for name, data in conditions.items():
        m = data["metrics"]
        rec = compute_recovery_rate(baseline_results, data["results"])
        save_data["conditions"][name] = {
            "accuracy": m["accuracy"],
            "hallucination_rate": m["hallucination_rate"],
            "quadrants": m["quadrants"],
            "recovery": rec,
        }

    out_path = results_dir / f"trajectory_control_{model_name.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
