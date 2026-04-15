"""
Experiment 4: Logit-Level Intervention

INSIGHT FROM E01-E03:
  - Φ exists geometrically (Cohen's d ~2.0)
  - But projection on residual stream doesn't change output
  - Because model compensates through subsequent layers
  - Late-layer intervention (L22-27) is catastrophic

NEW APPROACH:
  Instead of modifying hidden states (which get compensated),
  intervene at the LOGIT level — where there's nothing left to compensate.

  But critically: this must be Φ-INFORMED, not blind filtering.
  Otherwise it's just post-hoc censorship, not trajectory control.

THREE INTERVENTION TYPES:

  1. Φ-Informed Logit Bias:
     - Compute hallucination score from hidden state: s = h · Φ
     - Use s to modulate logit distribution
     - logits' = logits - λ * s * bias_vector

  2. Contrastive Decoding (Li et al. style):
     - Run model normally → logits_base
     - Run model WITH projection → logits_proj
     - Final logits = logits_base + α * (logits_base - logits_proj)
     - This amplifies what projection WOULD change

  3. Energy-Gated Logit Steering:
     - Compute energy E = ||Proj_Φ(h)||
     - If E > τ: logits' = logits + μ * steering_vector
     - steering_vector learned from correct vs wrong logit patterns

Usage:
    python logit_intervention.py [model_name]
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


# ─── Prompt Set ───────────────────────────────────────────────────

def get_prompts():
    """200 factual prompts."""
    prompts = [
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

        ("The official language of Brazil is", [" Portuguese", " Port"], "lang"),
        ("The official language of Japan is", [" Japanese"], "lang"),
        ("The most spoken language in the world is", [" Mandarin", " English", " Chinese"], "lang"),
        ("The currency of Japan is the", [" yen"], "lang"),
        ("The currency of the United Kingdom is the", [" pound"], "lang"),
        ("The currency of the European Union is the", [" euro"], "lang"),
        ("The currency of the United States is the", [" dollar"], "lang"),
        ("The currency of India is the", [" rupee"], "lang"),
        ("Mozart was from", [" Austria", " Salzburg"], "lang"),
        ("Beethoven was from", [" Germany", " Bonn"], "lang"),
        ("Leonardo da Vinci painted the", [" Mona", " Last"], "lang"),
        ("Michelangelo painted the", [" Sistine", " ceiling"], "lang"),

        ("Pi is approximately", [" 3"], "math"),
        ("The square root of 144 is", [" 12", " twelve"], "math"),
        ("The square root of 64 is", [" 8", " eight"], "math"),
        ("The square root of 100 is", [" 10", " ten"], "math"),
        ("A triangle has", [" 3", " three"], "math"),
        ("A hexagon has", [" 6", " six"], "math"),
        ("The sum of angles in a triangle is", [" 180"], "math"),
        ("Binary code uses only", [" 0", " two", " 1", " zeros"], "math"),
        ("A byte consists of", [" 8", " eight"], "math"),
        ("The decimal system is base", [" 10", " ten"], "math"),
        ("Roman numeral X represents", [" 10", " ten"], "math"),
        ("Roman numeral V represents", [" 5", " five"], "math"),
        ("Roman numeral C represents", [" 100", " one hundred"], "math"),
    ]
    return prompts


# ─── Core Functions ──────────────────────────────────────────────

def evaluate_with_logits(model, prompt, correct_answers, tokenizer):
    """Get full evaluation including logit vector."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)

    final_logits = logits[0, -1, :].float().cpu()
    probs = torch.softmax(final_logits, dim=-1)
    top1_id = probs.argmax().item()
    top1_prob = probs[top1_id].item()
    top1_token = tokenizer.decode([top1_id])
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    is_correct = any(
        top1_token.strip().lower().startswith(a.strip().lower())
        for a in correct_answers)

    correct_prob = 0.0
    correct_token_id = None
    for a in correct_answers:
        a_tokens = tokenizer.encode(a)
        if a_tokens:
            p = probs[a_tokens[0]].item()
            if p > correct_prob:
                correct_prob = p
                correct_token_id = a_tokens[0]

    return {
        "is_correct": is_correct,
        "top1_prob": top1_prob,
        "top1_token": top1_token,
        "top1_id": top1_id,
        "entropy": entropy,
        "correct_prob": correct_prob,
        "correct_token_id": correct_token_id,
        "logits": final_logits,  # full logit vector
    }


def compute_metrics(results):
    """Compute hallucination metrics."""
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


def compute_recovery(baseline, controlled):
    """Recovery rate analysis."""
    maintained = fixed = broken = persistent = 0
    for b, c in zip(baseline, controlled):
        bc, cc_ = b["is_correct"], c["is_correct"]
        if bc and cc_: maintained += 1
        elif bc and not cc_: broken += 1
        elif not bc and cc_: fixed += 1
        else: persistent += 1

    n_wrong = fixed + persistent
    return {
        "maintained": maintained, "broken": broken,
        "fixed": fixed, "persistent": persistent,
        "recovery_rate": fixed / max(n_wrong, 1),
        "damage_rate": broken / max(maintained + broken, 1),
        "net": fixed - broken,
    }


# ─── Method 1: Contrastive Decoding ─────────────────────────────

def contrastive_decode(model, prompt, correct_answers, tokenizer,
                       phi, target_layer, proj_lambda=0.5, alpha=1.0):
    """
    Contrastive decoding: amplify what projection changes.

    logits_final = logits_base + α * (logits_base - logits_projected)

    This doesn't modify hidden states — it uses the DIFFERENCE between
    normal and projected outputs as a signal.
    """
    tokens = model.to_tokens(prompt)

    # Base logits (no intervention)
    model.reset_hooks()
    with torch.no_grad():
        logits_base = model(tokens)[0, -1, :].float().cpu()

    # Projected logits
    phi_tensor = torch.tensor(phi, dtype=torch.float32, device=tokens.device)

    def proj_hook(value, hook):
        h = value[0, -1, :].float()
        orig_norm = h.norm()
        proj = torch.dot(h, phi_tensor)
        h_new = h - proj_lambda * proj * phi_tensor
        new_norm = h_new.norm()
        if new_norm > 1e-8:
            h_new = h_new * (orig_norm / new_norm)
        value[0, -1, :] = h_new.to(value.dtype)
        return value

    model.reset_hooks()
    model.add_hook(f"blocks.{target_layer}.hook_resid_post", proj_hook)
    with torch.no_grad():
        logits_proj = model(tokens)[0, -1, :].float().cpu()
    model.reset_hooks()

    # Contrastive: amplify the difference
    logits_final = logits_base + alpha * (logits_base - logits_proj)

    probs = torch.softmax(logits_final, dim=-1)
    top1_id = probs.argmax().item()
    top1_prob = probs[top1_id].item()
    top1_token = tokenizer.decode([top1_id])

    is_correct = any(
        top1_token.strip().lower().startswith(a.strip().lower())
        for a in correct_answers)

    return {
        "is_correct": is_correct,
        "top1_prob": top1_prob,
        "top1_token": top1_token,
        "top1_id": top1_id,
        "entropy": -(probs * torch.log(probs + 1e-10)).sum().item(),
        "correct_prob": 0.0,  # computed later if needed
    }


# ─── Method 1b: W_U @ Φ Vocab Projection (THE BRIDGE) ───────────

def wu_phi_control(model, prompt, correct_answers, tokenizer,
                   vocab_direction, phi, target_layer, lambda_scale=1.0, alpha_gate=5.0, tau=0.0):
    """
    The representation → behavior bridge.

    1. Compute vocab_direction = W_U @ Φ (maps Φ from hidden to token space)
    2. Compute state-dependent gate: λ = sigmoid(α * (Proj_Φ(h) - τ))
    3. logits' = logits - λ * vocab_direction

    This is NOT blind filtering — it's state-conditioned probability reshaping.
    The intervention strength depends on how much the hidden state
    aligns with the hallucination direction.
    """
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: f"blocks.{target_layer}.hook_resid_post" in name)

    h = cache[f"blocks.{target_layer}.hook_resid_post"][0, -1, :].float().cpu()
    phi_tensor = torch.tensor(phi, dtype=torch.float32)

    # State-dependent gating
    proj_energy = torch.dot(h, phi_tensor).abs().item()
    gate = 1.0 / (1.0 + np.exp(-alpha_gate * (proj_energy - tau)))

    # Apply: reshape logits using vocab-projected Φ
    final_logits = logits[0, -1, :].float().cpu()
    final_logits = final_logits - (gate * lambda_scale) * vocab_direction

    probs = torch.softmax(final_logits, dim=-1)
    top1_id = probs.argmax().item()
    top1_prob = probs[top1_id].item()
    top1_token = tokenizer.decode([top1_id])

    is_correct = any(
        top1_token.strip().lower().startswith(a.strip().lower())
        for a in correct_answers)

    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "is_correct": is_correct,
        "top1_prob": top1_prob,
        "top1_token": top1_token,
        "top1_id": top1_id,
        "entropy": -(probs * torch.log(probs + 1e-10)).sum().item(),
        "gate": gate,
        "proj_energy": proj_energy,
    }


# ─── Method 2: Φ-Informed Logit Bias ────────────────────────────

def phi_logit_bias(model, prompt, correct_answers, tokenizer,
                   phi, target_layer, logit_bias_vector, lambda_scale=1.0):
    """
    Use hallucination score from hidden state to bias logits.

    1. Run forward, extract h at target_layer
    2. Compute s = h · Φ (hallucination score)
    3. logits' = logits + λ * s * bias_vector

    bias_vector is learned: mean(logits|correct) - mean(logits|wrong)
    """
    tokens = model.to_tokens(prompt)
    phi_tensor = torch.tensor(phi, dtype=torch.float32, device=tokens.device)

    # Forward with cache to get hidden state
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: f"blocks.{target_layer}.hook_resid_post" in name)

    h = cache[f"blocks.{target_layer}.hook_resid_post"][0, -1, :].float()
    hall_score = torch.dot(h.cpu(), torch.tensor(phi, dtype=torch.float32)).item()

    # Bias logits proportionally to hallucination score
    final_logits = logits[0, -1, :].float().cpu()
    final_logits = final_logits + lambda_scale * hall_score * logit_bias_vector

    probs = torch.softmax(final_logits, dim=-1)
    top1_id = probs.argmax().item()
    top1_prob = probs[top1_id].item()
    top1_token = tokenizer.decode([top1_id])

    is_correct = any(
        top1_token.strip().lower().startswith(a.strip().lower())
        for a in correct_answers)

    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "is_correct": is_correct,
        "top1_prob": top1_prob,
        "top1_token": top1_token,
        "top1_id": top1_id,
        "entropy": -(probs * torch.log(probs + 1e-10)).sum().item(),
        "hall_score": hall_score,
    }


# ─── Method 3: Energy-Gated Logit Steering ──────────────────────

def energy_gated_steering(model, prompt, correct_answers, tokenizer,
                          phi_layers, tau_layers, steering_vector,
                          mu=1.0, alpha_gate=5.0):
    """
    Multi-layer energy check → logit steering.

    1. Run forward, collect h at multiple layers
    2. Compute energy E_l = |h_l · Φ_l| at each layer
    3. Aggregate: E_total = mean(sigmoid(α * (E_l - τ_l)))
    4. If E_total > 0.5: logits' = logits + μ * E_total * steering_vector

    This uses the trajectory information (multi-layer energy)
    to make a logit-level decision.
    """
    tokens = model.to_tokens(prompt)
    layers = sorted(phi_layers.keys())
    hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: any(h in name for h in hook_names))

    # Compute per-layer energy
    gate_scores = []
    for l in layers:
        h = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].float().cpu().numpy()
        phi = phi_layers[l]
        energy = abs(np.dot(h, phi))
        tau = tau_layers[l]
        gate = 1.0 / (1.0 + np.exp(-alpha_gate * (energy - tau)))
        gate_scores.append(gate)

    # Aggregate gate
    e_total = float(np.mean(gate_scores))

    # Steer logits
    final_logits = logits[0, -1, :].float().cpu()
    if e_total > 0.3:  # threshold for intervention
        final_logits = final_logits + mu * e_total * steering_vector

    probs = torch.softmax(final_logits, dim=-1)
    top1_id = probs.argmax().item()
    top1_prob = probs[top1_id].item()
    top1_token = tokenizer.decode([top1_id])

    is_correct = any(
        top1_token.strip().lower().startswith(a.strip().lower())
        for a in correct_answers)

    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "is_correct": is_correct,
        "top1_prob": top1_prob,
        "top1_token": top1_token,
        "top1_id": top1_id,
        "entropy": -(probs * torch.log(probs + 1e-10)).sum().item(),
        "e_total": e_total,
    }


# ─── Visualization ───────────────────────────────────────────────

def plot_results(conditions, save_path, title=""):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(conditions.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Accuracy
    ax = axes[0]
    vals = [conditions[n]["metrics"]["accuracy"] for n in names]
    bars = ax.bar(range(len(names)), vals, color="#2196F3", alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha='center', fontsize=8)

    # Hallucination rate
    ax = axes[1]
    vals = [conditions[n]["metrics"]["hallucination_rate"] for n in names]
    bars = ax.bar(range(len(names)), vals, color="#F44336", alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.003, f"{v:.3f}", ha='center', fontsize=8)

    # Net recovery
    ax = axes[2]
    vals = [conditions[n].get("recovery", {}).get("net", 0) for n in names]
    colors = ["#4CAF50" if v > 0 else "#F44336" if v < 0 else "#9E9E9E" for v in vals]
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel("Net (Fixed - Broken)")
    ax.set_title("Net Recovery")
    ax.axhline(y=0, color='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + (0.3 if v >= 0 else -0.5),
                f"{v:+d}", ha='center', fontsize=9)

    plt.suptitle(title or "E04: Logit-Level Intervention", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
    target_layer = 14  # from E02 causal analysis

    print(f"{'='*60}")
    print(f"E04: LOGIT-LEVEL INTERVENTION")
    print(f"Does controlling at the output change behavior?")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    print(f"Loading {model_name}...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    tokenizer = model.tokenizer
    print(f"Loaded in {time.time()-t0:.0f}s")

    all_prompts = get_prompts()

    # Train/test split
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_prompts))
    split = int(len(all_prompts) * 0.6)
    train_prompts = [all_prompts[i] for i in indices[:split]]
    test_prompts = [all_prompts[i] for i in indices[split:]]
    print(f"Train: {len(train_prompts)}, Test: {len(test_prompts)}")

    results_dir = Path(__file__).parent / "results"
    figures_dir = results_dir / "figures"

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Learn Φ and logit patterns from TRAIN set
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Step 1: Learning Φ and logit patterns (train)")
    print(f"{'='*60}\n")

    # Collect activations AND logits for train set
    train_acts = []
    train_logits_list = []
    train_labels = []

    # Multi-layer Φ for method 3
    multi_layers = list(range(10, 18))
    hook_names = [f"blocks.{l}.hook_resid_post" for l in multi_layers]

    for i, (prompt, correct_answers, cat) in enumerate(train_prompts):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(h in name for h in hook_names) or
                             f"blocks.{target_layer}.hook_resid_post" in name)

        final_logits = logits[0, -1, :].float().cpu()
        probs = torch.softmax(final_logits, dim=-1)
        top1_id = probs.argmax().item()
        top1_prob = probs[top1_id].item()
        top1_token = tokenizer.decode([top1_id])

        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        act = cache[f"blocks.{target_layer}.hook_resid_post"][0, -1, :].float().cpu().numpy()

        # Multi-layer acts
        ml_acts = {}
        for l in multi_layers:
            key = f"blocks.{l}.hook_resid_post"
            if key in cache:
                ml_acts[l] = cache[key][0, -1, :].float().cpu().numpy()

        train_acts.append(act)
        train_logits_list.append(final_logits.numpy())
        train_labels.append({
            "is_correct": is_correct,
            "top1_prob": top1_prob,
            "ml_acts": ml_acts,
        })

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (i+1) % 30 == 0:
            print(f"  [{i+1}/{len(train_prompts)}]")

    train_acts = np.array(train_acts)
    train_logits_arr = np.array(train_logits_list)

    n_correct = sum(1 for l in train_labels if l["is_correct"])
    n_wrong = sum(1 for l in train_labels if not l["is_correct"])
    print(f"  Train: {n_correct} correct, {n_wrong} wrong")

    # Learn Φ at target layer
    correct_acts = train_acts[[i for i, l in enumerate(train_labels) if l["is_correct"]]]
    wrong_acts = train_acts[[i for i, l in enumerate(train_labels) if not l["is_correct"]]]
    phi = (np.mean(wrong_acts, axis=0) - np.mean(correct_acts, axis=0))
    phi = phi / (np.linalg.norm(phi) + 1e-8)
    phi = phi.astype(np.float32)
    print(f"  Φ learned at L{target_layer}")

    # Learn logit bias vector: mean(logits|correct) - mean(logits|wrong)
    correct_logits = train_logits_arr[[i for i, l in enumerate(train_labels) if l["is_correct"]]]
    wrong_logits = train_logits_arr[[i for i, l in enumerate(train_labels) if not l["is_correct"]]]
    logit_bias = np.mean(correct_logits, axis=0) - np.mean(wrong_logits, axis=0)
    # Normalize to unit scale
    logit_bias = logit_bias / (np.linalg.norm(logit_bias) + 1e-8)
    logit_bias_tensor = torch.tensor(logit_bias, dtype=torch.float32)
    print(f"  Logit bias vector learned (||b||={np.linalg.norm(logit_bias):.3f})")

    # Learn steering vector (same as bias but from logit softmax space)
    correct_log_probs = np.log(np.clip(
        np.array([np.exp(l) / np.exp(l).sum() for l in correct_logits]), 1e-10, 1))
    wrong_log_probs = np.log(np.clip(
        np.array([np.exp(l) / np.exp(l).sum() for l in wrong_logits]), 1e-10, 1))
    steering = np.mean(correct_log_probs, axis=0) - np.mean(wrong_log_probs, axis=0)
    steering = steering / (np.linalg.norm(steering) + 1e-8)
    steering_tensor = torch.tensor(steering, dtype=torch.float32)
    print(f"  Steering vector learned")

    # Learn multi-layer Φ_l
    phi_layers = {}
    tau_layers = {}
    for l in multi_layers:
        l_acts_c = np.array([train_labels[i]["ml_acts"][l]
                             for i in range(len(train_labels))
                             if train_labels[i]["is_correct"] and l in train_labels[i]["ml_acts"]])
        l_acts_w = np.array([train_labels[i]["ml_acts"][l]
                             for i in range(len(train_labels))
                             if not train_labels[i]["is_correct"] and l in train_labels[i]["ml_acts"]])
        if len(l_acts_w) > 0 and len(l_acts_c) > 0:
            phi_l = np.mean(l_acts_w, axis=0) - np.mean(l_acts_c, axis=0)
            phi_l = phi_l / (np.linalg.norm(phi_l) + 1e-8)
            all_l_acts = np.vstack([l_acts_c, l_acts_w])
            energies = np.abs(all_l_acts @ phi_l)
            tau_layers[l] = float(np.median(energies))
        else:
            phi_l = np.zeros(model.cfg.d_model)
            tau_layers[l] = 0.0
        phi_layers[l] = phi_l.astype(np.float32)
    print(f"  Multi-layer Φ learned for L{multi_layers[0]}-L{multi_layers[-1]}")

    # THE BRIDGE: W_U @ Φ → vocab-space direction
    # This maps the hallucination direction from hidden space to token space
    W_U = model.W_U.float().cpu()  # (d_model, n_vocab)
    phi_tensor_cpu = torch.tensor(phi, dtype=torch.float32)
    vocab_direction = W_U.T @ phi_tensor_cpu  # (n_vocab,)
    vocab_dir_norm = vocab_direction.norm()
    # Don't normalize — keep the natural scale from W_U
    print(f"  W_U @ Φ computed: ||vocab_dir||={vocab_dir_norm:.3f}")

    # Energy threshold for gating
    energies_train = np.abs(train_acts @ phi)
    tau_gate = float(np.median(energies_train))
    print(f"  Gate threshold τ={tau_gate:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Evaluate all conditions on TEST set
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Step 2: Evaluating on test set ({len(test_prompts)} prompts)")
    print(f"{'='*60}\n")

    conditions = {}

    # --- Baseline ---
    print(f"  [0] Baseline...")
    model.reset_hooks()
    baseline = []
    for prompt, ca, cat in test_prompts:
        r = evaluate_with_logits(model, prompt, ca, tokenizer)
        del r["logits"]  # save memory
        baseline.append(r)
    bm = compute_metrics(baseline)
    q = bm["quadrants"]
    print(f"      acc={bm['accuracy']:.3f} hall={bm['hallucination_rate']:.3f} "
          f"[CC={q['CC']} CW={q['CW']} UC={q['UC']} UW={q['UW']}]")
    conditions["baseline"] = {"metrics": bm, "recovery": {"net": 0}}

    # --- Method 1: Contrastive Decoding ---
    print(f"\n  [1] Contrastive Decoding (sweep α)...")
    best_cd = None
    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
        cd_results = []
        for prompt, ca, cat in test_prompts:
            r = contrastive_decode(model, prompt, ca, tokenizer,
                                   phi, target_layer, proj_lambda=0.5, alpha=alpha)
            cd_results.append(r)
        cdm = compute_metrics(cd_results)
        rec = compute_recovery(baseline, cd_results)
        q = cdm["quadrants"]
        print(f"      α={alpha}: acc={cdm['accuracy']:.3f} hall={cdm['hallucination_rate']:.3f} "
              f"fixed={rec['fixed']} broken={rec['broken']} net={rec['net']:+d}")

        if best_cd is None or rec["net"] > best_cd["net"] or \
           (rec["net"] == best_cd["net"] and cdm["hallucination_rate"] < best_cd["hall"]):
            best_cd = {"alpha": alpha, "metrics": cdm, "results": cd_results,
                       "recovery": rec, "net": rec["net"], "hall": cdm["hallucination_rate"]}

    print(f"      Best: α={best_cd['alpha']}")
    conditions["contrastive"] = {
        "metrics": best_cd["metrics"],
        "recovery": best_cd["recovery"],
        "config": {"alpha": best_cd["alpha"]},
    }

    # --- Method 1b: W_U @ Φ Vocab Projection (THE BRIDGE) ---
    print(f"\n  [1b] W_U @ Φ state-conditioned control (sweep λ)...")
    best_wu = None
    for lam in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        wu_results = []
        for prompt, ca, cat in test_prompts:
            r = wu_phi_control(model, prompt, ca, tokenizer,
                               vocab_direction, phi, target_layer,
                               lambda_scale=lam, alpha_gate=5.0, tau=tau_gate)
            wu_results.append(r)
        wum = compute_metrics(wu_results)
        rec = compute_recovery(baseline, wu_results)
        q = wum["quadrants"]
        avg_gate = np.mean([r.get("gate", 0) for r in wu_results])
        print(f"      λ={lam}: acc={wum['accuracy']:.3f} hall={wum['hallucination_rate']:.3f} "
              f"fixed={rec['fixed']} broken={rec['broken']} net={rec['net']:+d} "
              f"avg_gate={avg_gate:.3f}")

        if best_wu is None or rec["net"] > best_wu["net"] or \
           (rec["net"] == best_wu["net"] and wum["hallucination_rate"] < best_wu["hall"]):
            best_wu = {"lambda": lam, "metrics": wum, "results": wu_results,
                       "recovery": rec, "net": rec["net"], "hall": wum["hallucination_rate"]}

    print(f"      Best: λ={best_wu['lambda']}")
    conditions["wu_phi_bridge"] = {
        "metrics": best_wu["metrics"],
        "recovery": best_wu["recovery"],
        "config": {"lambda": best_wu["lambda"]},
    }

    # --- Static W_U @ Φ (no gating — CONTROL for "is this just filtering?") ---
    print(f"\n  [1c] Static W_U @ Φ (no state gating — filtering control)...")
    static_results = []
    for prompt, ca, cat in test_prompts:
        tokens_s = model.to_tokens(prompt)
        with torch.no_grad():
            logits_s = model(tokens_s)[0, -1, :].float().cpu()
        logits_s = logits_s - best_wu["lambda"] * vocab_direction
        probs_s = torch.softmax(logits_s, dim=-1)
        top1_id_s = probs_s.argmax().item()
        top1_token_s = tokenizer.decode([top1_id_s])
        is_correct_s = any(top1_token_s.strip().lower().startswith(a.strip().lower()) for a in ca)
        static_results.append({
            "is_correct": is_correct_s,
            "top1_prob": probs_s[top1_id_s].item(),
            "top1_token": top1_token_s,
            "top1_id": top1_id_s,
            "entropy": -(probs_s * torch.log(probs_s + 1e-10)).sum().item(),
        })
    stm = compute_metrics(static_results)
    rec_st = compute_recovery(baseline, static_results)
    q = stm["quadrants"]
    print(f"      acc={stm['accuracy']:.3f} hall={stm['hallucination_rate']:.3f} "
          f"fixed={rec_st['fixed']} broken={rec_st['broken']} net={rec_st['net']:+d}")
    conditions["static_wu_phi"] = {"metrics": stm, "recovery": rec_st}

    # --- Method 2: Φ-Informed Logit Bias ---
    print(f"\n  [2] Φ-Informed Logit Bias (sweep λ)...")
    best_lb = None
    for lam in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        lb_results = []
        for prompt, ca, cat in test_prompts:
            r = phi_logit_bias(model, prompt, ca, tokenizer,
                               phi, target_layer, logit_bias_tensor, lambda_scale=lam)
            lb_results.append(r)
        lbm = compute_metrics(lb_results)
        rec = compute_recovery(baseline, lb_results)
        q = lbm["quadrants"]
        print(f"      λ={lam}: acc={lbm['accuracy']:.3f} hall={lbm['hallucination_rate']:.3f} "
              f"fixed={rec['fixed']} broken={rec['broken']} net={rec['net']:+d}")

        if best_lb is None or rec["net"] > best_lb["net"] or \
           (rec["net"] == best_lb["net"] and lbm["hallucination_rate"] < best_lb["hall"]):
            best_lb = {"lambda": lam, "metrics": lbm, "results": lb_results,
                       "recovery": rec, "net": rec["net"], "hall": lbm["hallucination_rate"]}

    print(f"      Best: λ={best_lb['lambda']}")
    conditions["phi_logit_bias"] = {
        "metrics": best_lb["metrics"],
        "recovery": best_lb["recovery"],
        "config": {"lambda": best_lb["lambda"]},
    }

    # --- Method 3: Energy-Gated Steering ---
    print(f"\n  [3] Energy-Gated Logit Steering (sweep μ)...")
    best_eg = None
    for mu in [0.5, 1.0, 2.0, 5.0, 10.0]:
        eg_results = []
        for prompt, ca, cat in test_prompts:
            r = energy_gated_steering(model, prompt, ca, tokenizer,
                                       phi_layers, tau_layers, steering_tensor,
                                       mu=mu, alpha_gate=5.0)
            eg_results.append(r)
        egm = compute_metrics(eg_results)
        rec = compute_recovery(baseline, eg_results)
        q = egm["quadrants"]
        print(f"      μ={mu}: acc={egm['accuracy']:.3f} hall={egm['hallucination_rate']:.3f} "
              f"fixed={rec['fixed']} broken={rec['broken']} net={rec['net']:+d}")

        if best_eg is None or rec["net"] > best_eg["net"] or \
           (rec["net"] == best_eg["net"] and egm["hallucination_rate"] < best_eg["hall"]):
            best_eg = {"mu": mu, "metrics": egm, "results": eg_results,
                       "recovery": rec, "net": rec["net"], "hall": egm["hallucination_rate"]}

    print(f"      Best: μ={best_eg['mu']}")
    conditions["energy_steering"] = {
        "metrics": best_eg["metrics"],
        "recovery": best_eg["recovery"],
        "config": {"mu": best_eg["mu"]},
    }

    # --- Control: Random logit bias ---
    print(f"\n  [4] Random logit bias (control)...")
    rng_r = np.random.RandomState(99)
    random_bias = rng_r.randn(model.cfg.d_vocab).astype(np.float32)
    random_bias = random_bias / np.linalg.norm(random_bias)
    random_bias_tensor = torch.tensor(random_bias, dtype=torch.float32)

    rl_results = []
    for prompt, ca, cat in test_prompts:
        r = phi_logit_bias(model, prompt, ca, tokenizer,
                           phi, target_layer, random_bias_tensor,
                           lambda_scale=best_lb["lambda"])
        rl_results.append(r)
    rlm = compute_metrics(rl_results)
    rec = compute_recovery(baseline, rl_results)
    q = rlm["quadrants"]
    print(f"      acc={rlm['accuracy']:.3f} hall={rlm['hallucination_rate']:.3f} "
          f"fixed={rec['fixed']} broken={rec['broken']} net={rec['net']:+d}")
    conditions["random_logit"] = {"metrics": rlm, "recovery": rec}

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Bootstrap CI
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Step 3: Bootstrap CI")
    print(f"{'='*60}\n")

    # Find best overall condition
    best_name = max([n for n in conditions if n != "baseline"],
                    key=lambda n: conditions[n]["recovery"]["net"])
    best_cond = conditions[best_name]

    # Get results for bootstrap
    if best_name == "contrastive":
        best_results = best_cd["results"]
    elif best_name == "phi_logit_bias":
        best_results = best_lb["results"]
    elif best_name == "energy_steering":
        best_results = best_eg["results"]
    else:
        best_results = baseline  # fallback

    n_test = len(test_prompts)
    rng_b = np.random.RandomState(42)
    diffs = []
    for _ in range(1000):
        idx = rng_b.randint(0, n_test, size=n_test)
        base_hall = sum(1 for i in idx if baseline[i].get("quadrant") == "CW") / n_test
        ctrl_hall = sum(1 for i in idx if best_results[i].get("quadrant") == "CW") / n_test
        diffs.append(ctrl_hall - base_hall)

    diffs = np.array(diffs)
    print(f"  Best condition: {best_name}")
    print(f"  Hall diff: {np.mean(diffs):+.4f} [{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}]")
    print(f"  P(decrease): {np.mean(diffs < 0):.3f}")

    # ═══════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Generating plots...")
    print(f"{'='*60}\n")

    model_short = model_name.replace("/", "_")
    plot_results(conditions,
                 figures_dir / f"01_logit_intervention_{model_short}.png",
                 f"E04: Logit Intervention ({model_short})")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  ABLATION TABLE                                         ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    for name, data in conditions.items():
        m = data["metrics"]
        net = data["recovery"]["net"]
        print(f"  ║  {name:20s}  acc={m['accuracy']:.3f}  "
              f"hall={m['hallucination_rate']:.3f}  net={net:+3d}  ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    # Key question
    b_hall = conditions["baseline"]["metrics"]["hallucination_rate"]
    best_hall = best_cond["metrics"]["hallucination_rate"]
    best_net = best_cond["recovery"]["net"]

    print(f"\n  KEY QUESTION: Does logit-level control reduce hallucination?")
    print(f"    Baseline:      hall={b_hall:.3f}")
    print(f"    Best ({best_name}): hall={best_hall:.3f}  net={best_net:+d}")

    if best_net > 0 and best_hall < b_hall:
        print(f"\n  ┌────────────────────────────────────────────────────┐")
        print(f"  │  YES — Logit control works where projection fails  │")
        print(f"  │  Control surface matters: output > hidden state    │")
        print(f"  │  TLoT needs logit-level π, not residual projection │")
        print(f"  └────────────────────────────────────────────────────┘")
    elif best_net >= 0 and best_hall <= b_hall:
        print(f"\n  ┌────────────────────────────────────────────────────┐")
        print(f"  │  MARGINAL — Some signal but not convincing         │")
        print(f"  │  Logit control slightly better than projection     │")
        print(f"  └────────────────────────────────────────────────────┘")
    else:
        print(f"\n  ┌────────────────────────────────────────────────────┐")
        print(f"  │  NO — Even logit control doesn't help              │")
        print(f"  │  The problem may not be solvable with post-hoc     │")
        print(f"  │  intervention on a fixed model                     │")
        print(f"  └────────────────────────────────────────────────────┘")

    # Comparison: projection vs logit
    proj_net = 0  # E03 result
    print(f"\n  PROJECTION vs LOGIT:")
    print(f"    E03 (best projection): net={proj_net:+d}")
    print(f"    E04 (best logit):      net={best_net:+d}")
    if best_net > proj_net:
        print(f"    → Logit control SUPERIOR to projection")
    elif best_net == proj_net:
        print(f"    → Both equally (in)effective")
    else:
        print(f"    → Projection was better (unexpected)")

    print(f"\n{'='*60}")
    print(f"Done.")

    # Save
    save_data = {
        "model": model_name,
        "n_train": len(train_prompts),
        "n_test": len(test_prompts),
        "conditions": {},
        "bootstrap": {
            "best_condition": best_name,
            "mean_diff": float(np.mean(diffs)),
            "ci_lower": float(np.percentile(diffs, 2.5)),
            "ci_upper": float(np.percentile(diffs, 97.5)),
            "p_decrease": float(np.mean(diffs < 0)),
        },
    }
    for name, data in conditions.items():
        save_data["conditions"][name] = {
            "accuracy": data["metrics"]["accuracy"],
            "hallucination_rate": data["metrics"]["hallucination_rate"],
            "quadrants": data["metrics"]["quadrants"],
            "recovery": data["recovery"],
            "config": data.get("config", {}),
        }

    out_path = results_dir / f"logit_intervention_{model_short}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
