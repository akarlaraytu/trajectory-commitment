"""
DIAGNOSTIC: Where is the correct token in wrong answers?

Before building E05, we need to know:
- Is the correct token in top-5? → decision competition, steering possible
- Is it in top-50? → retrieval failure, need amplification
- Is it rank 1000+? → model doesn't know, no post-hoc fix possible

This determines the ENTIRE direction of TLoT.

Usage:
    python diagnostic_rank.py [model_name]
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)


def get_prompts():
    """Full 200 prompt set."""
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


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"

    print(f"{'='*60}")
    print(f"DIAGNOSTIC: Correct token rank in wrong answers")
    print(f"{'='*60}")
    print(f"Model: {model_name}\n")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    print(f"Loading {model_name}...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    tokenizer = model.tokenizer
    print(f"Loaded in {time.time()-t0:.0f}s\n")

    prompts = get_prompts()

    # Evaluate ALL prompts
    correct_samples = []
    wrong_samples = []
    all_samples = []

    for i, (prompt, correct_answers, category) in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)

        final_logits = logits[0, -1, :].float().cpu()
        probs = torch.softmax(final_logits, dim=-1)

        # Get top-1
        top1_id = probs.argmax().item()
        top1_token = tokenizer.decode([top1_id])
        top1_prob = probs[top1_id].item()

        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        # Find best correct token and its rank
        sorted_indices = torch.argsort(probs, descending=True)
        ranks = torch.zeros(len(probs), dtype=torch.long)
        ranks[sorted_indices] = torch.arange(len(probs))

        best_correct_rank = len(probs)  # worst case
        best_correct_prob = 0.0
        best_correct_token = None
        best_correct_id = None
        best_correct_logit = None

        for a in correct_answers:
            a_tokens = tokenizer.encode(a)
            if a_tokens:
                tid = a_tokens[0]
                r = ranks[tid].item()
                p = probs[tid].item()
                if r < best_correct_rank:
                    best_correct_rank = r
                    best_correct_prob = p
                    best_correct_token = tokenizer.decode([tid])
                    best_correct_id = tid
                    best_correct_logit = final_logits[tid].item()

        # Logit gap
        top1_logit = final_logits[top1_id].item()
        logit_gap = top1_logit - (best_correct_logit if best_correct_logit is not None else 0)

        # Entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Top-10 tokens for context
        top10_ids = sorted_indices[:10].tolist()
        top10_tokens = [(tokenizer.decode([tid]), probs[tid].item()) for tid in top10_ids]

        sample = {
            "prompt": prompt,
            "category": category,
            "is_correct": is_correct,
            "top1_token": top1_token,
            "top1_prob": top1_prob,
            "top1_logit": top1_logit,
            "correct_token": best_correct_token,
            "correct_rank": best_correct_rank,
            "correct_prob": best_correct_prob,
            "correct_logit": best_correct_logit if best_correct_logit else 0,
            "logit_gap": logit_gap,
            "entropy": entropy,
            "top10": top10_tokens,
        }

        all_samples.append(sample)
        if is_correct:
            correct_samples.append(sample)
        else:
            wrong_samples.append(sample)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ─── Analysis ─────────────────────────────────────────────────
    n_total = len(all_samples)
    n_correct = len(correct_samples)
    n_wrong = len(wrong_samples)

    print(f"{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"\n  Total: {n_total}, Correct: {n_correct}, Wrong: {n_wrong}")
    print(f"  Accuracy: {n_correct/n_total:.1%}\n")

    # ─── Distribution bins ────────────────────────────────────────
    if n_wrong > 0:
        wrong_ranks = [s["correct_rank"] for s in wrong_samples]

        bins = {
            "top_1": sum(1 for r in wrong_ranks if r == 0),  # already counted as correct
            "top_2": sum(1 for r in wrong_ranks if r < 2),
            "top_5": sum(1 for r in wrong_ranks if r < 5),
            "top_10": sum(1 for r in wrong_ranks if r < 10),
            "top_50": sum(1 for r in wrong_ranks if r < 50),
            "top_100": sum(1 for r in wrong_ranks if r < 100),
            "100+": sum(1 for r in wrong_ranks if r >= 100),
        }

        print(f"  ┌────────────────────────────────────────────────┐")
        print(f"  │  CORRECT TOKEN RANK IN WRONG ANSWERS           │")
        print(f"  │  (n={n_wrong} wrong samples)                   │")
        print(f"  ├────────────────────────────────────────────────┤")
        for bin_name, count in bins.items():
            pct = count / n_wrong * 100
            bar = "█" * int(pct / 2)
            print(f"  │  {bin_name:>8s}: {count:3d}/{n_wrong}  ({pct:5.1f}%)  {bar}")
        print(f"  └────────────────────────────────────────────────┘")

        # Key metric
        recoverable = bins["top_5"]
        print(f"\n  🔑 RECOVERABLE SET (correct ∈ top-5): {recoverable}/{n_wrong} ({recoverable/n_wrong:.1%})")

        if recoverable / n_wrong < 0.10:
            print(f"  → MODEL DOESN'T KNOW. Post-hoc intervention impossible.")
        elif recoverable / n_wrong < 0.30:
            print(f"  → WEAK SIGNAL. Some competition but mostly knowledge gap.")
        elif recoverable / n_wrong < 0.60:
            print(f"  → GOLD MINE. Decision competition real. Steering possible!")
        else:
            print(f"  → STRONG COMPETITION. Model knows but picks wrong. E04 λ too weak?")

        # ─── Per-sample detail for wrong answers ──────────────────
        print(f"\n  {'='*60}")
        print(f"  PER-SAMPLE WRONG ANSWER ANALYSIS")
        print(f"  {'='*60}\n")

        # Sort by rank (most recoverable first)
        wrong_sorted = sorted(wrong_samples, key=lambda s: s["correct_rank"])

        for s in wrong_sorted:
            rank = s["correct_rank"]
            gap = s["logit_gap"]
            marker = "🟢" if rank < 5 else "🟡" if rank < 50 else "🔴"

            print(f"  {marker} rank={rank:5d}  gap={gap:+6.1f}  "
                  f"top1='{s['top1_token'].strip()}'({s['top1_prob']:.3f})  "
                  f"correct='{s['correct_token'].strip() if s['correct_token'] else '?'}'({s['correct_prob']:.4f})  "
                  f"| {s['prompt'][:55]}")

            # Show top-5 for competition analysis
            if rank < 20:
                top5_str = "  ".join([f"'{t.strip()}'={p:.3f}" for t, p in s["top10"][:5]])
                print(f"       top5: {top5_str}")

        # ─── Category breakdown ───────────────────────────────────
        print(f"\n  {'='*60}")
        print(f"  CATEGORY BREAKDOWN (wrong answers)")
        print(f"  {'='*60}\n")

        categories = {}
        for s in wrong_samples:
            cat = s["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "top5": 0, "top50": 0, "ranks": []}
            categories[cat]["total"] += 1
            categories[cat]["ranks"].append(s["correct_rank"])
            if s["correct_rank"] < 5:
                categories[cat]["top5"] += 1
            if s["correct_rank"] < 50:
                categories[cat]["top50"] += 1

        for cat, data in sorted(categories.items()):
            med_rank = int(np.median(data["ranks"]))
            print(f"  {cat:>6s}: {data['total']:2d} wrong, "
                  f"top5={data['top5']}/{data['total']}, "
                  f"top50={data['top50']}/{data['total']}, "
                  f"median_rank={med_rank}")

        # ─── Logit gap analysis ───────────────────────────────────
        print(f"\n  {'='*60}")
        print(f"  LOGIT GAP ANALYSIS (top1 - correct)")
        print(f"  {'='*60}\n")

        gaps = [s["logit_gap"] for s in wrong_samples]
        print(f"  Mean gap: {np.mean(gaps):.2f}")
        print(f"  Median gap: {np.median(gaps):.2f}")
        print(f"  Min gap: {np.min(gaps):.2f}")
        print(f"  Max gap: {np.max(gaps):.2f}")

        # How many would flip with X logit boost?
        for boost in [1, 2, 5, 10, 20]:
            flipped = sum(1 for g in gaps if g < boost)
            print(f"  Would flip with +{boost:2d} logit boost: {flipped}/{n_wrong} ({flipped/n_wrong:.1%})")

    # Also analyze CORRECT samples (for comparison)
    if n_correct > 0:
        print(f"\n  {'='*60}")
        print(f"  CORRECT ANSWER CONFIDENCE")
        print(f"  {'='*60}\n")

        correct_probs = [s["top1_prob"] for s in correct_samples]
        correct_logits = [s["top1_logit"] for s in correct_samples]
        print(f"  Mean prob: {np.mean(correct_probs):.3f}")
        print(f"  Mean logit: {np.mean(correct_logits):.2f}")
        print(f"  Min prob: {np.min(correct_probs):.3f}")
        print(f"  Min logit: {np.min(correct_logits):.2f}")

    # ─── VERDICT ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")

    if n_wrong > 0:
        pct_top5 = bins["top_5"] / n_wrong * 100
        pct_top10 = bins["top_10"] / n_wrong * 100
        pct_top50 = bins["top_50"] / n_wrong * 100
        med_gap = np.median(gaps)

        print(f"\n  Correct in top-5:  {pct_top5:.0f}%")
        print(f"  Correct in top-10: {pct_top10:.0f}%")
        print(f"  Correct in top-50: {pct_top50:.0f}%")
        print(f"  Median logit gap:  {med_gap:.1f}")

        print(f"\n  DIAGNOSIS:")
        if pct_top5 >= 60:
            print(f"  → COMPETITION: Model knows the answer but picks wrong.")
            print(f"  → Decision boundary steering IS viable.")
            print(f"  → E04 failed because intervention was too weak/diffuse.")
        elif pct_top10 >= 40:
            print(f"  → PARTIAL KNOWLEDGE: Answer is nearby but not dominant.")
            print(f"  → Targeted amplification could work.")
        elif pct_top50 >= 30:
            print(f"  → WEAK RETRIEVAL: Model has some knowledge but buried.")
            print(f"  → Need stronger amplification, not just steering.")
        else:
            print(f"  → KNOWLEDGE GAP: Model genuinely doesn't know.")
            print(f"  → No post-hoc intervention can fix this.")
            print(f"  → Need: fine-tuning, RAG, or bigger model.")

        print(f"\n  FOR TLoT:")
        if pct_top5 >= 30:
            print(f"  → Hallucination here IS a decision competition problem.")
            print(f"  → E05 should do TARGETED token steering (top-k only).")
        else:
            print(f"  → Hallucination here is a knowledge problem, not control.")
            print(f"  → TLoT cannot fix what the model doesn't know.")
    else:
        print(f"\n  No wrong answers — model is 100% correct!")

    print(f"\n{'='*60}")
    print(f"Done.")

    # Save
    save_data = {
        "model": model_name,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "accuracy": n_correct / n_total,
        "wrong_analysis": [{
            "prompt": s["prompt"],
            "category": s["category"],
            "top1_token": s["top1_token"],
            "correct_token": s["correct_token"],
            "correct_rank": s["correct_rank"],
            "correct_prob": s["correct_prob"],
            "logit_gap": s["logit_gap"],
        } for s in wrong_samples] if wrong_samples else [],
        "bins": bins if n_wrong > 0 else {},
    }

    out_path = Path(__file__).parent / "results" / f"diagnostic_rank_{model_name.replace('/', '_')}.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
