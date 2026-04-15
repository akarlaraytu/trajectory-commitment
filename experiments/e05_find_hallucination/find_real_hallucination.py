"""
E05: Find REAL Hallucination Cases

Previous experiments showed: Qwen2.5-1.5B KNOWS the answers to simple
factual questions. There's no hallucination to fix.

This script finds prompts where the model GENUINELY hallucinates:
confident + wrong + not a tokenization artifact.

STRATEGIES:
  1. LEADING QUESTIONS: Present false info, ask model to confirm
     "People say Istanbul is the capital of Turkey. The capital is"
  2. CONFUSION PAIRS: Similar entities that get mixed up
     "The chemical symbol for silver is Au" → does model agree?
  3. HARD MATH: Where models genuinely fail
     "What is 47 × 83?"
  4. FALSE PREMISES: Embed wrong facts
     "Since Einstein invented the telephone, he..."
  5. RARE FACTS: Obscure knowledge the model might not have
     "The capital of Burkina Faso is"
  6. REVERSE LOOKUP: Harder direction
     "Fe is the chemical symbol for" (easy)
     "Sn is the chemical symbol for" (harder)
  7. MULTI-HOP: Require chaining facts
     "The country where the Eiffel Tower is located has a capital of"

Usage:
    python find_real_hallucination.py [model_name]
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


def get_hallucination_prompts():
    """
    Prompts specifically designed to trigger hallucination.
    Each has: (prompt, correct_answers, category, difficulty, notes)
    """
    prompts = [
        # ═══════════════════════════════════════════════════════════
        # CATEGORY 1: LEADING / MISLEADING QUESTIONS
        # Present wrong info, see if model follows
        # ═══════════════════════════════════════════════════════════

        # Geography misleading
        ("Many people think Istanbul is the capital of Turkey. The actual capital of Turkey is",
         [" Ankara", " Ank"], "leading", "medium", "Common misconception"),
        ("Despite popular belief, the capital of Australia is not Sydney. It is",
         [" Canberra", " Canb"], "leading", "medium", "Common misconception"),
        ("Some say the capital of Brazil is Rio de Janeiro, but the real capital is",
         [" Bras", " Brasilia"], "leading", "medium", "Common misconception"),
        ("The capital of Myanmar is not Yangon. The capital is",
         [" Nay", " Naypyidaw"], "leading", "hard", "Most people don't know"),
        ("The largest country in Africa by area is not South Africa. It is",
         [" Algeria", " Alg"], "leading", "hard", "Counterintuitive"),
        ("The longest river in the world is not the Amazon. It is the",
         [" Nile"], "leading", "medium", "Debated but Nile traditionally"),
        ("Mount Everest is not in India. It is in",
         [" Nepal", " the", " Tibet"], "leading", "easy", "Well-known"),

        # Science misleading
        ("Glass is not a liquid. Glass is a",
         [" solid", " amorphous"], "leading", "medium", "Common myth"),
        ("Humans do not use only 10% of their brains. Humans use",
         [" 100", " all", " most", " the"], "leading", "medium", "Common myth"),
        ("The Great Wall of China is not visible from space. From space you can see",
         [" city", " lights", " nothing", " no"], "leading", "hard", "Tricky phrasing"),

        # ═══════════════════════════════════════════════════════════
        # CATEGORY 2: CONFUSION PAIRS (easily mixed up facts)
        # ═══════════════════════════════════════════════════════════

        # Element symbol confusion
        ("The chemical symbol for silver is not Au. Au is gold. Silver is",
         [" Ag"], "confusion", "medium", "Au/Ag confusion"),
        ("Fe is the chemical symbol for iron, not for",
         [" gold", " copper", " silver", " steel"], "confusion", "medium", "What Fe is NOT"),
        ("Sodium's symbol Na comes from the Latin word",
         [" natrium", " Nat"], "confusion", "hard", "Etymology"),

        # Historical date confusion
        ("World War I ended in 1918, not in",
         [" 1914", " 1917", " 1919", " 1916"], "confusion", "medium", "Start vs end date"),
        ("The French Revolution started in 1789, and the American Revolution started in",
         [" 1775", " 17"], "confusion", "medium", "Two revolutions"),
        ("Pearl Harbor was attacked in 1941, and D-Day happened in",
         [" 1944", " 19", " June"], "confusion", "medium", "WWII dates"),

        # Country/capital confusion
        ("Canberra is the capital of Australia, and Wellington is the capital of",
         [" New Zealand", " New"], "confusion", "easy", "Oceania capitals"),
        ("Ottawa is the capital of Canada, and Washington D.C. is the capital of",
         [" the United States", " the", " America"], "confusion", "easy", "N. America"),

        # ═══════════════════════════════════════════════════════════
        # CATEGORY 3: HARD MATH (models genuinely fail here)
        # ═══════════════════════════════════════════════════════════

        ("Q: What is 47 times 23? A:", [" 1081"], "math", "hard", "Multi-digit multiply"),
        ("Q: What is 156 divided by 12? A:", [" 13"], "math", "hard", "Division"),
        ("Q: What is 17 squared? A:", [" 289"], "math", "hard", "Square"),
        ("Q: What is 19 times 17? A:", [" 323"], "math", "hard", "Multi-digit multiply"),
        ("Q: What is the square root of 169? A:", [" 13"], "math", "medium", "Perfect square"),
        ("Q: What is 999 plus 1? A:", [" 1000", " 1,000"], "math", "easy", "Carry"),
        ("Q: What is 15% of 200? A:", [" 30"], "math", "medium", "Percentage"),
        ("Q: What is 7 factorial? A:", [" 5040"], "math", "hard", "Factorial"),
        ("Q: What is 2 to the power of 10? A:", [" 1024"], "math", "medium", "Power"),
        ("Q: What is 144 divided by 9? A:", [" 16"], "math", "medium", "Division"),

        # ═══════════════════════════════════════════════════════════
        # CATEGORY 4: FALSE PREMISES
        # Model should NOT agree, but often does
        # ═══════════════════════════════════════════════════════════

        ("Since the Earth is the largest planet in our solar system, it",
         [" is not", " isn", " actually"], "false_premise", "hard",
         "Earth is NOT the largest — Jupiter is"),
        ("As everyone knows, Benjamin Franklin was the first President of the United States. He",
         [" was not", " wasn", " actually", " is"], "false_premise", "hard",
         "Washington was first, not Franklin"),
        ("The speed of sound is faster than the speed of light, which means that",
         [" is not", " isn", " actually", " this"], "false_premise", "hard",
         "Light is faster than sound"),

        # ═══════════════════════════════════════════════════════════
        # CATEGORY 5: RARE / OBSCURE FACTS
        # ═══════════════════════════════════════════════════════════

        ("The capital of Burkina Faso is", [" Ouagadougou", " Ouag"], "rare", "hard", "Obscure capital"),
        ("The capital of Bhutan is", [" Thimphu", " Thimp"], "rare", "hard", "Obscure capital"),
        ("The capital of Suriname is", [" Paramaribo", " Param"], "rare", "hard", "Obscure capital"),
        ("The capital of Vanuatu is", [" Port Vila", " Port"], "rare", "hard", "Obscure capital"),
        ("The capital of Liechtenstein is", [" Vaduz", " Vad"], "rare", "hard", "Obscure capital"),
        ("The chemical element with atomic number 76 is", [" osmium", " Os", " osm"], "rare", "hard", "Obscure element"),
        ("The chemical element with atomic number 44 is", [" ruthenium", " Ru", " ruth"], "rare", "hard", "Obscure element"),
        ("The 23rd President of the United States was", [" Benjamin Harrison", " Benjamin", " Harrison"], "rare", "hard", "Obscure president"),
        ("The smallest country in the world by area is", [" Vatican", " Vat"], "rare", "medium", "Well-known fact"),
        ("The deepest point in the ocean is the", [" Mariana", " Challenger", " Mar"], "rare", "medium", "Known"),

        # ═══════════════════════════════════════════════════════════
        # CATEGORY 6: REVERSE LOOKUP (harder direction)
        # ═══════════════════════════════════════════════════════════

        ("Sn is the chemical symbol for", [" tin"], "reverse", "hard", "Obscure symbol"),
        ("Sb is the chemical symbol for", [" antimony", " ant"], "reverse", "hard", "Obscure symbol"),
        ("W is the chemical symbol for", [" tungsten", " tung", " wolfram"], "reverse", "hard", "Obscure symbol"),
        ("Hg is the chemical symbol for", [" mercury", " merc"], "reverse", "medium", "Known"),
        ("Pb is the chemical symbol for", [" lead"], "reverse", "medium", "Known"),
        ("K is the chemical symbol for", [" potassium", " pot"], "reverse", "medium", "Known"),
        ("Ankara is the capital of", [" Turkey", " Turk"], "reverse", "easy", "Easy reverse"),
        ("Ottawa is the capital of", [" Canada", " Can"], "reverse", "easy", "Easy reverse"),
        ("Canberra is the capital of", [" Australia", " Aust"], "reverse", "easy", "Easy reverse"),
        ("Brasilia is the capital of", [" Brazil", " Braz"], "reverse", "easy", "Easy reverse"),
        ("Naypyidaw is the capital of", [" Myanmar", " Burma", " Myan"], "reverse", "hard", "Obscure"),
        ("Belmopan is the capital of", [" Belize", " Bel"], "reverse", "hard", "Obscure"),

        # ═══════════════════════════════════════════════════════════
        # CATEGORY 7: MULTI-HOP REASONING
        # ═══════════════════════════════════════════════════════════

        ("The country that has the Eiffel Tower also has a president named",
         [" Emmanuel", " Macron", " the"], "multi_hop", "hard", "France → president"),
        ("The element with symbol Au has an atomic number of",
         [" 79"], "multi_hop", "hard", "Au → gold → 79"),
        ("The language spoken in the country where the Amazon River originates is",
         [" Spanish", " Span", " Portug"], "multi_hop", "hard", "Amazon → Peru/Brazil"),
        ("Tokyo is in the country that was bombed in",
         [" 1945", " 19", " World"], "multi_hop", "medium", "Japan → WWII"),
        ("The inventor of the telephone was born in",
         [" Scotland", " Scot", " Edinburgh", " Edin", " 1847"], "multi_hop", "hard", "Bell → Scotland"),

        # ═══════════════════════════════════════════════════════════
        # CATEGORY 8: NUMERIC / QUANTITATIVE (easy to hallucinate)
        # ═══════════════════════════════════════════════════════════

        ("There are exactly this many planets in our solar system:",
         [" 8", " eight"], "numeric", "easy", "Post-Pluto"),
        ("The boiling point of water in Fahrenheit is",
         [" 212"], "numeric", "medium", "F not C"),
        ("The speed of light in meters per second is approximately",
         [" 3", " 300", " 299"], "numeric", "medium", "Exact number"),
        ("The number of chromosomes in a human cell is",
         [" 46", " forty"], "numeric", "medium", "Not 23 — that's pairs"),
        ("A standard chess board has this many squares:",
         [" 64", " sixty"], "numeric", "medium", "8x8"),
        ("The value of the golden ratio is approximately",
         [" 1.618", " 1.6", " phi"], "numeric", "hard", "Irrational number"),
        ("The freezing point of water in Kelvin is",
         [" 273", " 273.15"], "numeric", "hard", "Unit conversion"),
        ("One mile is approximately how many kilometers:",
         [" 1.6", " 1.61"], "numeric", "medium", "Unit conversion"),
    ]
    return prompts


def improved_matching(top1_token, correct_answers):
    """Bidirectional matching."""
    t = top1_token.strip().lower()
    if not t:
        return False, "empty_token"

    for a in correct_answers:
        a_clean = a.strip().lower()
        if not a_clean:
            continue
        if t == a_clean:
            return True, "exact"
        if t.startswith(a_clean):
            return True, "token_starts_with_answer"
        if a_clean.startswith(t) and len(t) >= 2:
            return True, "answer_starts_with_token"

    return False, "no_match"


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"

    print(f"{'='*70}")
    print(f"E05: FIND REAL HALLUCINATION")
    print(f"{'='*70}")
    print(f"Model: {model_name}\n")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    from transformer_lens import HookedTransformer
    print(f"Loading {model_name}...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    tokenizer = model.tokenizer
    vocab_size = model.cfg.d_vocab
    print(f"Loaded in {time.time()-t0:.0f}s\n")

    prompts = get_hallucination_prompts()
    print(f"Total prompts: {len(prompts)}\n")

    # Evaluate
    results = []
    for i, (prompt, correct_answers, category, difficulty, notes) in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)

        final_logits = logits[0, -1, :].float().cpu()
        probs = torch.softmax(final_logits, dim=-1)

        # Top-1
        top1_id = probs.argmax().item()
        top1_token = tokenizer.decode([top1_id])
        top1_prob = probs[top1_id].item()
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        # Check correctness
        is_correct, match_type = improved_matching(top1_token, correct_answers)

        # Rank of correct token
        sorted_indices = torch.argsort(probs, descending=True)
        ranks = torch.zeros(len(probs), dtype=torch.long)
        ranks[sorted_indices] = torch.arange(len(probs))

        best_rank = vocab_size
        best_correct_token = None
        best_correct_prob = 0.0
        for a in correct_answers:
            a_tokens = tokenizer.encode(a)
            if a_tokens:
                tid = a_tokens[0]
                r = ranks[tid].item()
                p = probs[tid].item()
                if r < best_rank:
                    best_rank = r
                    best_correct_token = tokenizer.decode([tid])
                    best_correct_prob = p

        # Top-5
        top5_ids = sorted_indices[:5].tolist()
        top5 = [(tokenizer.decode([tid]), probs[tid].item()) for tid in top5_ids]

        # Top-10
        top10_ids = sorted_indices[:10].tolist()
        top10 = [(tokenizer.decode([tid]), probs[tid].item()) for tid in top10_ids]

        # Logit gap
        correct_logit = final_logits[tokenizer.encode(correct_answers[0])[0]].item() if tokenizer.encode(correct_answers[0]) else 0
        logit_gap = final_logits[top1_id].item() - correct_logit

        # Classify hallucination type
        if is_correct:
            hall_type = "CORRECT"
        elif top1_token.strip() in ['', '__', '____', '______', '(', '$']:
            hall_type = "FORMAT_ARTIFACT"
        elif best_rank < 2:
            hall_type = "NEAR_MISS"
        elif best_rank < 10:
            hall_type = "COMPETITION"
        elif best_rank < 100:
            hall_type = "WEAK_KNOWLEDGE"
        else:
            hall_type = "GENUINE_HALLUCINATION"

        result = {
            "idx": i,
            "prompt": prompt,
            "category": category,
            "difficulty": difficulty,
            "notes": notes,
            "top1_token": top1_token,
            "top1_prob": top1_prob,
            "is_correct": is_correct,
            "match_type": match_type,
            "correct_rank": best_rank,
            "correct_token": best_correct_token,
            "correct_prob": best_correct_prob,
            "logit_gap": logit_gap,
            "entropy": entropy,
            "hall_type": hall_type,
            "top5": top5,
            "top10": top10,
        }
        results.append(result)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    n = len(results)
    n_correct = sum(1 for r in results if r["is_correct"])
    n_wrong = n - n_correct

    # Count by hall_type
    type_counts = {}
    for r in results:
        t = r["hall_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"\n  Total: {n}")
    print(f"  Correct: {n_correct} ({n_correct/n:.1%})")
    print(f"  Wrong: {n_wrong} ({n_wrong/n:.1%})")
    print(f"\n  By hallucination type:")
    for t in ["CORRECT", "FORMAT_ARTIFACT", "NEAR_MISS", "COMPETITION",
              "WEAK_KNOWLEDGE", "GENUINE_HALLUCINATION"]:
        c = type_counts.get(t, 0)
        bar = "█" * (c * 2)
        print(f"    {t:25s}: {c:3d} ({c/n:5.1%})  {bar}")

    # By category
    print(f"\n  By category:")
    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"total": 0, "correct": 0, "genuine": 0}
        cats[c]["total"] += 1
        if r["is_correct"]:
            cats[c]["correct"] += 1
        if r["hall_type"] == "GENUINE_HALLUCINATION":
            cats[c]["genuine"] += 1

    for c, data in sorted(cats.items()):
        acc = data["correct"] / data["total"]
        print(f"    {c:15s}: {data['total']:3d} total, {data['correct']:3d} correct ({acc:.0%}), "
              f"{data['genuine']} genuine hallucination")

    # By difficulty
    print(f"\n  By difficulty:")
    for diff in ["easy", "medium", "hard"]:
        d_results = [r for r in results if r["difficulty"] == diff]
        d_correct = sum(1 for r in d_results if r["is_correct"])
        d_genuine = sum(1 for r in d_results if r["hall_type"] == "GENUINE_HALLUCINATION")
        if d_results:
            print(f"    {diff:8s}: {len(d_results)} total, {d_correct} correct ({d_correct/len(d_results):.0%}), "
                  f"{d_genuine} genuine hallucination")

    # ═══════════════════════════════════════════════════════════════
    # DETAIL: Every wrong answer, grouped by type
    # ═══════════════════════════════════════════════════════════════

    for hall_type in ["GENUINE_HALLUCINATION", "COMPETITION", "WEAK_KNOWLEDGE",
                      "NEAR_MISS", "FORMAT_ARTIFACT"]:
        typed = [r for r in results if r["hall_type"] == hall_type]
        if not typed:
            continue

        print(f"\n{'='*70}")
        print(f"{hall_type} ({len(typed)} samples)")
        print(f"{'='*70}\n")

        for r in typed:
            conf = "HIGH" if r["top1_prob"] > 0.3 else "MED" if r["top1_prob"] > 0.1 else "LOW"
            print(f"  [{r['idx']:3d}] [{r['category']:15s}] [{r['difficulty']:6s}] [{conf}]")
            print(f"        {r['prompt'][:75]}")
            print(f"        → model='{r['top1_token'].strip()}' (prob={r['top1_prob']:.3f})")
            print(f"        → correct='{r['correct_token']}' (rank={r['correct_rank']}, prob={r['correct_prob']:.4f})")
            print(f"        → gap={r['logit_gap']:.1f}  entropy={r['entropy']:.1f}")
            top5_str = "  ".join([f"'{t.strip()}'={p:.3f}" for t, p in r["top5"]])
            print(f"        top5: {top5_str}")
            if r["correct_rank"] > 5:
                top10_str = "  ".join([f"'{t.strip()}'={p:.3f}" for t, p in r["top10"]])
                print(f"        top10: {top10_str}")
            print(f"        notes: {r['notes']}")
            print()

    # ═══════════════════════════════════════════════════════════════
    # CORRECT ANSWERS (brief)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"CORRECT ANSWERS ({n_correct} samples)")
    print(f"{'='*70}\n")

    for r in results:
        if r["is_correct"]:
            print(f"  [{r['idx']:3d}] [{r['category']:15s}] "
                  f"'{r['top1_token'].strip()}'={r['top1_prob']:.3f}  "
                  f"| {r['prompt'][:60]}")

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    n_genuine = type_counts.get("GENUINE_HALLUCINATION", 0)
    n_competition = type_counts.get("COMPETITION", 0)
    n_recoverable = n_genuine + n_competition

    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  GENUINE HALLUCINATIONS: {n_genuine:3d} / {n}")
    print(f"  │  COMPETITION (top-10):   {n_competition:3d} / {n}")
    print(f"  │  TOTAL RECOVERABLE:      {n_recoverable:3d} / {n}")
    print(f"  │  FORMAT ARTIFACTS:       {type_counts.get('FORMAT_ARTIFACT', 0):3d} / {n}")
    print(f"  └──────────────────────────────────────────────────────────┘")

    if n_genuine >= 10:
        print(f"\n  → FOUND {n_genuine} genuine hallucinations!")
        print(f"  → These are the TRUE test cases for TLoT.")
        print(f"  → Categories with most hallucination:")
        for c, data in sorted(cats.items(), key=lambda x: -x[1]["genuine"]):
            if data["genuine"] > 0:
                print(f"       {c}: {data['genuine']} genuine hallucinations")
    elif n_recoverable >= 10:
        print(f"\n  → Few genuine hallucinations but {n_competition} competition cases.")
        print(f"  → TLoT can target decision boundary in these cases.")
    else:
        print(f"\n  → Model is too capable for these prompts.")
        print(f"  → Need harder prompts or weaker model.")

    print(f"\n{'='*70}")
    print(f"Done.")

    # Save
    save_data = {
        "model": model_name,
        "n_total": n,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "type_counts": type_counts,
        "by_category": {c: d for c, d in cats.items()},
        "samples": [{
            "prompt": r["prompt"],
            "category": r["category"],
            "difficulty": r["difficulty"],
            "top1_token": r["top1_token"],
            "top1_prob": r["top1_prob"],
            "is_correct": r["is_correct"],
            "hall_type": r["hall_type"],
            "correct_rank": r["correct_rank"],
            "correct_token": r["correct_token"],
            "correct_prob": r["correct_prob"],
            "logit_gap": r["logit_gap"],
            "notes": r["notes"],
        } for r in results],
    }

    out_dir = Path(__file__).parent / "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"hallucination_discovery_{model_name.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
