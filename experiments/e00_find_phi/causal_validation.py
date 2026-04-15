"""
FINAL 3 TESTS: Is Φ truthfulness or just confidence?

TEST 1 — Logit Correlation
  Proj_Φ(h) vs log P(correct token)
  High correlation → Φ ≈ confidence (not necessarily bad, but simpler)
  Low correlation → Φ captures something beyond raw confidence

TEST 2 — Sign Flip Causality
  h' = h + λv → does adding Φ direction INCREASE wrong answers?
  If yes → Φ is CAUSAL, not just correlational
  If no → Φ is epiphenomenal

TEST 3 — Difficulty Stratification
  Easy facts (model very confident) vs Hard facts (model uncertain)
  If Φ works only on easy → confidence signal
  If Φ works on hard too → epistemic signal (deeper)
"""

import numpy as np
import torch
import os
import sys
from pathlib import Path
from dataclasses import dataclass

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'

from transformer_lens import HookedTransformer


@dataclass
class FactPrompt:
    prompt: str
    correct_answers: list
    category: str


def get_fact_prompts():
    return [
        FactPrompt("The capital of France is", [" Paris"], "geography"),
        FactPrompt("The capital of Germany is", [" Berlin"], "geography"),
        FactPrompt("The capital of Japan is", [" Tokyo"], "geography"),
        FactPrompt("The capital of Italy is", [" Rome"], "geography"),
        FactPrompt("The capital of Spain is", [" Madrid"], "geography"),
        FactPrompt("The capital of Australia is", [" Canberra"], "geography"),
        FactPrompt("The capital of Canada is", [" Ottawa"], "geography"),
        FactPrompt("The capital of Russia is", [" Moscow"], "geography"),
        FactPrompt("The capital of China is", [" Beijing"], "geography"),
        FactPrompt("The capital of India is", [" New", " Delhi"], "geography"),
        FactPrompt("The capital of Turkey is", [" Ankara"], "geography"),
        FactPrompt("The capital of Egypt is", [" Cairo"], "geography"),
        FactPrompt("The capital of Brazil is", [" Bras"], "geography"),
        FactPrompt("The capital of South Korea is", [" Seoul"], "geography"),
        FactPrompt("The capital of Mexico is", [" Mexico"], "geography"),
        FactPrompt("The capital of Poland is", [" Warsaw"], "geography"),
        FactPrompt("The capital of Sweden is", [" Stockholm"], "geography"),
        FactPrompt("The capital of Greece is", [" Athens"], "geography"),
        FactPrompt("The capital of Argentina is", [" Buenos"], "geography"),
        FactPrompt("The capital of Norway is", [" Oslo"], "geography"),
        FactPrompt("The chemical symbol for gold is", [" Au"], "science"),
        FactPrompt("The chemical symbol for iron is", [" Fe"], "science"),
        FactPrompt("The chemical symbol for silver is", [" Ag"], "science"),
        FactPrompt("The Earth orbits the", [" Sun"], "science"),
        FactPrompt("The Moon orbits the", [" Earth"], "science"),
        FactPrompt("Diamonds are made of", [" carbon"], "science"),
        FactPrompt("The speed of light is approximately", [" 3", " 300"], "science"),
        FactPrompt("The atomic number of oxygen is", [" 8", " eight"], "science"),
        FactPrompt("The atomic number of hydrogen is", [" 1", " one"], "science"),
        FactPrompt("The atomic number of carbon is", [" 6", " six"], "science"),
        FactPrompt("Albert Einstein developed the theory of", [" relat"], "science"),
        FactPrompt("Isaac Newton discovered the law of", [" grav"], "science"),
        FactPrompt("Charles Darwin proposed the theory of", [" evol", " natural"], "science"),
        FactPrompt("The closest star to Earth is the", [" Sun"], "science"),
        FactPrompt("The pH of pure water is", [" 7", " seven"], "science"),
        FactPrompt("Electrons have a", [" negative"], "science"),
        FactPrompt("Photosynthesis produces", [" oxygen", " glucose"], "science"),
        FactPrompt("Water boils at 100 degrees", [" Celsius", " C"], "science"),
        FactPrompt("The periodic table was created by", [" Mend", " Dmitri"], "science"),
        FactPrompt("Sound cannot travel through a", [" vacuum"], "science"),
        FactPrompt("World War II ended in", [" 1945", " 19"], "history"),
        FactPrompt("The Berlin Wall fell in", [" 1989", " 19"], "history"),
        FactPrompt("The first moon landing was in", [" 1969", " 19"], "history"),
        FactPrompt("Columbus reached the Americas in", [" 1492", " 14"], "history"),
        FactPrompt("The Declaration of Independence was signed in", [" 1776", " 17"], "history"),
        FactPrompt("The French Revolution began in", [" 1789", " 17"], "history"),
        FactPrompt("The Titanic sank in", [" 1912", " 19", " April"], "history"),
        FactPrompt("Shakespeare wrote", [" Hamlet", " Romeo", " Mac", " plays"], "history"),
        FactPrompt("Napoleon was defeated at", [" Water"], "history"),
        FactPrompt("The Soviet Union dissolved in", [" 1991", " 19"], "history"),
        FactPrompt("The Industrial Revolution started in", [" Britain", " England", " the"], "history"),
        FactPrompt("Alexander the Great was from", [" Mac", " Greece"], "history"),
        FactPrompt("The Wright brothers invented the", [" airplane", " air", " first"], "history"),
        FactPrompt("The printing press was invented by", [" Gut", " Johannes"], "history"),
        FactPrompt("The Magna Carta was signed in", [" 12"], "history"),
        FactPrompt("Martin Luther King Jr. gave his famous", [" \"", " I", " speech"], "history"),
        FactPrompt("The Cold War was between the", [" United", " US", " Soviet"], "history"),
        FactPrompt("Julius Caesar was assassinated in", [" 44"], "history"),
        FactPrompt("The Renaissance began in", [" Italy", " the"], "history"),
        FactPrompt("The Great Wall of China was built", [" to", " during", " over"], "history"),
    ]


def run_causal_validation(model_name="pythia-410m"):
    print("=" * 60)
    print("CAUSAL VALIDATION: 3 Final Tests")
    print("=" * 60)
    print(f"Model: {model_name}")

    model = HookedTransformer.from_pretrained(
        model_name, device='mps',
        dtype=torch.float16,
    )
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers

    facts = get_fact_prompts()

    # Test on the best middle-late layer
    test_layers = [n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print(f"Testing layers: {test_layers}")

    # ═══ STEP 0: Collect per-prompt data ═══
    print(f"\n--- Collecting per-prompt data ---")

    prompt_data = []  # list of dicts with all info per prompt

    for fp in facts:
        tokens = model.to_tokens(fp.prompt)

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(
                    f"blocks.{l}.hook_resid_post" in name for l in test_layers
                ),
            )

        # Decision point activations per layer
        layer_acts = {}
        for l in test_layers:
            key = f"blocks.{l}.hook_resid_post"
            layer_acts[l] = cache[key][0, -1, :].float().cpu().numpy()

        # Logits at last position
        final_logits = logits[0, -1, :].float().cpu()
        log_probs = torch.log_softmax(final_logits, dim=-1)

        # Top prediction
        top_id = final_logits.argmax().item()
        top_token = tokenizer.decode([top_id])
        top_logprob = log_probs[top_id].item()

        # Log prob of correct answer
        correct_logprobs = []
        for ans in fp.correct_answers:
            ans_ids = tokenizer.encode(ans, add_special_tokens=False)
            if ans_ids:
                correct_logprobs.append(log_probs[ans_ids[0]].item())
        max_correct_logprob = max(correct_logprobs) if correct_logprobs else -float('inf')

        # Is top prediction correct?
        is_correct = any(
            top_token.strip().lower().startswith(ans.strip().lower())
            for ans in fp.correct_answers
        )

        # Confidence = P(correct token)
        confidence = np.exp(max_correct_logprob)

        # Entropy of top-10 distribution (uncertainty)
        top10_probs = torch.softmax(final_logits, dim=-1).topk(10).values
        entropy = -float((top10_probs * torch.log(top10_probs + 1e-10)).sum())

        prompt_data.append({
            'prompt': fp.prompt,
            'category': fp.category,
            'is_correct': is_correct,
            'top_token': top_token,
            'top_logprob': top_logprob,
            'correct_logprob': max_correct_logprob,
            'confidence': confidence,
            'entropy': entropy,
            'layer_acts': layer_acts,
        })

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    n_correct = sum(1 for d in prompt_data if d['is_correct'])
    n_wrong = len(prompt_data) - n_correct
    print(f"Total: {len(prompt_data)} prompts ({n_correct} correct, {n_wrong} wrong)")

    # Show some examples
    print(f"\nExamples:")
    for d in prompt_data[:10]:
        mark = "Y" if d['is_correct'] else "X"
        print(f"  [{mark}] {d['prompt']}... -> '{d['top_token'].strip()}' "
              f"(conf={d['confidence']:.3f}, H={d['entropy']:.2f})")

    for layer in test_layers:
        print(f"\n{'='*60}")
        print(f"LAYER {layer} ({layer/n_layers:.0%})")
        print(f"{'='*60}")

        # Build Φ direction
        correct_acts = np.array([d['layer_acts'][layer] for d in prompt_data if d['is_correct']])
        wrong_acts = np.array([d['layer_acts'][layer] for d in prompt_data if not d['is_correct']])

        if len(correct_acts) < 3 or len(wrong_acts) < 3:
            print("  SKIP: too few samples")
            continue

        phi_dir = wrong_acts.mean(0) - correct_acts.mean(0)
        phi_dir = phi_dir / (np.linalg.norm(phi_dir) + 1e-8)

        # ═══ TEST 1: Logit Correlation ═══
        print(f"\n--- TEST 1: Logit Correlation ---")
        print(f"  Proj_Φ(h) vs log P(correct token)")

        projections = []
        correct_logprobs = []
        confidences = []
        entropies = []

        for d in prompt_data:
            act = d['layer_acts'][layer]
            proj = float(act @ phi_dir)
            projections.append(proj)
            correct_logprobs.append(d['correct_logprob'])
            confidences.append(d['confidence'])
            entropies.append(d['entropy'])

        projections = np.array(projections)
        correct_logprobs = np.array(correct_logprobs)
        confidences = np.array(confidences)
        entropies = np.array(entropies)

        # Pearson correlation
        from scipy import stats

        # Filter out -inf logprobs
        valid = np.isfinite(correct_logprobs)
        if valid.sum() > 5:
            corr_logprob, p_logprob = stats.pearsonr(projections[valid], correct_logprobs[valid])
            corr_conf, p_conf = stats.pearsonr(projections[valid], confidences[valid])
            corr_entropy, p_entropy = stats.pearsonr(projections[valid], entropies[valid])

            print(f"  Φ ↔ log P(correct):  r = {corr_logprob:+.3f}  (p = {p_logprob:.4f})")
            print(f"  Φ ↔ confidence:      r = {corr_conf:+.3f}  (p = {p_conf:.4f})")
            print(f"  Φ ↔ entropy:         r = {corr_entropy:+.3f}  (p = {p_entropy:.4f})")

            if abs(corr_logprob) > 0.7:
                print(f"  ⚠ HIGH correlation with logit → Φ ≈ confidence signal")
            elif abs(corr_logprob) > 0.4:
                print(f"  ~ MODERATE correlation → Φ partially captures confidence")
            else:
                print(f"  ✓ LOW correlation → Φ captures something BEYOND confidence")
        else:
            print(f"  Cannot compute (too few valid logprobs)")
            corr_logprob = 0

        # ═══ TEST 2: Sign Flip Causality ═══
        print(f"\n--- TEST 2: Sign Flip Causality ---")
        print(f"  h' = h + λ·Φ → does adding Φ increase wrong answers?")

        lambdas = [0.5, 1.0, 2.0, 5.0, 10.0]
        baseline_correct = 0
        total = 0

        # Baseline: normal model
        for d in prompt_data:
            if d['is_correct']:
                baseline_correct += 1
            total += 1
        baseline_acc = baseline_correct / total
        print(f"  Baseline accuracy: {baseline_acc:.3f}")

        for lam in lambdas:
            # Hook that adds λ·Φ to the residual stream at this layer
            phi_tensor = torch.tensor(phi_dir, dtype=torch.float16, device=model.cfg.device)

            flip_correct = 0
            flip_total = 0

            def make_hook(phi_t, lambda_val):
                def hook_fn(act, hook):
                    # act shape: (batch, seq, d_model)
                    # Add Φ direction to last token position
                    act[:, -1, :] = act[:, -1, :] + lambda_val * phi_t
                    return act
                return hook_fn

            hook_name = f"blocks.{layer}.hook_resid_post"

            for d in prompt_data:
                tokens = model.to_tokens(d['prompt'])

                hook_fn = make_hook(phi_tensor, lam)

                with torch.no_grad():
                    logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, hook_fn)],
                    )

                # Check if top prediction changed
                new_top_id = logits[0, -1, :].float().argmax().item()
                new_top_token = tokenizer.decode([new_top_id])

                new_correct = any(
                    new_top_token.strip().lower().startswith(ans.strip().lower())
                    for ans in [fp.correct_answers for fp in facts
                                if fp.prompt == d['prompt']][0]
                )
                if new_correct:
                    flip_correct += 1
                flip_total += 1

            flip_acc = flip_correct / flip_total
            delta = flip_acc - baseline_acc
            print(f"  λ={lam:5.1f}: acc={flip_acc:.3f} (Δ={delta:+.3f})")

        # Also test NEGATIVE direction (subtracting Φ should help)
        print(f"\n  Subtracting Φ (should IMPROVE accuracy):")
        for lam in [1.0, 5.0, 10.0]:
            phi_tensor = torch.tensor(phi_dir, dtype=torch.float16, device=model.cfg.device)

            sub_correct = 0
            sub_total = 0

            for d in prompt_data:
                tokens = model.to_tokens(d['prompt'])

                hook_fn = make_hook(phi_tensor, -lam)  # SUBTRACT

                with torch.no_grad():
                    logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, hook_fn)],
                    )

                new_top_id = logits[0, -1, :].float().argmax().item()
                new_top_token = tokenizer.decode([new_top_id])

                new_correct = any(
                    new_top_token.strip().lower().startswith(ans.strip().lower())
                    for ans in [fp.correct_answers for fp in facts
                                if fp.prompt == d['prompt']][0]
                )
                if new_correct:
                    sub_correct += 1
                sub_total += 1

            sub_acc = sub_correct / sub_total
            delta = sub_acc - baseline_acc
            print(f"  λ={-lam:+5.1f}: acc={sub_acc:.3f} (Δ={delta:+.3f})")

        # ═══ TEST 3: Difficulty Stratification ═══
        print(f"\n--- TEST 3: Difficulty Stratification ---")

        # Sort prompts by confidence (P(correct token))
        sorted_data = sorted(prompt_data, key=lambda d: d['confidence'], reverse=True)

        n_half = len(sorted_data) // 2
        easy_data = sorted_data[:n_half]
        hard_data = sorted_data[n_half:]

        print(f"  Easy (top {n_half} by confidence):")
        easy_correct = [d for d in easy_data if d['is_correct']]
        easy_wrong = [d for d in easy_data if not d['is_correct']]
        print(f"    {len(easy_correct)} correct, {len(easy_wrong)} wrong")
        print(f"    Mean confidence: {np.mean([d['confidence'] for d in easy_data]):.3f}")

        print(f"  Hard (bottom {len(hard_data)} by confidence):")
        hard_correct = [d for d in hard_data if d['is_correct']]
        hard_wrong = [d for d in hard_data if not d['is_correct']]
        print(f"    {len(hard_correct)} correct, {len(hard_wrong)} wrong")
        print(f"    Mean confidence: {np.mean([d['confidence'] for d in hard_data]):.3f}")

        # Φ accuracy on easy vs hard
        for subset_name, subset_correct, subset_wrong in [
            ("Easy", easy_correct, easy_wrong),
            ("Hard", hard_correct, hard_wrong),
        ]:
            if len(subset_correct) < 2 or len(subset_wrong) < 2:
                print(f"    {subset_name}: SKIP (too few in one class)")
                continue

            sc = np.array([d['layer_acts'][layer] for d in subset_correct])
            sw = np.array([d['layer_acts'][layer] for d in subset_wrong])

            # Use the GLOBAL phi direction (not recomputed)
            proj_c = sc @ phi_dir
            proj_w = sw @ phi_dir
            thresh = (proj_c.mean() + proj_w.mean()) / 2
            acc = float(((proj_c < thresh).mean() + (proj_w >= thresh).mean()) / 2)
            gap = float(proj_w.mean() - proj_c.mean())

            print(f"    {subset_name}: mean-diff acc = {acc:.3f}, gap = {gap:.3f}")

        # ═══ LAYER VERDICT ═══
        print(f"\n--- LAYER {layer} CAUSAL VERDICT ---")

        # Summarize
        is_confidence = abs(corr_logprob) > 0.7 if 'corr_logprob' in dir() else False
        print(f"  Logit correlation:  r = {corr_logprob:+.3f} "
              f"({'≈confidence' if is_confidence else 'beyond confidence'})")

    print(f"\n{'='*60}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*60}")
    print("""
  If sign flip is CAUSAL (adding Φ → accuracy drops):
    → Φ CAUSES wrong outputs, not just correlates
    → This is the strongest evidence for TLoT

  If Φ ≈ confidence (high logit correlation):
    → Still useful! Confidence IS related to truthfulness
    → But less novel — need to show Φ adds beyond P(token)

  If Φ works on HARD facts too:
    → Epistemic signal, not just "easy = confident = correct"
    → This is the TLoT sweet spot

  BEST CASE: causal + works on hard + low logit correlation
    → Φ is a genuine epistemic state direction
    → Publishable finding, proceed to e01
""")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
    run_causal_validation(model_name)
