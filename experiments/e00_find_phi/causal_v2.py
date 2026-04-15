"""
CAUSAL VALIDATION v2: With proper prompt format (Q:A:)

Pythia-410M with Q:A: format gets 75% argmax accuracy.
Now the model ACTUALLY KNOWS facts.
If Φ still works here → it's real truthfulness, not token type artifact.

Uses ONLY prompts where model baseline is meaningful.
"""

import numpy as np
import torch
import os
import sys
from scipy import stats

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
from transformer_lens import HookedTransformer


def get_qa_prompts():
    """
    Q:A: format prompts — only facts where Pythia-410M has a reasonable chance.
    Mix of known (model gets right) and unknown (model gets wrong).
    """
    return [
        # Geography Q:A format (model knows most of these)
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
        # Science (direct format works well)
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


def run(model_name="pythia-410m"):
    print("=" * 60)
    print("CAUSAL VALIDATION v2 (Q:A: format)")
    print("=" * 60)
    print(f"Model: {model_name}\n")

    model = HookedTransformer.from_pretrained(
        model_name, device='mps', dtype=torch.float16)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers

    qa_prompts = get_qa_prompts()
    test_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    # ═══ Collect data ═══
    print("--- Collecting per-prompt data ---")
    prompt_data = []
    facts_lookup = {}  # prompt -> correct_answers

    for prompt, correct_answers, category in qa_prompts:
        tokens = model.to_tokens(prompt)
        facts_lookup[prompt] = correct_answers

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: any(
                    f"blocks.{l}.hook_resid_post" in name for l in test_layers),
            )

        layer_acts = {}
        for l in test_layers:
            key = f"blocks.{l}.hook_resid_post"
            layer_acts[l] = cache[key][0, -1, :].float().cpu().numpy()

        final_logits = logits[0, -1, :].float().cpu()
        log_probs = torch.log_softmax(final_logits, dim=-1)

        top1_id = final_logits.argmax().item()
        top1_token = tokenizer.decode([top1_id])

        # Correct answer logprob
        correct_logprobs = []
        for ans in correct_answers:
            aids = tokenizer.encode(ans, add_special_tokens=False)
            if aids:
                correct_logprobs.append(log_probs[aids[0]].item())
        max_correct_lp = max(correct_logprobs) if correct_logprobs else -50.0

        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        confidence = np.exp(max_correct_lp)

        prompt_data.append({
            'prompt': prompt,
            'category': category,
            'is_correct': is_correct,
            'top_token': top1_token,
            'correct_logprob': max_correct_lp,
            'confidence': confidence,
            'layer_acts': layer_acts,
        })

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    n_correct = sum(d['is_correct'] for d in prompt_data)
    n_wrong = len(prompt_data) - n_correct
    print(f"Baseline: {n_correct}/{len(prompt_data)} = {n_correct/len(prompt_data):.1%}")
    print(f"  Correct: {n_correct}, Wrong: {n_wrong}")

    if n_correct < 5 or n_wrong < 5:
        print("ERROR: Too imbalanced. Model too weak for this test.")
        return

    # Show examples
    for d in prompt_data[:5]:
        m = "Y" if d['is_correct'] else "X"
        print(f"  [{m}] ...{d['prompt'][-40:]} -> '{d['top_token'].strip()}' "
              f"(conf={d['confidence']:.3f})")

    for layer in test_layers:
        print(f"\n{'='*60}")
        print(f"LAYER {layer} ({layer/n_layers:.0%})")
        print(f"{'='*60}")

        correct_acts = np.array([d['layer_acts'][layer] for d in prompt_data if d['is_correct']])
        wrong_acts = np.array([d['layer_acts'][layer] for d in prompt_data if not d['is_correct']])

        phi = wrong_acts.mean(0) - correct_acts.mean(0)
        phi = phi / (np.linalg.norm(phi) + 1e-8)

        # ═══ TEST 1: Logit Correlation ═══
        print(f"\n  TEST 1: Logit Correlation")
        projections = np.array([d['layer_acts'][layer] @ phi for d in prompt_data])
        confidences = np.array([d['confidence'] for d in prompt_data])
        correct_lps = np.array([d['correct_logprob'] for d in prompt_data])

        valid = np.isfinite(correct_lps)
        if valid.sum() > 5:
            r_lp, p_lp = stats.pearsonr(projections[valid], correct_lps[valid])
            r_conf, p_conf = stats.pearsonr(projections[valid], confidences[valid])
            print(f"    Φ ↔ logP(correct): r={r_lp:+.3f} (p={p_lp:.4f})")
            print(f"    Φ ↔ confidence:    r={r_conf:+.3f} (p={p_conf:.4f})")
            if abs(r_lp) > 0.7:
                print(f"    ⚠ Φ ≈ confidence")
            elif abs(r_lp) > 0.4:
                print(f"    ~ Φ partially = confidence")
            else:
                print(f"    ✓ Φ BEYOND confidence")

        # ═══ TEST 2: Sign Flip ═══
        print(f"\n  TEST 2: Sign Flip Causality")
        baseline_acc = n_correct / len(prompt_data)
        print(f"    Baseline: {baseline_acc:.3f}")

        phi_tensor = torch.tensor(phi, dtype=torch.float16, device=model.cfg.device)
        hook_name = f"blocks.{layer}.hook_resid_post"

        for lam in [1.0, 3.0, 5.0, 10.0, -1.0, -3.0, -5.0, -10.0]:
            def make_hook(phi_t, lv):
                def fn(act, hook):
                    act[:, -1, :] = act[:, -1, :] + lv * phi_t
                    return act
                return fn

            flip_correct = 0
            for d in prompt_data:
                tokens = model.to_tokens(d['prompt'])
                with torch.no_grad():
                    logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, make_hook(phi_tensor, lam))],
                    )
                new_top = tokenizer.decode([logits[0, -1, :].float().argmax().item()])
                if any(new_top.strip().lower().startswith(a.strip().lower())
                       for a in facts_lookup[d['prompt']]):
                    flip_correct += 1

            flip_acc = flip_correct / len(prompt_data)
            delta = flip_acc - baseline_acc
            direction = "ADD" if lam > 0 else "SUB"
            print(f"    λ={lam:+6.1f} ({direction}): acc={flip_acc:.3f} (Δ={delta:+.3f})")

        # ═══ TEST 3: Difficulty Stratification ═══
        print(f"\n  TEST 3: Difficulty Stratification")
        sorted_data = sorted(prompt_data, key=lambda d: d['confidence'], reverse=True)
        n_half = len(sorted_data) // 2

        for name, subset in [("Easy", sorted_data[:n_half]), ("Hard", sorted_data[n_half:])]:
            sc = [d for d in subset if d['is_correct']]
            sw = [d for d in subset if not d['is_correct']]
            print(f"    {name}: {len(sc)} correct, {len(sw)} wrong "
                  f"(mean_conf={np.mean([d['confidence'] for d in subset]):.3f})")
            if len(sc) >= 2 and len(sw) >= 2:
                ac = np.array([d['layer_acts'][layer] for d in sc])
                aw = np.array([d['layer_acts'][layer] for d in sw])
                pc = ac @ phi
                pw = aw @ phi
                t = (pc.mean() + pw.mean()) / 2
                acc = float(((pc < t).mean() + (pw >= t).mean()) / 2)
                gap = float(pw.mean() - pc.mean())
                print(f"      Φ acc={acc:.3f}, gap={gap:.3f}")
            else:
                print(f"      SKIP (imbalanced)")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline accuracy: {baseline_acc:.1%} (model actually knows facts)")
    print(f"If +Φ → acc drops AND -Φ → acc rises → CAUSAL")
    print(f"If Φ↔logP correlation LOW → BEYOND confidence")
    print(f"If works on Hard → EPISTEMIC, not just confidence")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
    run(model_name)
