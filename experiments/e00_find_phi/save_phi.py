"""
Save Φ direction as .npy for use in Experiment 1.

Computes Φ = mean(wrong_activations) - mean(correct_activations) at a given layer,
using Q:A: format prompts where model actually knows facts.

Usage:
    python save_phi.py [model_name] [layer]
    python save_phi.py pythia-410m 12
"""

import numpy as np
import torch
import os
import sys

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'
from transformer_lens import HookedTransformer


def get_qa_prompts():
    """Same prompts as causal_v2.py"""
    return [
        # Geography Q:A format
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
        # Science
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


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
    target_layer = int(sys.argv[2]) if len(sys.argv) > 2 else 12

    print(f"Computing Φ direction for {model_name} at layer {target_layer}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16)
    tokenizer = model.tokenizer

    qa_prompts = get_qa_prompts()

    correct_acts = []
    wrong_acts = []

    for prompt, correct_answers, category in qa_prompts:
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: f"blocks.{target_layer}.hook_resid_post" in name,
            )

        key = f"blocks.{target_layer}.hook_resid_post"
        act = cache[key][0, -1, :].float().cpu().numpy()

        final_logits = logits[0, -1, :].float().cpu()
        top1_id = final_logits.argmax().item()
        top1_token = tokenizer.decode([top1_id])

        is_correct = any(
            top1_token.strip().lower().startswith(a.strip().lower())
            for a in correct_answers)

        if is_correct:
            correct_acts.append(act)
        else:
            wrong_acts.append(act)

        del cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"Correct: {len(correct_acts)}, Wrong: {len(wrong_acts)}")

    if len(correct_acts) < 5 or len(wrong_acts) < 5:
        print("ERROR: Too imbalanced to compute meaningful Φ")
        sys.exit(1)

    correct_acts = np.array(correct_acts)
    wrong_acts = np.array(wrong_acts)

    # Φ = mean(wrong) - mean(correct), normalized
    phi = wrong_acts.mean(0) - correct_acts.mean(0)
    phi = phi / (np.linalg.norm(phi) + 1e-8)

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    safe_name = model_name.replace("/", "_")
    out_path = os.path.join(results_dir, f"phi_direction_layer{target_layer}_{safe_name}.npy")
    np.save(out_path, phi.astype(np.float32))

    print(f"\nΦ direction saved to: {out_path}")
    print(f"  Shape: {phi.shape}")
    print(f"  Norm: {np.linalg.norm(phi):.6f}")
    print(f"  Baseline: {len(correct_acts)}/{len(qa_prompts)} correct ({len(correct_acts)/len(qa_prompts):.1%})")


if __name__ == "__main__":
    main()
