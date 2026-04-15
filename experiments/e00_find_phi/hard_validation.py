"""
HARD VALIDATION: Is Φ real or prompt-level artifact?

Three brutal tests:

TEST 1 — Prompt-Level Holdout
  Train on 40 prompts, test on 20 UNSEEN prompts.
  If Φ is real: unseen prompts should also separate.
  If prompt artifact: accuracy drops to chance.

TEST 2 — Paraphrase Generalization
  Train on "The capital of France is"
  Test on "France's capital city is"
  Same fact, different phrasing.
  If Φ is real: still works.
  If surface pattern: fails.

TEST 3 — Label Shuffle Control
  Shuffle correct/wrong labels randomly, retrain.
  If original accuracy >> shuffle accuracy: signal is real.
  If similar: artifact.

SUCCESS CRITERION:
  Unseen prompt accuracy > 0.65 AND shuffle control < 0.55
  → Φ is real, proceed to e01
"""

import numpy as np
import torch
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ['TRANSFORMERLENS_ALLOW_MPS'] = '1'


@dataclass
class FactPrompt:
    prompt: str
    correct_answers: list
    category: str
    paraphrase: str = ""  # alternative phrasing of same fact


def get_fact_prompts_with_paraphrases():
    """Facts with alternative phrasings for paraphrase test."""
    return [
        # Geography — Original + Paraphrase
        FactPrompt("The capital of France is", [" Paris"], "geography",
                    "France's capital city is"),
        FactPrompt("The capital of Germany is", [" Berlin"], "geography",
                    "Germany's capital city is"),
        FactPrompt("The capital of Japan is", [" Tokyo"], "geography",
                    "Japan's capital city is"),
        FactPrompt("The capital of Italy is", [" Rome"], "geography",
                    "Italy's capital city is"),
        FactPrompt("The capital of Spain is", [" Madrid"], "geography",
                    "Spain's capital city is"),
        FactPrompt("The capital of Australia is", [" Canberra"], "geography",
                    "Australia's capital city is"),
        FactPrompt("The capital of Canada is", [" Ottawa"], "geography",
                    "Canada's capital city is"),
        FactPrompt("The capital of Russia is", [" Moscow"], "geography",
                    "Russia's capital city is"),
        FactPrompt("The capital of China is", [" Beijing"], "geography",
                    "China's capital city is"),
        FactPrompt("The capital of India is", [" New", " Delhi"], "geography",
                    "India's capital city is"),
        FactPrompt("The capital of Turkey is", [" Ankara"], "geography",
                    "Turkey's capital city is"),
        FactPrompt("The capital of Egypt is", [" Cairo"], "geography",
                    "Egypt's capital city is"),
        FactPrompt("The capital of Brazil is", [" Bras"], "geography",
                    "Brazil's capital city is"),
        FactPrompt("The capital of South Korea is", [" Seoul"], "geography",
                    "South Korea's capital city is"),
        FactPrompt("The capital of Mexico is", [" Mexico"], "geography",
                    "Mexico's capital city is"),
        FactPrompt("The capital of Poland is", [" Warsaw"], "geography",
                    "Poland's capital city is"),
        FactPrompt("The capital of Sweden is", [" Stockholm"], "geography",
                    "Sweden's capital city is"),
        FactPrompt("The capital of Greece is", [" Athens"], "geography",
                    "Greece's capital city is"),
        FactPrompt("The capital of Argentina is", [" Buenos"], "geography",
                    "Argentina's capital city is"),
        FactPrompt("The capital of Norway is", [" Oslo"], "geography",
                    "Norway's capital city is"),
        # Science — Original + Paraphrase
        FactPrompt("The chemical symbol for gold is", [" Au"], "science",
                    "Gold's chemical symbol is"),
        FactPrompt("The chemical symbol for iron is", [" Fe"], "science",
                    "Iron's chemical symbol is"),
        FactPrompt("The chemical symbol for silver is", [" Ag"], "science",
                    "Silver's chemical symbol is"),
        FactPrompt("The Earth orbits the", [" Sun"], "science",
                    "Our planet orbits the"),
        FactPrompt("The Moon orbits the", [" Earth"], "science",
                    "Earth's natural satellite orbits the"),
        FactPrompt("Diamonds are made of", [" carbon"], "science",
                    "The element that makes up diamonds is"),
        FactPrompt("The speed of light is approximately", [" 3", " 300"], "science",
                    "Light travels at roughly"),
        FactPrompt("The atomic number of oxygen is", [" 8", " eight"], "science",
                    "Oxygen's atomic number is"),
        FactPrompt("The atomic number of hydrogen is", [" 1", " one"], "science",
                    "Hydrogen's atomic number is"),
        FactPrompt("The atomic number of carbon is", [" 6", " six"], "science",
                    "Carbon's atomic number is"),
        FactPrompt("Albert Einstein developed the theory of", [" relat"], "science",
                    "The theory of relativity was developed by"),
        FactPrompt("Isaac Newton discovered the law of", [" grav"], "science",
                    "The law of gravity was discovered by"),
        FactPrompt("Charles Darwin proposed the theory of", [" evol", " natural"], "science",
                    "The theory of evolution was proposed by"),
        FactPrompt("The closest star to Earth is the", [" Sun"], "science",
                    "Earth's nearest star is the"),
        FactPrompt("The pH of pure water is", [" 7", " seven"], "science",
                    "Pure water has a pH of"),
        FactPrompt("Electrons have a", [" negative"], "science",
                    "The charge of an electron is"),
        FactPrompt("Photosynthesis produces", [" oxygen", " glucose", " energy"], "science",
                    "The process of photosynthesis creates"),
        FactPrompt("Water boils at 100 degrees", [" Celsius", " C"], "science",
                    "The boiling point of water is 100 degrees"),
        FactPrompt("The periodic table was created by", [" Mend", " Dmitri"], "science",
                    "Mendeleev is known for creating the"),
        FactPrompt("Sound cannot travel through a", [" vacuum"], "science",
                    "In a vacuum, sound"),
        # History — Original + Paraphrase
        FactPrompt("World War II ended in", [" 1945", " 19"], "history",
                    "The year World War II ended was"),
        FactPrompt("The Berlin Wall fell in", [" 1989", " 19"], "history",
                    "The year the Berlin Wall came down was"),
        FactPrompt("The first moon landing was in", [" 1969", " 19"], "history",
                    "Humans first walked on the moon in"),
        FactPrompt("Columbus reached the Americas in", [" 1492", " 14"], "history",
                    "The year Columbus arrived in the Americas was"),
        FactPrompt("The Declaration of Independence was signed in", [" 1776", " 17"], "history",
                    "America declared independence in"),
        FactPrompt("The French Revolution began in", [" 1789", " 17"], "history",
                    "The year the French Revolution started was"),
        FactPrompt("The Titanic sank in", [" 1912", " 19", " April"], "history",
                    "The year the Titanic went down was"),
        FactPrompt("Shakespeare wrote", [" Hamlet", " Romeo", " Mac", " plays"], "history",
                    "Famous plays by Shakespeare include"),
        FactPrompt("Napoleon was defeated at", [" Water"], "history",
                    "The battle where Napoleon lost was"),
        FactPrompt("The Soviet Union dissolved in", [" 1991", " 19"], "history",
                    "The year the USSR collapsed was"),
        FactPrompt("The Industrial Revolution started in", [" Britain", " England", " the"], "history",
                    "The country where the Industrial Revolution began was"),
        FactPrompt("Alexander the Great was from", [" Mac", " Greece"], "history",
                    "The homeland of Alexander the Great was"),
        FactPrompt("The Wright brothers invented the", [" airplane", " air", " first"], "history",
                    "The airplane was invented by the"),
        FactPrompt("The printing press was invented by", [" Gut", " Johannes"], "history",
                    "Gutenberg is famous for inventing the"),
        FactPrompt("The Magna Carta was signed in", [" 12"], "history",
                    "The year the Magna Carta was signed was"),
        FactPrompt("Martin Luther King Jr. gave his famous", [" \"", " I", " speech"], "history",
                    "The famous speech by Martin Luther King Jr. is"),
        FactPrompt("The Cold War was between the", [" United", " US", " Soviet"], "history",
                    "The two superpowers in the Cold War were the"),
        FactPrompt("Julius Caesar was assassinated in", [" 44"], "history",
                    "The year Julius Caesar was killed was"),
        FactPrompt("The Renaissance began in", [" Italy", " the"], "history",
                    "The country where the Renaissance started was"),
        FactPrompt("The Great Wall of China was built", [" to", " during", " over"], "history",
                    "Construction of the Great Wall of China"),
    ]


def get_activations_for_prompts(model, prompts, correct_answers_list,
                                 n_samples=30, temperature=1.5, layer=None):
    """
    For each prompt, sample completions and get decision-point activation.
    Returns: activations (n_prompts, d_model), labels (n_prompts,) based on
    whether the TOP completion is correct.

    KEY INSIGHT: One activation per prompt. Label = is the model's most likely
    completion correct?
    """
    tokenizer = model.tokenizer
    device = model.cfg.device

    if layer is None:
        layer = model.cfg.n_layers // 2  # middle layer

    activations = []
    labels = []

    for prompt, correct_answers in zip(prompts, correct_answers_list):
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name, l=layer: f"blocks.{l}.hook_resid_post" in name,
            )

            # Decision point activation (last prompt token)
            key = f"blocks.{layer}.hook_resid_post"
            act = cache[key][0, -1, :].float().cpu()
            activations.append(act)

            del cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Get model's top prediction
        with torch.no_grad():
            logits = model(tokens)[:, -1, :]
            top_token_id = logits[0].float().argmax().item()
            top_token = tokenizer.decode([top_token_id])

        # Label: is the model's top prediction correct?
        is_correct = any(
            top_token.strip().lower().startswith(ans.strip().lower())
            for ans in correct_answers
        )
        labels.append(is_correct)

    return torch.stack(activations).numpy(), np.array(labels)


def run_hard_validation(model_name="pythia-410m"):
    from transformer_lens import HookedTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("=" * 60)
    print("HARD VALIDATION: Is Φ real or artifact?")
    print("=" * 60)
    print(f"Model: {model_name}")

    model = HookedTransformer.from_pretrained(
        model_name, device='mps',
        dtype=torch.float16,
    )

    facts = get_fact_prompts_with_paraphrases()
    n_layers = model.cfg.n_layers

    # Test multiple layers including middle ones
    test_layers = [
        n_layers // 4,       # early-mid
        n_layers // 2,       # middle
        3 * n_layers // 4,   # mid-late (sweet spot?)
        n_layers - 1,        # exit door
    ]

    print(f"\nTesting layers: {test_layers}")
    print(f"Total fact prompts: {len(facts)}")

    for layer in test_layers:
        print(f"\n{'='*60}")
        print(f"LAYER {layer} (position: {layer/n_layers:.0%})")
        print(f"{'='*60}")

        # ═══ TEST 1: Prompt-Level Holdout ═══
        print(f"\n--- TEST 1: Prompt-Level Holdout ---")

        # Shuffle and split prompts
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(facts))
        n_train = 40
        n_test = len(facts) - n_train

        train_facts = [facts[i] for i in indices[:n_train]]
        test_facts = [facts[i] for i in indices[n_train:]]

        # Get activations + labels for train prompts
        train_prompts = [f.prompt for f in train_facts]
        train_answers = [f.correct_answers for f in train_facts]
        train_acts, train_labels = get_activations_for_prompts(
            model, train_prompts, train_answers, layer=layer)

        # Get activations + labels for test prompts (UNSEEN)
        test_prompts = [f.prompt for f in test_facts]
        test_answers = [f.correct_answers for f in test_facts]
        test_acts, test_labels = get_activations_for_prompts(
            model, test_prompts, test_answers, layer=layer)

        print(f"  Train: {len(train_acts)} prompts "
              f"({train_labels.sum()} correct, {(~train_labels).sum()} wrong)")
        print(f"  Test:  {len(test_acts)} prompts "
              f"({test_labels.sum()} correct, {(~test_labels).sum()} wrong)")

        # Check class balance
        if train_labels.sum() < 3 or (~train_labels).sum() < 3:
            print(f"  SKIP: Too imbalanced for meaningful test")
            continue
        if test_labels.sum() < 2 or (~test_labels).sum() < 2:
            print(f"  SKIP: Test set too imbalanced")
            continue

        # Mean-diff direction from TRAIN only
        train_correct = train_acts[train_labels]
        train_wrong = train_acts[~train_labels]

        mean_diff_dir = train_wrong.mean(0) - train_correct.mean(0)
        mean_diff_dir = mean_diff_dir / (np.linalg.norm(mean_diff_dir) + 1e-8)

        # Test on UNSEEN prompts with mean-diff
        test_correct = test_acts[test_labels]
        test_wrong = test_acts[~test_labels]

        proj_c = test_correct @ mean_diff_dir
        proj_w = test_wrong @ mean_diff_dir
        threshold = (proj_c.mean() + proj_w.mean()) / 2
        holdout_mean_acc = float(
            ((proj_c < threshold).mean() + (proj_w >= threshold).mean()) / 2
        )

        # LDA on train, test on unseen
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_acts)
        y_train = train_labels.astype(float)
        X_test = scaler.transform(test_acts)
        y_test = test_labels.astype(float)

        clf = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(X_train, y_train)
        holdout_lda_acc = float(clf.score(X_test, y_test))

        print(f"  Holdout mean-diff acc: {holdout_mean_acc:.3f}")
        print(f"  Holdout LDA acc:       {holdout_lda_acc:.3f}")

        # ═══ TEST 2: Paraphrase Generalization ═══
        print(f"\n--- TEST 2: Paraphrase Generalization ---")

        # Train on original phrasings, test on paraphrases
        original_prompts = [f.prompt for f in facts if f.paraphrase]
        original_answers = [f.correct_answers for f in facts if f.paraphrase]
        paraphrase_prompts = [f.paraphrase for f in facts if f.paraphrase]
        paraphrase_answers = [f.correct_answers for f in facts if f.paraphrase]

        orig_acts, orig_labels = get_activations_for_prompts(
            model, original_prompts, original_answers, layer=layer)
        para_acts, para_labels = get_activations_for_prompts(
            model, paraphrase_prompts, paraphrase_answers, layer=layer)

        print(f"  Original:   {len(orig_acts)} prompts "
              f"({orig_labels.sum()} correct, {(~orig_labels).sum()} wrong)")
        print(f"  Paraphrase: {len(para_acts)} prompts "
              f"({para_labels.sum()} correct, {(~para_labels).sum()} wrong)")

        if orig_labels.sum() < 3 or (~orig_labels).sum() < 3:
            print(f"  SKIP: Too imbalanced")
            continue

        # Mean-diff from originals → test on paraphrases
        orig_correct = orig_acts[orig_labels]
        orig_wrong = orig_acts[~orig_labels]

        para_dir = orig_wrong.mean(0) - orig_correct.mean(0)
        para_dir = para_dir / (np.linalg.norm(para_dir) + 1e-8)

        if para_labels.sum() >= 2 and (~para_labels).sum() >= 2:
            para_correct = para_acts[para_labels]
            para_wrong = para_acts[~para_labels]

            pp_c = para_correct @ para_dir
            pp_w = para_wrong @ para_dir
            p_thresh = (pp_c.mean() + pp_w.mean()) / 2
            para_mean_acc = float(
                ((pp_c < p_thresh).mean() + (pp_w >= p_thresh).mean()) / 2
            )

            # LDA
            scaler2 = StandardScaler()
            clf2 = LogisticRegression(max_iter=5000, C=1.0)
            clf2.fit(scaler2.fit_transform(orig_acts), orig_labels.astype(float))
            para_lda_acc = float(clf2.score(scaler2.transform(para_acts), para_labels.astype(float)))

            print(f"  Paraphrase mean-diff acc: {para_mean_acc:.3f}")
            print(f"  Paraphrase LDA acc:       {para_lda_acc:.3f}")
        else:
            print(f"  SKIP: Paraphrase labels too imbalanced")
            para_mean_acc = None
            para_lda_acc = None

        # ═══ TEST 3: Label Shuffle Control ═══
        print(f"\n--- TEST 3: Label Shuffle Control ---")

        shuffle_accs = []
        rng2 = np.random.RandomState(42)
        all_acts = np.vstack([train_acts, test_acts])
        all_labels = np.concatenate([train_labels, test_labels])

        for trial in range(50):
            shuffled = all_labels.copy()
            rng2.shuffle(shuffled)

            # Split
            s_train = all_acts[:n_train]
            s_test = all_acts[n_train:]
            y_tr = shuffled[:n_train].astype(float)
            y_te = shuffled[n_train:].astype(float)

            if y_tr.sum() < 2 or (1-y_tr).sum() < 2:
                continue
            if y_te.sum() < 2 or (1-y_te).sum() < 2:
                continue

            sc = StandardScaler()
            cr = LogisticRegression(max_iter=2000, C=1.0)
            cr.fit(sc.fit_transform(s_train), y_tr)
            shuffle_accs.append(cr.score(sc.transform(s_test), y_te))

        if shuffle_accs:
            shuffle_mean = np.mean(shuffle_accs)
            shuffle_std = np.std(shuffle_accs)
            print(f"  Shuffle control acc: {shuffle_mean:.3f} +/- {shuffle_std:.3f}")
            print(f"  Real LDA acc:        {holdout_lda_acc:.3f}")
            print(f"  Gap (real - shuffle): {holdout_lda_acc - shuffle_mean:.3f}")
        else:
            shuffle_mean = 0.5
            print(f"  Shuffle control: could not compute (imbalanced)")

        # ═══ LAYER VERDICT ═══
        print(f"\n--- LAYER {layer} VERDICT ---")

        real_signal = holdout_lda_acc > 0.65 or holdout_mean_acc > 0.65
        shuffle_ok = shuffle_mean < 0.55 if shuffle_accs else True
        para_ok = (para_mean_acc is not None and para_mean_acc > 0.6) or \
                  (para_lda_acc is not None and para_lda_acc > 0.6)

        if real_signal and shuffle_ok and para_ok:
            print(f"  ✓ STRONG: Φ is REAL at this layer")
            print(f"    Survives holdout + paraphrase + shuffle control")
        elif real_signal and shuffle_ok:
            print(f"  ~ MODERATE: Holdout works but paraphrase uncertain")
        elif real_signal:
            print(f"  ? SUSPICIOUS: High accuracy but shuffle also high")
        else:
            print(f"  ✗ FAIL: No signal at this layer")

    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")
    print("Check results above for each layer.")
    print("If ANY middle layer shows STRONG → Φ is real → proceed to e01")
    print("If all FAIL or SUSPICIOUS → need bigger model or different approach")


if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "pythia-410m"
    run_hard_validation(model_name)
