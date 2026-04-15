"""
Experiment 0: Does Φ Exist?

The most important experiment in the entire TLoT program.
If we can't find a stable hallucination direction in activation space,
nothing else matters.

This script:
1. Generates correct and incorrect statements (arithmetic OR factual)
2. Runs them through a model, recording residual stream activations
3. Computes contrastive directions (mean_wrong - mean_correct) per layer
4. Tests stability via split-half cosine similarity
5. Tests separability via linear probe accuracy
6. Tests generalization across categories
7. **CONTROL: compares real direction vs random direction**
8. **Generates diagnostic plots for every metric**

Usage:
    python find_phi.py --model pythia-410m --task factual
    python find_phi.py --model pythia-1.4b --task arithmetic
    python find_phi.py --model meta-llama/Meta-Llama-3-8B --backend nnsight --task factual
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# ─── Data Generation ───────────────────────────────────────────────

@dataclass
class ArithmeticSample:
    prompt: str
    is_correct: bool
    operation: str  # "addition", "multiplication", "subtraction"
    answer: int
    wrong_answer: Optional[int] = None


def generate_arithmetic_data(
    n_per_type: int = 200,
    operations: list[str] = None,
    seed: int = 42,
) -> list[ArithmeticSample]:
    """
    Generate paired correct/incorrect arithmetic statements.
    For each correct statement, we generate a corresponding wrong one
    with the same operands but a plausible but incorrect answer.
    """
    if operations is None:
        operations = ["addition", "multiplication", "subtraction"]

    rng = random.Random(seed)
    samples = []

    for op in operations:
        for _ in range(n_per_type):
            a = rng.randint(2, 99)
            b = rng.randint(2, 99)

            if op == "addition":
                correct_ans = a + b
                symbol = "+"
            elif op == "multiplication":
                correct_ans = a * b
                symbol = "*"
            elif op == "subtraction":
                correct_ans = a - b
                symbol = "-"
            else:
                raise ValueError(f"Unknown operation: {op}")

            offset = rng.choice([-3, -2, -1, 1, 2, 3, 5, 7, 10])
            wrong_ans = correct_ans + offset

            samples.append(ArithmeticSample(
                prompt=f"{a} {symbol} {b} = {correct_ans}",
                is_correct=True, operation=op, answer=correct_ans,
            ))
            samples.append(ArithmeticSample(
                prompt=f"{a} {symbol} {b} = {wrong_ans}",
                is_correct=False, operation=op,
                answer=correct_ans, wrong_answer=wrong_ans,
            ))

    return samples


# ─── Activation Extraction ─────────────────────────────────────────

def extract_activations_transformerlens(
    model_name: str,
    prompts: list[str],
    layers: list[int] = None,
    device: str = "auto",
    extract_all_positions: bool = False,
) -> dict[int, torch.Tensor]:
    """
    Extract residual stream activations using TransformerLens.

    Returns: {layer_idx: tensor of shape (n_prompts, d_model)}
             or (n_prompts, seq_len, d_model) if extract_all_positions=True
    We take the activation at the LAST token position by default.
    """
    import transformer_lens as tl

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading {model_name} on {device}...")
    model = tl.HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    if layers is None:
        layers = list(range(model.cfg.n_layers))

    n_prompts = len(prompts)
    d_model = model.cfg.d_model
    activations = {l: [] for l in layers}

    batch_size = 32
    print(f"Extracting activations from {n_prompts} prompts, {len(layers)} layers...")

    for batch_start in range(0, n_prompts, batch_size):
        batch_end = min(batch_start + batch_size, n_prompts)
        batch_prompts = prompts[batch_start:batch_end]

        _, cache = model.run_with_cache(
            batch_prompts,
            names_filter=lambda name: any(f"blocks.{l}.hook_resid_post" in name for l in layers),
        )

        for l in layers:
            key = f"blocks.{l}.hook_resid_post"
            if key in cache:
                if extract_all_positions:
                    act = cache[key].float().cpu()  # (batch, seq, d)
                else:
                    act = cache[key][:, -1, :].float().cpu()  # (batch, d)
                activations[l].append(act)

        del cache
        if device == "cuda":
            torch.cuda.empty_cache()

        if (batch_start // batch_size) % 5 == 0:
            print(f"  Processed {batch_end}/{n_prompts}")

    # Stack batches
    for l in layers:
        activations[l] = torch.cat(activations[l], dim=0)

    return activations, model.cfg.d_model, model.cfg.n_layers


def extract_activations_nnsight(
    model_name: str,
    prompts: list[str],
    layers: list[int] = None,
) -> dict[int, torch.Tensor]:
    """Extract residual stream activations using nnsight (remote)."""
    from nnsight import LanguageModel

    print(f"Loading {model_name} via nnsight...")
    model = LanguageModel(model_name)

    if layers is None:
        n_layers = len(model.model.layers)
        layers = list(range(n_layers))

    activations = {l: [] for l in layers}
    batch_size = 16

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        with model.trace(batch_prompts) as tracer:
            for l in layers:
                act = model.model.layers[l].output[0][:, -1, :].save()
                activations[l].append(act)

    for l in layers:
        activations[l] = torch.cat([a.value.float().cpu() for a in activations[l]], dim=0)

    n_layers_total = len(model.model.layers)
    d_model = activations[layers[0]].shape[1]
    return activations, d_model, n_layers_total


# ─── Analysis ──────────────────────────────────────────────────────

@dataclass
class PhiAnalysisResult:
    """Results of searching for the hallucination direction at one layer."""
    layer: int
    contrastive_direction: np.ndarray  # (d,) unit vector — mean diff
    lda_direction: Optional[np.ndarray] = None  # (d,) — logistic regression
    mean_vs_lda_cosine: float = 0.0  # agreement between two methods
    split_half_cosine: float = 0.0
    probe_accuracy: float = 0.0  # mean-diff based
    lda_probe_accuracy: float = 0.0  # logistic regression based
    mean_correct_projection: float = 0.0
    mean_wrong_projection: float = 0.0
    separation_gap: float = 0.0
    # random baseline comparison
    random_probe_accuracy: float = 0.5
    random_separation_gap: float = 0.0
    # projection distributions for histograms
    proj_correct: Optional[np.ndarray] = None
    proj_wrong: Optional[np.ndarray] = None


def _fit_lda_direction(X_train, y_train, X_test, y_test, d):
    """
    Fit logistic regression to find separation direction.
    Returns (direction, test_accuracy).
    Uses sklearn if available, falls back to manual gradient descent.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        # Standardize to help convergence and prevent scale-dependent bias
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")
        clf.fit(X_train_s, y_train)
        # Transform direction back to original space
        lda_dir = (clf.coef_[0] / scaler.scale_).copy()
        lda_dir = lda_dir / (np.linalg.norm(lda_dir) + 1e-8)
        lda_acc = float(clf.score(X_test_s, y_test))
        return lda_dir, lda_acc
    except ImportError:
        # Manual logistic regression via gradient descent
        w = np.zeros(d, dtype=np.float32)
        lr = 0.01
        for _ in range(200):
            logits = X_train @ w
            preds = 1.0 / (1.0 + np.exp(-logits.clip(-30, 30)))
            grad = X_train.T @ (preds - y_train) / len(y_train)
            w -= lr * grad
        lda_dir = w / (np.linalg.norm(w) + 1e-8)
        # Test accuracy
        test_logits = X_test @ lda_dir
        test_preds = (test_logits > 0).astype(float)
        lda_acc = float((test_preds == y_test).mean())
        return lda_dir, lda_acc


def analyze_layer(
    layer: int,
    act_correct: torch.Tensor,
    act_wrong: torch.Tensor,
    n_random_baselines: int = 10,
    seed: int = 42,
) -> PhiAnalysisResult:
    """
    For a single layer, compute:
    1. Contrastive direction (mean_wrong - mean_correct)
    2. LDA/logistic regression direction (optimizes separation)
    3. Agreement between the two (cosine similarity)
    4. Split-half stability (cosine similarity)
    5. Probe accuracy for both methods
    6. Random baseline
    7. Per-sample projections for distribution plots
    """
    rng = np.random.RandomState(seed)
    n = min(len(act_correct), len(act_wrong))
    act_c = act_correct[:n].numpy()
    act_w = act_wrong[:n].numpy()
    d = act_c.shape[1]

    # ─── 1. Mean-diff contrastive direction ───
    direction = act_w.mean(axis=0) - act_c.mean(axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # ─── 2. Split-half stability ───
    indices = rng.permutation(n)
    half = n // 2
    split_a_c, split_b_c = act_c[indices[:half]], act_c[indices[half:2*half]]
    split_a_w, split_b_w = act_w[indices[:half]], act_w[indices[half:2*half]]

    dir_a = split_a_w.mean(0) - split_a_c.mean(0)
    dir_a = dir_a / (np.linalg.norm(dir_a) + 1e-8)
    dir_b = split_b_w.mean(0) - split_b_c.mean(0)
    dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-8)

    cosine_sim = float(np.dot(dir_a, dir_b))

    # ─── 3. Mean-diff probe accuracy (train/test split) ───
    train_c = act_c[indices[:half]]
    train_w = act_w[indices[:half]]
    test_c = act_c[indices[half:2*half]]
    test_w = act_w[indices[half:2*half]]

    train_dir = train_w.mean(0) - train_c.mean(0)
    train_dir = train_dir / (np.linalg.norm(train_dir) + 1e-8)

    test_proj_c = test_c @ train_dir
    test_proj_w = test_w @ train_dir
    threshold = (test_proj_c.mean() + test_proj_w.mean()) / 2
    acc_c = (test_proj_c < threshold).mean()
    acc_w = (test_proj_w >= threshold).mean()
    probe_accuracy = float((acc_c + acc_w) / 2)

    # ─── 4. LDA / Logistic Regression direction ───
    X_train = np.vstack([train_c, train_w])
    y_train = np.array([0.0] * len(train_c) + [1.0] * len(train_w))
    X_test = np.vstack([test_c, test_w])
    y_test = np.array([0.0] * len(test_c) + [1.0] * len(test_w))

    lda_dir, lda_acc = _fit_lda_direction(X_train, y_train, X_test, y_test, d)

    # Agreement between mean-diff and LDA
    mean_vs_lda_cos = float(np.dot(direction, lda_dir))

    # ─── 5. Per-sample projections (for histograms) ───
    proj_c_all = act_c @ direction
    proj_w_all = act_w @ direction
    mean_proj_c = float(proj_c_all.mean())
    mean_proj_w = float(proj_w_all.mean())
    separation_gap = mean_proj_w - mean_proj_c

    # ─── 6. RANDOM BASELINE ───
    random_accs = []
    random_gaps = []
    for _ in range(n_random_baselines):
        rand_dir = rng.randn(d).astype(np.float32)
        rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-8)

        rp_c = test_c @ rand_dir
        rp_w = test_w @ rand_dir
        r_thresh = (rp_c.mean() + rp_w.mean()) / 2
        r_acc_c = (rp_c < r_thresh).mean()
        r_acc_w = (rp_w >= r_thresh).mean()
        random_accs.append(float((r_acc_c + r_acc_w) / 2))
        random_gaps.append(float(rp_w.mean() - rp_c.mean()))

    return PhiAnalysisResult(
        layer=layer,
        contrastive_direction=direction,
        lda_direction=lda_dir,
        mean_vs_lda_cosine=mean_vs_lda_cos,
        split_half_cosine=cosine_sim,
        probe_accuracy=probe_accuracy,
        lda_probe_accuracy=lda_acc,
        mean_correct_projection=mean_proj_c,
        mean_wrong_projection=mean_proj_w,
        separation_gap=separation_gap,
        random_probe_accuracy=float(np.mean(random_accs)),
        random_separation_gap=float(np.mean(np.abs(random_gaps))),
        proj_correct=proj_c_all,
        proj_wrong=proj_w_all,
    )


def test_generalization(
    direction: np.ndarray,
    act_correct_new: torch.Tensor,
    act_wrong_new: torch.Tensor,
) -> float:
    """Test if a direction generalizes to another operation type."""
    act_c = act_correct_new.numpy()
    act_w = act_wrong_new.numpy()
    proj_c = act_c @ direction
    proj_w = act_w @ direction
    threshold = (proj_c.mean() + proj_w.mean()) / 2
    acc_c = (proj_c < threshold).mean()
    acc_w = (proj_w >= threshold).mean()
    return float((acc_c + acc_w) / 2)


# ─── Non-Arithmetic Domain: Fake Citations ────────────────────────

def generate_citation_data(n: int = 100, seed: int = 42) -> list[ArithmeticSample]:
    """
    Generate real vs fabricated citation statements.
    Tests whether Φ transfers from arithmetic to factual domain.

    If arithmetic Φ separates these too → hallucination is a general phenomenon
    If not → need per-domain Φ (still useful but weaker)

    NOTE: Uses well-known facts. Real experiment should use a verified
    fact database like LAMA or WikiFact.
    """
    import random as rng_mod
    rng = rng_mod.Random(seed)

    # Real facts (checkable, well-known)
    real_facts = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth orbits the Sun.",
        "DNA has a double helix structure.",
        "Light travels at approximately 300,000 km per second.",
        "The human body has 206 bones.",
        "Shakespeare wrote Hamlet.",
        "The speed of sound in air is about 343 m/s.",
        "Oxygen has atomic number 8.",
        "The Great Wall of China is visible from space is a myth.",
        "Pi is approximately 3.14159.",
        "Gold has the chemical symbol Au.",
        "The Moon orbits the Earth.",
        "Photosynthesis converts CO2 and water into glucose.",
        "The Pythagorean theorem relates the sides of a right triangle.",
    ]

    # Fabricated facts (plausible but wrong)
    fake_facts = [
        "The capital of France is Lyon.",
        "Water boils at 90 degrees Celsius at sea level.",
        "The Sun orbits the Earth.",
        "DNA has a triple helix structure.",
        "Light travels at approximately 500,000 km per second.",
        "The human body has 312 bones.",
        "Shakespeare wrote The Odyssey.",
        "The speed of sound in air is about 1200 m/s.",
        "Oxygen has atomic number 12.",
        "The Great Wall of China is easily visible from the Moon.",
        "Pi is approximately 3.17320.",
        "Gold has the chemical symbol Gd.",
        "The Moon orbits Mars.",
        "Photosynthesis converts nitrogen and water into glucose.",
        "The Pythagorean theorem relates the angles of any triangle.",
    ]

    samples = []
    for _ in range(n):
        idx = rng.randint(0, len(real_facts) - 1)
        # Real
        samples.append(ArithmeticSample(
            prompt=real_facts[idx],
            is_correct=True,
            operation="citation",
            answer=0,
        ))
        # Fake
        samples.append(ArithmeticSample(
            prompt=fake_facts[idx],
            is_correct=False,
            operation="citation",
            answer=0,
            wrong_answer=1,
        ))

    return samples


# ─── Factual Knowledge Domain ────────────────────────────────────

@dataclass
class FactualSample:
    prompt: str
    is_correct: bool
    category: str  # "geography", "science", "history", "biology", "physics"
    answer: int = 0
    wrong_answer: Optional[int] = None


def generate_factual_data(
    n_per_category: int = 100,
    categories: list[str] = None,
    seed: int = 42,
) -> list[FactualSample]:
    """
    Generate paired correct/incorrect factual statements.
    Each category has diverse real/fake pairs with plausible wrong answers.

    Categories serve the same role as arithmetic operations:
    cross-category generalization tests whether Φ is universal.
    """
    rng = random.Random(seed)

    if categories is None:
        categories = ["geography", "science", "history"]

    # Large pool of real/fake pairs per category
    fact_pool = {
        "geography": [
            ("The capital of France is Paris.", "The capital of France is Lyon."),
            ("The capital of Germany is Berlin.", "The capital of Germany is Munich."),
            ("The capital of Japan is Tokyo.", "The capital of Japan is Osaka."),
            ("The capital of Italy is Rome.", "The capital of Italy is Milan."),
            ("The capital of Spain is Madrid.", "The capital of Spain is Barcelona."),
            ("The capital of Australia is Canberra.", "The capital of Australia is Sydney."),
            ("The capital of Canada is Ottawa.", "The capital of Canada is Toronto."),
            ("The capital of Brazil is Brasilia.", "The capital of Brazil is Rio de Janeiro."),
            ("The capital of Egypt is Cairo.", "The capital of Egypt is Alexandria."),
            ("The capital of India is New Delhi.", "The capital of India is Mumbai."),
            ("The capital of Turkey is Ankara.", "The capital of Turkey is Istanbul."),
            ("The capital of Russia is Moscow.", "The capital of Russia is St. Petersburg."),
            ("The capital of China is Beijing.", "The capital of China is Shanghai."),
            ("The capital of South Korea is Seoul.", "The capital of South Korea is Busan."),
            ("The capital of Mexico is Mexico City.", "The capital of Mexico is Guadalajara."),
            ("The longest river in the world is the Nile.", "The longest river in the world is the Amazon."),
            ("The largest ocean is the Pacific Ocean.", "The largest ocean is the Atlantic Ocean."),
            ("The smallest continent is Australia.", "The smallest continent is Europe."),
            ("Mount Everest is the tallest mountain.", "Mount Kilimanjaro is the tallest mountain."),
            ("The Sahara is the largest hot desert.", "The Gobi is the largest hot desert."),
            ("The Amazon rainforest is in South America.", "The Amazon rainforest is in Africa."),
            ("The Great Barrier Reef is in Australia.", "The Great Barrier Reef is in Indonesia."),
            ("The Dead Sea is the lowest point on land.", "The Caspian Sea is the lowest point on land."),
            ("Iceland is in the North Atlantic Ocean.", "Iceland is in the Arctic Ocean."),
            ("The Andes are in South America.", "The Andes are in Asia."),
        ],
        "science": [
            ("Water boils at 100 degrees Celsius.", "Water boils at 90 degrees Celsius."),
            ("The speed of light is approximately 300,000 km/s.", "The speed of light is approximately 500,000 km/s."),
            ("Oxygen has atomic number 8.", "Oxygen has atomic number 12."),
            ("Gold has the chemical symbol Au.", "Gold has the chemical symbol Gd."),
            ("DNA has a double helix structure.", "DNA has a triple helix structure."),
            ("The Earth orbits the Sun.", "The Sun orbits the Earth."),
            ("Light travels faster than sound.", "Sound travels faster than light."),
            ("Water freezes at 0 degrees Celsius.", "Water freezes at 10 degrees Celsius."),
            ("The chemical formula for water is H2O.", "The chemical formula for water is H3O."),
            ("Diamonds are made of carbon.", "Diamonds are made of silicon."),
            ("Iron has the chemical symbol Fe.", "Iron has the chemical symbol Ir."),
            ("Helium is the second lightest element.", "Helium is the third lightest element."),
            ("The speed of sound in air is about 343 m/s.", "The speed of sound in air is about 1200 m/s."),
            ("Absolute zero is minus 273 degrees Celsius.", "Absolute zero is minus 200 degrees Celsius."),
            ("Pi is approximately 3.14159.", "Pi is approximately 3.17320."),
            ("Gravity on Earth is about 9.8 m/s squared.", "Gravity on Earth is about 12.5 m/s squared."),
            ("The pH of pure water is 7.", "The pH of pure water is 5."),
            ("The Planck constant is fundamental to quantum mechanics.", "The Planck constant is fundamental to thermodynamics."),
            ("Copper is a good conductor of electricity.", "Glass is a good conductor of electricity."),
            ("Photosynthesis produces oxygen.", "Photosynthesis produces nitrogen."),
            ("Einstein developed the theory of relativity.", "Einstein developed the theory of evolution."),
            ("Newton discovered the law of gravity.", "Darwin discovered the law of gravity."),
            ("The periodic table was created by Mendeleev.", "The periodic table was created by Bohr."),
            ("Sound cannot travel through a vacuum.", "Sound can travel through a vacuum."),
            ("Electrons have a negative charge.", "Electrons have a positive charge."),
        ],
        "history": [
            ("World War II ended in 1945.", "World War II ended in 1943."),
            ("The Berlin Wall fell in 1989.", "The Berlin Wall fell in 1985."),
            ("The French Revolution began in 1789.", "The French Revolution began in 1776."),
            ("Columbus reached the Americas in 1492.", "Columbus reached the Americas in 1498."),
            ("The Roman Empire fell in 476 AD.", "The Roman Empire fell in 530 AD."),
            ("The Declaration of Independence was signed in 1776.", "The Declaration of Independence was signed in 1781."),
            ("The first moon landing was in 1969.", "The first moon landing was in 1965."),
            ("The Titanic sank in 1912.", "The Titanic sank in 1915."),
            ("World War I started in 1914.", "World War I started in 1912."),
            ("The Renaissance began in Italy.", "The Renaissance began in France."),
            ("The printing press was invented by Gutenberg.", "The printing press was invented by Galileo."),
            ("The Great Fire of London was in 1666.", "The Great Fire of London was in 1670."),
            ("Napoleon was defeated at Waterloo.", "Napoleon was defeated at Trafalgar."),
            ("The Soviet Union dissolved in 1991.", "The Soviet Union dissolved in 1988."),
            ("Alexander the Great was from Macedonia.", "Alexander the Great was from Persia."),
            ("The Industrial Revolution started in Britain.", "The Industrial Revolution started in France."),
            ("Julius Caesar was assassinated in 44 BC.", "Julius Caesar was assassinated in 50 BC."),
            ("The Wright brothers flew the first airplane.", "The Wright brothers invented the telephone."),
            ("The Magna Carta was signed in 1215.", "The Magna Carta was signed in 1250."),
            ("The Black Death peaked in Europe in the 1340s.", "The Black Death peaked in Europe in the 1240s."),
            ("The first Olympic Games were held in ancient Greece.", "The first Olympic Games were held in ancient Rome."),
            ("Magellan's expedition circumnavigated the globe.", "Columbus's expedition circumnavigated the globe."),
            ("The Cold War was between the US and Soviet Union.", "The Cold War was between the US and China."),
            ("The Suez Canal opened in 1869.", "The Suez Canal opened in 1890."),
            ("The United Nations was founded in 1945.", "The United Nations was founded in 1940."),
        ],
    }

    samples = []
    for cat in categories:
        if cat not in fact_pool:
            raise ValueError(f"Unknown category: {cat}. Available: {list(fact_pool.keys())}")

        pool = fact_pool[cat]
        for _ in range(n_per_category):
            idx = rng.randint(0, len(pool) - 1)
            real, fake = pool[idx]

            samples.append(FactualSample(
                prompt=real, is_correct=True, category=cat,
            ))
            samples.append(FactualSample(
                prompt=fake, is_correct=False, category=cat, wrong_answer=1,
            ))

    return samples


# ─── Visualization ────────────────────────────────────────────────

def generate_all_plots(
    results: list[PhiAnalysisResult],
    gen_results: dict,
    best_layer: int,
    model_name: str,
    output_dir: Path,
):
    """Generate all diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    layers = [r.layer for r in results]
    stabilities = [r.split_half_cosine for r in results]
    probe_accs = [r.probe_accuracy for r in results]
    random_accs = [r.random_probe_accuracy for r in results]
    gaps = [r.separation_gap for r in results]
    random_gaps = [r.random_separation_gap for r in results]

    # ─── PLOT 1: Layer Sensitivity Dashboard ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Experiment 0: Does Φ Exist? — {model_name}", fontsize=14, fontweight="bold")

    # 1a. Stability across layers
    ax = axes[0, 0]
    ax.bar(layers, stabilities, color=["#e74c3c" if s > 0.7 else "#3498db" if s > 0.5 else "#95a5a6" for s in stabilities])
    ax.axhline(y=0.7, color="red", linestyle="--", alpha=0.7, label="threshold (0.7)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Split-Half Cosine Similarity")
    ax.set_title("Direction Stability per Layer")
    ax.legend()

    # 1b. Probe accuracy: REAL vs RANDOM
    ax = axes[0, 1]
    x = np.arange(len(layers))
    width = 0.35
    ax.bar(x - width/2, probe_accs, width, label="Real Φ", color="#2ecc71")
    ax.bar(x + width/2, random_accs, width, label="Random direction", color="#e74c3c", alpha=0.7)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="chance")
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="target (0.8)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Real Φ vs Random Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(fontsize=8)

    # 1c. Separation gap: REAL vs RANDOM
    ax = axes[1, 0]
    ax.bar(x - width/2, gaps, width, label="Real Φ gap", color="#2ecc71")
    ax.bar(x + width/2, random_gaps, width, label="Random gap (abs)", color="#e74c3c", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Separation Gap")
    ax.set_title("Projection Separation: Real vs Random")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(fontsize=8)

    # 1d. Combined score heatmap
    ax = axes[1, 1]
    combined_scores = [s * a for s, a in zip(stabilities, probe_accs)]
    colors = ["#27ae60" if c > 0.56 else "#f39c12" if c > 0.35 else "#c0392b" for c in combined_scores]
    ax.barh(layers, combined_scores, color=colors)
    ax.set_ylabel("Layer")
    ax.set_xlabel("Combined Score (stability × accuracy)")
    ax.set_title("Best Layer Selection")
    ax.axvline(x=0.56, color="green", linestyle="--", alpha=0.5, label="strong threshold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / "01_layer_sensitivity.png", dpi=150)
    plt.close()
    print(f"  Saved: 01_layer_sensitivity.png")

    # ─── PLOT 2: Projection Distribution at Best Layer ───
    best_result = [r for r in results if r.layer == best_layer][0]
    if best_result.proj_correct is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.hist(best_result.proj_correct, bins=50, alpha=0.6, color="#2ecc71",
                label=f"Correct (n={len(best_result.proj_correct)})", density=True)
        ax.hist(best_result.proj_wrong, bins=50, alpha=0.6, color="#e74c3c",
                label=f"Wrong (n={len(best_result.proj_wrong)})", density=True)
        ax.axvline(x=best_result.mean_correct_projection, color="#27ae60",
                   linestyle="--", linewidth=2, label=f"Mean correct: {best_result.mean_correct_projection:.3f}")
        ax.axvline(x=best_result.mean_wrong_projection, color="#c0392b",
                   linestyle="--", linewidth=2, label=f"Mean wrong: {best_result.mean_wrong_projection:.3f}")

        ax.set_xlabel("Projection onto Φ direction")
        ax.set_ylabel("Density")
        ax.set_title(f"Projection Distribution — Layer {best_layer}\n"
                     f"Gap={best_result.separation_gap:.4f}, Acc={best_result.probe_accuracy:.3f}")
        ax.legend()

        plt.tight_layout()
        plt.savefig(figures_dir / "02_projection_distribution.png", dpi=150)
        plt.close()
        print(f"  Saved: 02_projection_distribution.png")

    # ─── PLOT 3: Cross-Category Generalization Matrix ───
    if gen_results:
        # Extract category names from gen_results keys
        cat_names = sorted(set(
            k.split("→")[0] for k in gen_results.keys()
        ))
        n_cats = len(cat_names)
        gen_matrix = np.eye(n_cats)
        for i, train_cat in enumerate(cat_names):
            for j, test_cat in enumerate(cat_names):
                if train_cat == test_cat:
                    continue
                key = f"{train_cat}→{test_cat}"
                if key in gen_results:
                    gen_matrix[i, j] = gen_results[key]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(gen_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(n_cats))
        ax.set_yticks(range(n_cats))
        ax.set_xticklabels([c[:5] for c in cat_names])
        ax.set_yticklabels([c[:5] for c in cat_names])
        ax.set_xlabel("Test Category")
        ax.set_ylabel("Train Category")
        ax.set_title(f"Cross-Category Generalization — Layer {best_layer}")

        for i in range(n_cats):
            for j in range(n_cats):
                color = "white" if gen_matrix[i, j] < 0.65 else "black"
                ax.text(j, i, f"{gen_matrix[i,j]:.2f}", ha="center", va="center",
                        color=color, fontweight="bold", fontsize=14)

        plt.colorbar(im, ax=ax, label="Probe Accuracy")
        plt.tight_layout()
        plt.savefig(figures_dir / "03_generalization_matrix.png", dpi=150)
        plt.close()
        print(f"  Saved: 03_generalization_matrix.png")

    # ─── PLOT 4: Real vs Random — Effect Size ───
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    effect_sizes = [
        (pa - ra) / max(ra, 0.01) for pa, ra in zip(probe_accs, random_accs)
    ]
    colors = ["#27ae60" if e > 0.3 else "#f39c12" if e > 0.1 else "#95a5a6" for e in effect_sizes]
    ax.bar(layers, effect_sizes, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.5, label="strong effect")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Effect Size: (real_acc - random_acc) / random_acc")
    ax.set_title("Direction Quality: How Much Better Than Random?")
    ax.legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "04_effect_size.png", dpi=150)
    plt.close()
    print(f"  Saved: 04_effect_size.png")

    # ─── PLOT 5: Per-Layer Projection Distributions (small multiples) ───
    n_layers = len(results)
    n_cols = min(6, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes_grid = [axes_grid]
    if n_cols == 1:
        axes_grid = [[ax] for ax in axes_grid]

    fig.suptitle("Projection Distributions per Layer", fontsize=12, fontweight="bold")

    for idx, r in enumerate(results):
        row, col = idx // n_cols, idx % n_cols
        ax = axes_grid[row][col] if isinstance(axes_grid[row], (list, np.ndarray)) else axes_grid[row]

        if r.proj_correct is not None and r.proj_wrong is not None:
            ax.hist(r.proj_correct, bins=30, alpha=0.5, color="#2ecc71", density=True)
            ax.hist(r.proj_wrong, bins=30, alpha=0.5, color="#e74c3c", density=True)

        marker = "**" if r.split_half_cosine > 0.7 and r.probe_accuracy > 0.8 else ""
        ax.set_title(f"L{r.layer} {marker}\nacc={r.probe_accuracy:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide empty subplots
    for idx in range(len(results), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        ax = axes_grid[row][col] if isinstance(axes_grid[row], (list, np.ndarray)) else axes_grid[row]
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(figures_dir / "05_all_layers_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved: 05_all_layers_distributions.png")

    print(f"\n  All figures saved to: {figures_dir}/")


# ─── Token Position Analysis ──────────────────────────────────────

def analyze_token_positions(
    model_name: str,
    samples: list[ArithmeticSample],
    best_layer: int,
    best_direction: np.ndarray,
    device: str = "auto",
    output_dir: Path = None,
    n_subset: int = 100,
):
    """
    Analyze whether the Φ direction is active at ALL token positions
    or only at the final (answer) token.

    If only at final token → weaker signal, surface-level
    If throughout → deeper, structural → stronger TLoT case
    """
    import transformer_lens as tl

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = tl.HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    # Take a subset
    correct_samples = [s for s in samples if s.is_correct][:n_subset]
    wrong_samples = [s for s in samples if not s.is_correct][:n_subset]

    direction_tensor = torch.tensor(best_direction, dtype=torch.float32)

    position_projections = {"correct": [], "wrong": []}

    for label, subset in [("correct", correct_samples), ("wrong", wrong_samples)]:
        for sample in subset:
            tokens = model.to_tokens(sample.prompt)
            seq_len = tokens.shape[1]

            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: f"blocks.{best_layer}.hook_resid_post" in name,
            )

            key = f"blocks.{best_layer}.hook_resid_post"
            act = cache[key][0].float().cpu()  # (seq_len, d_model)

            # Project each position onto Φ
            projections = (act @ direction_tensor).numpy()  # (seq_len,)
            position_projections[label].append(projections)

            del cache

    # Pad to same length and compute stats
    max_len = max(
        max(len(p) for p in position_projections["correct"]),
        max(len(p) for p in position_projections["wrong"]),
    )

    def pad_and_stack(lst, max_len):
        padded = np.full((len(lst), max_len), np.nan)
        for i, p in enumerate(lst):
            padded[i, :len(p)] = p
        return padded

    correct_matrix = pad_and_stack(position_projections["correct"], max_len)
    wrong_matrix = pad_and_stack(position_projections["wrong"], max_len)

    # Per-position mean and gap
    correct_means = np.nanmean(correct_matrix, axis=0)
    wrong_means = np.nanmean(wrong_matrix, axis=0)
    gaps = wrong_means - correct_means

    result = {
        "max_seq_len": max_len,
        "per_position_gap": gaps.tolist(),
        "final_token_gap": float(gaps[-1]) if len(gaps) > 0 else 0.0,
        "mean_gap_all_positions": float(np.nanmean(gaps)),
        "gap_ratio_final_vs_all": (
            float(gaps[-1] / np.nanmean(np.abs(gaps)))
            if np.nanmean(np.abs(gaps)) > 1e-8 else 0.0
        ),
    }

    # Plot
    if output_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: per-position gap
        valid_positions = ~np.isnan(gaps)
        positions = np.arange(max_len)[valid_positions]
        gap_values = gaps[valid_positions]

        ax = axes[0]
        ax.bar(positions, gap_values,
               color=["#e74c3c" if g > 0 else "#3498db" for g in gap_values])
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Projection Gap (wrong - correct)")
        ax.set_title(f"Φ Signal by Token Position — Layer {best_layer}")
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Right: correct vs wrong projections over positions
        ax = axes[1]
        ax.plot(positions, correct_means[valid_positions], "g-o",
                markersize=4, label="Correct", alpha=0.8)
        ax.plot(positions, wrong_means[valid_positions], "r-o",
                markersize=4, label="Wrong", alpha=0.8)
        ax.fill_between(positions, correct_means[valid_positions],
                        wrong_means[valid_positions], alpha=0.15, color="purple")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Mean Projection onto Φ")
        ax.set_title("Correct vs Wrong: Φ Component per Position")
        ax.legend()

        plt.tight_layout()
        plt.savefig(figures_dir / "06_token_position_analysis.png", dpi=150)
        plt.close()
        print(f"  Saved: 06_token_position_analysis.png")

    return result


# ─── Main ──────────────────────────────────────────────────────────

def run_experiment(
    model_name: str = "pythia-1.4b",
    backend: str = "transformerlens",
    n_samples: int = 200,
    output_dir: str = None,
    device: str = "auto",
    skip_token_analysis: bool = False,
    task: str = "factual",
):
    """
    Full Experiment 0 pipeline with diagnostics and plots.

    1. Generate data (arithmetic or factual)
    2. Extract activations
    3. Analyze each layer (with random baseline)
    4. Test cross-category generalization
    5. Generate diagnostic plots
    6. Token position analysis
    7. Verdict
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 0: DOES Φ EXIST?")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"Backend: {backend}")
    print(f"Task: {task}")
    print(f"Samples per category: {n_samples}")

    # ─── 1. Generate data ───
    if task == "arithmetic":
        category_names = ["addition", "multiplication", "subtraction"]
        samples = generate_arithmetic_data(
            n_per_type=n_samples,
            operations=category_names,
        )
        all_prompts = [s.prompt for s in samples]
        is_correct = [s.is_correct for s in samples]
        categories = [s.operation for s in samples]
    elif task == "factual":
        category_names = ["geography", "science", "history"]
        factual_samples = generate_factual_data(
            n_per_category=n_samples,
            categories=category_names,
        )
        all_prompts = [s.prompt for s in factual_samples]
        is_correct = [s.is_correct for s in factual_samples]
        categories = [s.category for s in factual_samples]
        # Wrap into ArithmeticSample-like for token analysis compatibility
        samples = [
            ArithmeticSample(
                prompt=s.prompt, is_correct=s.is_correct,
                operation=s.category, answer=s.answer,
                wrong_answer=s.wrong_answer,
            )
            for s in factual_samples
        ]
    else:
        raise ValueError(f"Unknown task: {task}. Use 'arithmetic' or 'factual'")

    # ─── 2. Extract activations ───
    print(f"\n--- Extracting activations ---")
    if backend == "transformerlens":
        activations, d_model, n_layers = extract_activations_transformerlens(
            model_name, all_prompts, device=device,
        )
    elif backend == "nnsight":
        activations, d_model, n_layers = extract_activations_nnsight(model_name, all_prompts)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    correct_mask = torch.tensor(is_correct)
    wrong_mask = ~correct_mask

    op_masks = {}
    for cat in category_names:
        op_masks[cat] = torch.tensor([c == cat for c in categories])

    # ─── 3. Analyze each layer ───
    print(f"\n--- Analyzing {len(activations)} layers ---")
    print(f"{'Layer':>6s}  {'Stab':>6s}  {'MeanAcc':>8s}  {'LDAAcc':>8s}  {'RandAcc':>8s}  "
          f"{'M↔L cos':>8s}  {'Signal':>8s}")
    print("-" * 72)

    results = []
    for layer_idx in sorted(activations.keys()):
        act = activations[layer_idx]
        act_c = act[correct_mask]
        act_w = act[wrong_mask]

        result = analyze_layer(layer_idx, act_c, act_w)
        results.append(result)

        signal = result.probe_accuracy - result.random_probe_accuracy

        marker = ""
        if (result.split_half_cosine > 0.7 and result.probe_accuracy > 0.8
                and result.mean_vs_lda_cosine > 0.7):
            marker = " *** STRONG (mean≈LDA)"
        elif result.split_half_cosine > 0.7 and result.probe_accuracy > 0.8:
            marker = " ** STRONG"
        elif result.split_half_cosine > 0.5 and result.probe_accuracy > 0.7:
            marker = " * moderate"

        print(f"  {layer_idx:4d}  {result.split_half_cosine:>6.3f}  "
              f"{result.probe_accuracy:>8.3f}  {result.lda_probe_accuracy:>8.3f}  "
              f"{result.random_probe_accuracy:>8.3f}  "
              f"{result.mean_vs_lda_cosine:>8.3f}  "
              f"{signal:>8.3f}{marker}")

    # ─── 4. Find best layer ───
    # Use max of mean-diff and LDA accuracy for layer selection
    best_mean = max(results, key=lambda r: r.split_half_cosine * r.probe_accuracy)
    best_lda = max(results, key=lambda r: r.lda_probe_accuracy)

    # Pick the layer with best overall signal
    best_mean_score = best_mean.split_half_cosine * best_mean.probe_accuracy
    best_lda_score = best_lda.lda_probe_accuracy
    if best_lda_score > 0.7 and best_lda_score > best_mean_score:
        best = best_lda
        best_method = "LDA"
    else:
        best = best_mean
        best_method = "mean-diff"

    print(f"\n--- Best layer: {best.layer} (selected by {best_method}) ---")
    print(f"  Stability:  {best.split_half_cosine:.4f}")
    print(f"  Probe acc (mean-diff): {best.probe_accuracy:.4f}")
    print(f"  Probe acc (LDA):       {best.lda_probe_accuracy:.4f}")
    print(f"  Random baseline:       {best.random_probe_accuracy:.4f}")
    print(f"  Gap:        {best.separation_gap:.4f} (random: {best.random_separation_gap:.4f})")
    print(f"  Signal (mean-diff): {best.probe_accuracy - best.random_probe_accuracy:.4f}")
    print(f"  Signal (LDA):       {best.lda_probe_accuracy - best.random_probe_accuracy:.4f}")

    # ─── 5. Cross-category generalization ───
    print(f"\n--- Cross-category generalization (layer {best.layer}) ---")
    best_act = activations[best.layer]
    gen_results = {}

    for train_cat in category_names:
        for test_cat in category_names:
            if train_cat == test_cat:
                continue

            train_mask_c = correct_mask & op_masks[train_cat]
            train_mask_w = wrong_mask & op_masks[train_cat]
            train_dir = best_act[train_mask_w].mean(0) - best_act[train_mask_c].mean(0)
            train_dir = train_dir.numpy()
            train_dir = train_dir / (np.linalg.norm(train_dir) + 1e-8)

            test_mask_c = correct_mask & op_masks[test_cat]
            test_mask_w = wrong_mask & op_masks[test_cat]

            gen_acc = test_generalization(
                train_dir, best_act[test_mask_c], best_act[test_mask_w],
            )
            gen_results[f"{train_cat}→{test_cat}"] = gen_acc
            print(f"  {train_cat:15s} → {test_cat:15s}: accuracy = {gen_acc:.3f}")

    # ─── 6. Cross-direction cosine similarity ───
    print(f"\n--- Cross-direction cosine similarity (layer {best.layer}) ---")
    op_directions = {}
    for cat in category_names:
        mc = correct_mask & op_masks[cat]
        mw = wrong_mask & op_masks[cat]
        d_cat = best_act[mw].mean(0) - best_act[mc].mean(0)
        d_cat = d_cat.numpy()
        d_cat = d_cat / (np.linalg.norm(d_cat) + 1e-8)
        op_directions[cat] = d_cat

    cross_cosines = {}
    for i, cat1 in enumerate(category_names):
        for cat2 in category_names[i+1:]:
            cos_val = float(np.dot(op_directions[cat1], op_directions[cat2]))
            cross_cosines[f"{cat1}↔{cat2}"] = cos_val
            print(f"  cos({cat1[:4]}, {cat2[:4]}) = {cos_val:.3f}")

    mean_cross_cos = float(np.mean(list(cross_cosines.values())))
    print(f"  Mean cross-cosine: {mean_cross_cos:.3f}")
    if mean_cross_cos > 0.7:
        print(f"  → Same underlying phenomenon across categories (strong)")
    elif mean_cross_cos > 0.3 and all(v > 0.65 for v in gen_results.values()):
        print(f"  → Different directions but transfer works → DEEPER STRUCTURE (very strong)")
    else:
        print(f"  → Directions are category-specific")

    # ─── 7. Cross-domain transfer ───
    cross_domain_transfer = None
    if task == "arithmetic":
        # Test arithmetic Φ → factual domain
        print(f"\n--- Cross-domain transfer test (arithmetic Φ → factual claims) ---")
        try:
            citation_samples = generate_citation_data(n=50)
            citation_prompts = [s.prompt for s in citation_samples]
            citation_correct = [s.is_correct for s in citation_samples]

            if backend == "transformerlens":
                citation_acts, _, _ = extract_activations_transformerlens(
                    model_name, citation_prompts,
                    layers=[best.layer], device=device,
                )
            else:
                citation_acts, _, _ = extract_activations_nnsight(
                    model_name, citation_prompts, layers=[best.layer],
                )

            cite_act = citation_acts[best.layer]
            cite_correct_mask = torch.tensor(citation_correct)

            cite_transfer_acc = test_generalization(
                best.contrastive_direction,
                cite_act[cite_correct_mask],
                cite_act[~cite_correct_mask],
            )
            print(f"  Arithmetic Φ → factual domain: accuracy = {cite_transfer_acc:.3f}")
            cross_domain_transfer = cite_transfer_acc
        except Exception as e:
            print(f"  Cross-domain test failed: {e}")
    elif task == "factual":
        # Test factual Φ → arithmetic domain (if model can do arithmetic)
        print(f"\n--- Cross-domain transfer test (factual Φ → arithmetic) ---")
        try:
            arith_samples = generate_arithmetic_data(n_per_type=50, operations=["addition"])
            arith_prompts = [s.prompt for s in arith_samples]
            arith_correct = [s.is_correct for s in arith_samples]

            if backend == "transformerlens":
                arith_acts, _, _ = extract_activations_transformerlens(
                    model_name, arith_prompts,
                    layers=[best.layer], device=device,
                )
            else:
                arith_acts, _, _ = extract_activations_nnsight(
                    model_name, arith_prompts, layers=[best.layer],
                )

            arith_act = arith_acts[best.layer]
            arith_correct_mask = torch.tensor(arith_correct)

            arith_transfer_acc = test_generalization(
                best.contrastive_direction,
                arith_act[arith_correct_mask],
                arith_act[~arith_correct_mask],
            )
            print(f"  Factual Φ → arithmetic domain: accuracy = {arith_transfer_acc:.3f}")
            if arith_transfer_acc > 0.65:
                print(f"  CROSS-DOMAIN TRANSFER — Φ may be a GENERAL truthfulness direction!")
            else:
                print(f"  No cross-domain transfer (expected — model may not do arithmetic)")
            cross_domain_transfer = arith_transfer_acc
        except Exception as e:
            print(f"  Cross-domain test failed: {e}")

    # ─── 8. Generate plots (renumbered) ───
    print(f"\n--- Generating diagnostic plots ---")
    try:
        generate_all_plots(results, gen_results, best.layer, model_name, output_dir)
    except ImportError:
        print("  matplotlib not available, skipping plots")

    # ─── 7. Token position analysis ───
    token_position_result = None
    if not skip_token_analysis and backend == "transformerlens":
        print(f"\n--- Token position analysis ---")
        try:
            token_position_result = analyze_token_positions(
                model_name, samples, best.layer, best.contrastive_direction,
                device=device, output_dir=output_dir, n_subset=50,
            )
            print(f"  Final token gap:     {token_position_result['final_token_gap']:.4f}")
            print(f"  Mean gap all pos:    {token_position_result['mean_gap_all_positions']:.4f}")
            print(f"  Ratio final/all:     {token_position_result['gap_ratio_final_vs_all']:.2f}")

            if token_position_result["gap_ratio_final_vs_all"] > 3.0:
                print(f"  Warning: Signal concentrated at final token — may be surface-level")
            elif token_position_result["gap_ratio_final_vs_all"] < 1.5:
                print(f"  Good: Signal distributed across positions — structural")
        except Exception as e:
            print(f"  Token analysis failed: {e}")

    # ─── 9. Verdict ───
    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")

    signal_mean = best.probe_accuracy - best.random_probe_accuracy
    signal_lda = best.lda_probe_accuracy - best.random_probe_accuracy
    # Use the BEST signal between mean-diff and LDA
    signal = max(signal_mean, signal_lda)
    methods_agree = best.mean_vs_lda_cosine > 0.7

    # Criteria check using EITHER method:
    # A direction exists if either mean-diff or LDA finds it
    phi_exists_mean = best.split_half_cosine > 0.7 and best.probe_accuracy > 0.8
    phi_exists_lda = best.lda_probe_accuracy > 0.75  # LDA threshold slightly lower — it optimizes
    phi_exists = phi_exists_mean or phi_exists_lda
    phi_better_than_random = signal > 0.2
    phi_generalizes = all(v > 0.65 for v in gen_results.values())

    print(f"  Signal (mean-diff):  {signal_mean:.3f}")
    print(f"  Signal (LDA):        {signal_lda:.3f}")
    print(f"  Best signal:         {signal:.3f} (threshold: 0.2)")
    print(f"  LDA acc (best layer): {best.lda_probe_accuracy:.3f}")
    print(f"  Mean↔LDA cosine:     {best.mean_vs_lda_cosine:.3f} ({'AGREE' if methods_agree else 'DISAGREE'})")
    print(f"  Cross-cat cosine:    {mean_cross_cos:.3f}")
    print()

    if phi_exists and phi_better_than_random and phi_generalizes and methods_agree:
        print("VERY STRONG: Φ EXISTS, BEATS RANDOM, GENERALIZES, METHODS AGREE")
        print("  Mean-diff and logistic regression find the SAME direction.")
        print("  → Proceed to Experiment 1 (projection + sign flip)")
        verdict = "PHI_VERY_STRONG"
    elif phi_exists and phi_better_than_random and phi_generalizes:
        print("STRONG: Φ EXISTS, BEATS RANDOM, GENERALIZES")
        print("  → Proceed to Experiment 1, but investigate why methods disagree")
        verdict = "PHI_EXISTS_AND_GENERALIZES"
    elif phi_exists and phi_better_than_random:
        print("MODERATE: Φ EXISTS and BEATS RANDOM but category-specific")
        if phi_exists_lda and not phi_exists_mean:
            print("  NOTE: LDA finds the direction but mean-diff does not.")
            print("  → The truthfulness direction is NOT the mean difference.")
            print("  → Use LDA direction for Experiment 1.")
        print("  → Proceed to Experiment 1 with per-category or LDA directions")
        verdict = "PHI_EXISTS_BUT_SPECIFIC"
    elif phi_better_than_random:
        print("WEAK: Direction better than random but not stable/strong enough")
        print("  → Try more data, different contrastive method, or SAE features")
        verdict = "PHI_WEAK_SIGNAL"
    elif phi_exists_lda:
        # LDA finds something but signal < 0.2 — could be template leakage
        print("CAUTIOUS: LDA finds separation but signal is marginal")
        print("  → Check for template leakage, try with more diverse data")
        print("  → Consider: direction may be capturing style, not truthfulness")
        verdict = "PHI_NEEDS_VALIDATION"
    else:
        print("NEGATIVE: No direction found better than random")
        print("  → Hallucination may not be a single linear direction")
        print("  → Try: multi-direction Φ, SAE decomposition, nonlinear probe")
        verdict = "PHI_NOT_FOUND"

    # ─── 10. Save results ───
    results_dict = {
        "model": model_name,
        "backend": backend,
        "task": task,
        "categories": category_names,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_samples_per_category": n_samples,
        "verdict": verdict,
        "best_layer": best.layer,
        "best_stability": best.split_half_cosine,
        "best_probe_accuracy_mean_diff": best.probe_accuracy,
        "best_probe_accuracy_lda": best.lda_probe_accuracy,
        "best_random_accuracy": best.random_probe_accuracy,
        "best_signal_mean_diff": signal_mean,
        "best_signal_lda": signal_lda,
        "mean_vs_lda_cosine": best.mean_vs_lda_cosine,
        "best_separation_gap": best.separation_gap,
        "cross_direction_cosines": cross_cosines,
        "mean_cross_cosine": mean_cross_cos,
        "generalization": gen_results,
        "cross_domain_transfer_citation": cross_domain_transfer,
        "token_position": token_position_result,
        "per_layer": [
            {
                "layer": r.layer,
                "stability": r.split_half_cosine,
                "probe_accuracy_mean_diff": r.probe_accuracy,
                "probe_accuracy_lda": r.lda_probe_accuracy,
                "mean_vs_lda_cosine": r.mean_vs_lda_cosine,
                "random_accuracy": r.random_probe_accuracy,
                "signal": r.probe_accuracy - r.random_probe_accuracy,
                "separation_gap": r.separation_gap,
                "random_gap": r.random_separation_gap,
            }
            for r in results
        ],
    }

    results_path = output_dir / f"e00_results_{model_name.replace('/', '_')}.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save best directions (both methods)
    direction_path = output_dir / f"phi_direction_layer{best.layer}_{model_name.replace('/', '_')}.npy"
    np.save(direction_path, best.contrastive_direction)
    print(f"Best Φ direction (mean-diff) saved to: {direction_path}")

    if best.lda_direction is not None:
        lda_path = output_dir / f"phi_lda_direction_layer{best.layer}_{model_name.replace('/', '_')}.npy"
        np.save(lda_path, best.lda_direction)
        print(f"Best Φ direction (LDA) saved to: {lda_path}")

    # Save all layer directions for later analysis
    save_dict = {}
    for r in results:
        save_dict[f"layer_{r.layer}_mean_diff"] = r.contrastive_direction
        if r.lda_direction is not None:
            save_dict[f"layer_{r.layer}_lda"] = r.lda_direction
    all_directions_path = output_dir / f"phi_all_layers_{model_name.replace('/', '_')}.npz"
    np.savez(all_directions_path, **save_dict)
    print(f"All layer directions saved to: {all_directions_path}")

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 0: Does Φ exist?")
    parser.add_argument("--model", type=str, default="pythia-410m",
                        help="Model name (TransformerLens or HuggingFace format)")
    parser.add_argument("--backend", type=str, default="transformerlens",
                        choices=["transformerlens", "nnsight"])
    parser.add_argument("--task", type=str, default="factual",
                        choices=["arithmetic", "factual"],
                        help="Task domain: factual (recommended) or arithmetic")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Samples per category")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for TransformerLens (auto/cuda/mps/cpu)")
    parser.add_argument("--skip-token-analysis", action="store_true",
                        help="Skip token position analysis (faster)")
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        backend=args.backend,
        task=args.task,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        device=args.device,
        skip_token_analysis=args.skip_token_analysis,
    )
