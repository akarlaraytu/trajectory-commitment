"""
Experiment 2: Can We Build ψ?

ψ : R^d → P(S)  —  reads reasoning state from hidden states

This is the Door Lock Axiom in practice:
    We don't ASK the model what it's doing.
    We OBSERVE its hidden states and classify.

If ψ works:
    We can dynamically switch Φ based on reasoning state.
    TLoT becomes state-dependent → the real system.

If ψ fails:
    We can still use static Φ (from e00/e01).
    But we lose the temporal/evolving part of TLoT.

This experiment:
1. Generates reasoning traces with labeled states
   (PARSING → COMPUTING → VERIFYING → OUTPUT)
2. Records hidden states at each reasoning step
3. Trains a linear probe: h → P(state)
4. Tests multi-probe disagreement (ψ₁ ≠ ψ₂ → HALT)

Usage:
    python build_probe.py --model pythia-1.4b
"""

import argparse
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─── Reasoning States ─────────────────────────────────────────────

class ReasoningPhase(IntEnum):
    """
    Simplified reasoning phases for arithmetic.
    These are the S in our ⟨S, ⇒, Φ, π, ψ⟩ system.
    """
    PARSING = 0       # reading and understanding the problem
    COMPUTING = 1     # doing the calculation
    VERIFYING = 2     # checking the result
    OUTPUTTING = 3    # producing the final answer


# ─── Training Data Generation ─────────────────────────────────────

def generate_labeled_traces(n: int = 500, seed: int = 42) -> list[dict]:
    """
    Generate prompts where we KNOW which reasoning phase
    each token belongs to, by construction.

    Strategy: use structured prompts where phases are explicit.
    """
    import random
    rng = random.Random(seed)

    traces = []
    for _ in range(n):
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        result = eval(f"{a}{op}{b}")

        # Each segment maps to a reasoning phase
        segments = [
            # PARSING: the problem statement
            {
                "text": f"Calculate: {a} {op} {b}. ",
                "phase": ReasoningPhase.PARSING,
            },
            # COMPUTING: step-by-step work
            {
                "text": f"Step 1: {a} {op} {b} = ",
                "phase": ReasoningPhase.COMPUTING,
            },
            # VERIFYING: checking
            {
                "text": f"{result}. Let me verify: ",
                "phase": ReasoningPhase.VERIFYING,
            },
            # OUTPUTTING: final answer
            {
                "text": f"Yes, the answer is {result}.",
                "phase": ReasoningPhase.OUTPUTTING,
            },
        ]

        full_text = "".join(s["text"] for s in segments)
        traces.append({
            "full_text": full_text,
            "segments": segments,
            "a": a, "b": b, "op": op, "result": result,
        })

    return traces


def extract_labeled_activations(
    model_name: str,
    traces: list[dict],
    target_layers: list[int] = None,
    device: str = "auto",
) -> dict:
    """
    For each trace, tokenize the full text, record which tokens
    belong to which reasoning phase, and extract activations.

    Returns: {
        layer_idx: {
            "activations": tensor (n_tokens, d_model),
            "labels": tensor (n_tokens,) — ReasoningPhase integer
        }
    }
    """
    import transformer_lens as tl

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = tl.HookedTransformer.from_pretrained(
        model_name, device=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    if target_layers is None:
        n = model.cfg.n_layers
        # Check early, middle, and late layers
        target_layers = [n // 4, n // 2, 3 * n // 4, n - 1]

    all_activations = {l: [] for l in target_layers}
    all_labels = {l: [] for l in target_layers}

    print(f"Extracting labeled activations from {len(traces)} traces...")

    for i, trace in enumerate(traces):
        # Tokenize each segment to know phase boundaries
        segment_token_counts = []
        for seg in trace["segments"]:
            tokens = model.to_tokens(seg["text"], prepend_bos=False)
            segment_token_counts.append(tokens.shape[1])

        # Tokenize full text
        full_tokens = model.to_tokens(trace["full_text"])

        # Create label array: one label per token
        # BOS token gets PARSING label
        labels = []
        labels.append(ReasoningPhase.PARSING)  # BOS

        for seg, n_tok in zip(trace["segments"], segment_token_counts):
            labels.extend([seg["phase"]] * n_tok)

        # Trim to actual token count (tokenization might differ)
        actual_len = full_tokens.shape[1]
        labels = labels[:actual_len]
        while len(labels) < actual_len:
            labels.append(labels[-1])

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Run model and cache
        with torch.no_grad():
            _, cache = model.run_with_cache(
                full_tokens,
                names_filter=lambda name: any(
                    f"blocks.{l}.hook_resid_post" in name for l in target_layers
                ),
            )

        for l in target_layers:
            key = f"blocks.{l}.hook_resid_post"
            if key in cache:
                act = cache[key][0].float().cpu()  # (seq_len, d_model)
                all_activations[l].append(act)
                all_labels[l].append(labels_tensor)

        del cache
        if device == "cuda":
            torch.cuda.empty_cache()

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(traces)}")

    # Stack
    result = {}
    for l in target_layers:
        result[l] = {
            "activations": torch.cat(all_activations[l], dim=0),
            "labels": torch.cat(all_labels[l], dim=0),
        }
        print(f"  Layer {l}: {result[l]['activations'].shape[0]} tokens")

    return result


# ─── Linear Probe ─────────────────────────────────────────────────

class ReasoningProbe(nn.Module):
    """
    ψ: R^d → P(S)

    A linear probe that maps hidden states to reasoning phase
    probabilities. Linear because we want to test if the information
    is LINEARLY accessible (not hidden in nonlinear combinations).

    If linear probe works → the model has a clean representation
    of reasoning state that we can read without asking.
    """

    def __init__(self, d_model: int, n_states: int = 4):
        super().__init__()
        self.linear = nn.Linear(d_model, n_states)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities over reasoning states."""
        return self.linear(h)

    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        """Returns probabilities over reasoning states."""
        with torch.no_grad():
            logits = self.forward(h)
            return F.softmax(logits, dim=-1)

    def predict(self, h: torch.Tensor) -> torch.Tensor:
        """Returns predicted state index."""
        with torch.no_grad():
            return self.forward(h).argmax(dim=-1)


def train_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    d_model: int,
    n_states: int = 4,
    train_fraction: float = 0.8,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 42,
) -> dict:
    """
    Train a linear probe and return train/test accuracy.
    """
    torch.manual_seed(seed)
    n = len(activations)
    n_train = int(n * train_fraction)

    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train, y_train = activations[train_idx], labels[train_idx]
    X_test, y_test = activations[test_idx], labels[test_idx]

    probe = ReasoningProbe(d_model, n_states)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
    )

    # Train
    for epoch in range(epochs):
        probe.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            logits = probe(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        train_preds = probe.predict(X_train)
        test_preds = probe.predict(X_test)
        train_acc = (train_preds == y_train).float().mean().item()
        test_acc = (test_preds == y_test).float().mean().item()

        # Per-class accuracy
        per_class = {}
        for state in ReasoningPhase:
            mask = y_test == state.value
            if mask.sum() > 0:
                class_acc = (test_preds[mask] == y_test[mask]).float().mean().item()
                per_class[state.name] = {
                    "accuracy": class_acc,
                    "n_samples": mask.sum().item(),
                }

        # Confidence distribution
        test_probs = probe.predict_proba(X_test)
        max_probs = test_probs.max(dim=-1).values
        confidence_stats = {
            "mean": max_probs.mean().item(),
            "std": max_probs.std().item(),
            "min": max_probs.min().item(),
            "pct_above_70": (max_probs > 0.7).float().mean().item(),
            "pct_above_90": (max_probs > 0.9).float().mean().item(),
        }

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "per_class": per_class,
        "confidence_stats": confidence_stats,
        "probe": probe,
    }


# ─── Multi-Probe Disagreement Test ───────────────────────────────

def test_multi_probe_disagreement(
    activations: torch.Tensor,
    labels: torch.Tensor,
    d_model: int,
    n_probes: int = 3,
    seeds: list[int] = None,
) -> dict:
    """
    Train multiple independent probes (different random seeds)
    and measure disagreement rate.

    LEMMA: If ψ₁(h) ≠ ψ₂(h), state(h) is undefined → system should HALT.

    This tests how often that happens, and whether disagreement
    correlates with actual misclassification.
    """
    if seeds is None:
        seeds = list(range(n_probes))

    probes = []
    for seed in seeds:
        result = train_probe(activations, labels, d_model, seed=seed)
        probes.append(result["probe"])
        print(f"  Probe {seed}: test_acc={result['test_accuracy']:.3f}")

    # Test disagreement on held-out data
    n = len(activations)
    test_idx = torch.randperm(n)[int(n * 0.8):]
    X_test = activations[test_idx]
    y_test = labels[test_idx]

    predictions = []
    for probe in probes:
        probe.eval()
        with torch.no_grad():
            preds = probe.predict(X_test)
            predictions.append(preds)

    predictions = torch.stack(predictions, dim=0)  # (n_probes, n_test)

    # Disagreement: not all probes agree
    all_agree = (predictions == predictions[0:1]).all(dim=0)
    disagreement_rate = 1.0 - all_agree.float().mean().item()

    # When probes disagree, how often is at least one wrong?
    any_correct_when_disagree = torch.zeros(0)
    if (~all_agree).any():
        disagree_preds = predictions[:, ~all_agree]
        disagree_true = y_test[~all_agree]
        any_correct = (disagree_preds == disagree_true.unsqueeze(0)).any(dim=0)
        any_correct_rate = any_correct.float().mean().item()
    else:
        any_correct_rate = 1.0

    # When probes agree, how often are they correct?
    if all_agree.any():
        agree_preds = predictions[0, all_agree]
        agree_true = y_test[all_agree]
        agree_accuracy = (agree_preds == agree_true).float().mean().item()
    else:
        agree_accuracy = 0.0

    return {
        "n_probes": n_probes,
        "disagreement_rate": disagreement_rate,
        "accuracy_when_agree": agree_accuracy,
        "any_correct_when_disagree": any_correct_rate,
        "interpretation": (
            "LOW disagreement + HIGH agree_accuracy = ψ is reliable"
            if disagreement_rate < 0.1 and agree_accuracy > 0.9
            else "HIGH disagreement = ψ needs improvement"
            if disagreement_rate > 0.2
            else "MODERATE — usable with caution"
        ),
    }


# ─── Main ──────────────────────────────────────────────────────────

def run_experiment(
    model_name: str = "pythia-1.4b",
    n_traces: int = 300,
    device: str = "auto",
    output_dir: str = None,
):
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 2: CAN WE BUILD ψ?")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Traces: {n_traces}")

    # 1. Generate labeled data
    print("\n--- Generating labeled reasoning traces ---")
    traces = generate_labeled_traces(n=n_traces)
    print(f"Generated {len(traces)} traces")

    # 2. Extract activations with labels
    print("\n--- Extracting activations ---")
    labeled_data = extract_labeled_activations(
        model_name, traces, device=device,
    )

    # 3. Train probe at each layer
    print("\n--- Training probes ---")
    layer_results = {}

    for layer_idx, data in sorted(labeled_data.items()):
        print(f"\n  Layer {layer_idx}:")
        d_model = data["activations"].shape[1]
        result = train_probe(
            data["activations"], data["labels"], d_model,
        )
        layer_results[layer_idx] = {
            "train_accuracy": result["train_accuracy"],
            "test_accuracy": result["test_accuracy"],
            "per_class": result["per_class"],
            "confidence_stats": result["confidence_stats"],
        }
        print(f"    Train acc: {result['train_accuracy']:.3f}")
        print(f"    Test acc:  {result['test_accuracy']:.3f}")
        print(f"    Confidence >70%: {result['confidence_stats']['pct_above_70']:.1%}")

        # Per-class breakdown
        for cls_name, cls_data in result["per_class"].items():
            print(f"      {cls_name:12s}: {cls_data['accuracy']:.3f} (n={cls_data['n_samples']})")

    # 4. Multi-probe disagreement on best layer
    best_layer = max(layer_results, key=lambda l: layer_results[l]["test_accuracy"])
    print(f"\n--- Multi-probe disagreement test (layer {best_layer}) ---")
    best_data = labeled_data[best_layer]
    d_model = best_data["activations"].shape[1]

    disagreement = test_multi_probe_disagreement(
        best_data["activations"], best_data["labels"], d_model,
    )
    print(f"  Disagreement rate: {disagreement['disagreement_rate']:.3f}")
    print(f"  Accuracy when agree: {disagreement['accuracy_when_agree']:.3f}")
    print(f"  Interpretation: {disagreement['interpretation']}")

    # 5. Verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    best_acc = layer_results[best_layer]["test_accuracy"]
    best_conf = layer_results[best_layer]["confidence_stats"]["pct_above_70"]

    if best_acc > 0.85 and disagreement["disagreement_rate"] < 0.1:
        print(f"✓ ψ WORKS — layer {best_layer}")
        print(f"  Test accuracy: {best_acc:.3f}")
        print(f"  Probe agreement: {1 - disagreement['disagreement_rate']:.1%}")
        print(f"  → Proceed to TLoT runtime (state-dependent Φ)")
        verdict = "PSI_WORKS"
    elif best_acc > 0.7:
        print(f"◆ ψ PARTIALLY WORKS — layer {best_layer}")
        print(f"  Test accuracy: {best_acc:.3f}")
        print(f"  → Usable with confidence thresholding")
        verdict = "PSI_PARTIAL"
    else:
        print(f"✗ ψ FAILS — reasoning states not linearly separable")
        print(f"  Best accuracy: {best_acc:.3f}")
        print(f"  → Try nonlinear probe or different state definitions")
        verdict = "PSI_FAILS"

    # Save
    output = {
        "model": model_name,
        "n_traces": n_traces,
        "verdict": verdict,
        "best_layer": best_layer,
        "layer_results": {str(k): v for k, v in layer_results.items()},
        "multi_probe_disagreement": disagreement,
    }

    results_path = output_dir / f"e02_results_{model_name.replace('/', '_')}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: Can we build ψ?")
    parser.add_argument("--model", type=str, default="pythia-1.4b")
    parser.add_argument("--n-traces", type=int, default=300)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        n_traces=args.n_traces,
        device=args.device,
        output_dir=args.output_dir,
    )
