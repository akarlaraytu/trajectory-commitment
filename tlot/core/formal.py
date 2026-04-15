"""
TLoT Formal Core — ⟨S, ⇒, Φ, π, ψ⟩

This module defines the mathematical objects of TLoT.
No ML dependencies — pure math + numpy.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np


class ReasoningState(Enum):
    """S: The set of logical reasoning states."""
    IDLE = auto()
    PARSING = auto()
    ARITHMETIC = auto()
    FACTUAL_RECALL = auto()
    SOURCE_CITATION = auto()
    LOGICAL_DEDUCTION = auto()
    SYNTHESIS = auto()
    OUTPUT = auto()


@dataclass
class ForbiddenSubspace:
    """
    Φ(s): For a given state s, the forbidden directions in R^d.

    Stored as a matrix V of shape (k, d) where each row is a
    forbidden direction (unit vector). The subspace spanned by
    these rows is what gets projected out.

    This is the object we need to LEARN. If Φ doesn't exist
    as a clean subspace, TLoT doesn't work. Experiment e00
    tests exactly this.
    """
    state: ReasoningState
    directions: np.ndarray  # shape (k, d), each row is unit vector
    confidence: float = 0.0  # how confident we are this Φ is real

    @property
    def rank(self) -> int:
        """Dimensionality of the forbidden subspace."""
        return self.directions.shape[0]

    @property
    def d(self) -> int:
        """Ambient space dimension."""
        return self.directions.shape[1]

    def projection_matrix(self) -> np.ndarray:
        """
        P = V^T V  (projection onto forbidden subspace)
        Used as: h_safe = h - P @ h
        """
        V = self.directions  # (k, d)
        return V.T @ V  # (d, d)


@dataclass
class SoftProjection:
    """
    π_λ(h, s) = normalize(h - λ · Proj_{Φ(s)}(h))

    Key design decisions (from critical analysis):
    1. λ ∈ [0, 1] — soft, not hard projection
    2. Norm preservation — prevents gradient collapse
    3. State-dependent — different Φ for different reasoning states
    """
    lam: float = 1.0  # projection strength, 0 = no effect, 1 = full removal

    def project(
        self,
        h: np.ndarray,
        phi: ForbiddenSubspace,
        preserve_norm: bool = True,
    ) -> np.ndarray:
        """
        Apply soft orthogonal projection.

        h: hidden state vector, shape (d,)
        phi: forbidden subspace for current state
        preserve_norm: if True, rescale to original norm (prevents collapse)

        Returns: h_safe, shape (d,)
        """
        P = phi.projection_matrix()  # (d, d)
        h_proj = P @ h  # component in forbidden subspace
        h_safe = h - self.lam * h_proj

        if preserve_norm:
            original_norm = np.linalg.norm(h)
            safe_norm = np.linalg.norm(h_safe)
            if safe_norm > 1e-8:
                h_safe = h_safe * (original_norm / safe_norm)

        return h_safe

    def project_batch(
        self,
        H: np.ndarray,
        phi: ForbiddenSubspace,
        preserve_norm: bool = True,
    ) -> np.ndarray:
        """
        Apply to batch of hidden states. H shape: (batch, d)
        """
        P = phi.projection_matrix()
        H_proj = H @ P.T
        H_safe = H - self.lam * H_proj

        if preserve_norm:
            orig_norms = np.linalg.norm(H, axis=1, keepdims=True)
            safe_norms = np.linalg.norm(H_safe, axis=1, keepdims=True)
            mask = safe_norms > 1e-8
            H_safe = np.where(mask, H_safe * (orig_norms / np.maximum(safe_norms, 1e-8)), H_safe)

        return H_safe


@dataclass
class ProbeResult:
    """
    ψ(h) = P(s | h)  — distributional, not deterministic.

    UNDEFINED conditions (either triggers HALT/fallback):
    1. max P(s|h) < confidence_threshold  → not confident enough
    2. H(P(s|h)) > entropy_threshold      → distribution too flat

    Multi-probe disagreement lemma:
    If ψ_1(h) ≠ ψ_2(h) → state(h) is undefined → π halts.
    """
    probabilities: dict[ReasoningState, float]
    confidence_threshold: float = 0.7
    entropy_threshold: float = 1.0  # nats; log(4)≈1.39 = max entropy for 4 states

    @property
    def entropy(self) -> float:
        """Shannon entropy of the state distribution (nats)."""
        probs = np.array(list(self.probabilities.values()))
        probs = probs[probs > 1e-10]  # avoid log(0)
        return float(-np.sum(probs * np.log(probs)))

    @property
    def predicted_state(self) -> Optional[ReasoningState]:
        """Returns predicted state if both confidence AND entropy pass."""
        best_state = max(self.probabilities, key=self.probabilities.get)
        if (self.probabilities[best_state] >= self.confidence_threshold
                and self.entropy <= self.entropy_threshold):
            return best_state
        return None

    @property
    def is_uncertain(self) -> bool:
        return self.predicted_state is None

    @property
    def uncertainty_reason(self) -> Optional[str]:
        """Why is the probe uncertain? For diagnostics/logging."""
        if not self.is_uncertain:
            return None
        best_state = max(self.probabilities, key=self.probabilities.get)
        reasons = []
        if self.probabilities[best_state] < self.confidence_threshold:
            reasons.append(f"low_confidence({self.probabilities[best_state]:.3f}<{self.confidence_threshold})")
        if self.entropy > self.entropy_threshold:
            reasons.append(f"high_entropy({self.entropy:.3f}>{self.entropy_threshold})")
        return "+".join(reasons)

    @property
    def max_confidence(self) -> float:
        return max(self.probabilities.values())


@dataclass
class TLoTConfig:
    """Configuration for a TLoT runtime instance."""
    projection_lambda: float = 0.8  # soft projection strength
    probe_confidence_threshold: float = 0.7  # below this → undefined state
    intervention_layers: list[int] = field(default_factory=lambda: [])  # which layers to intervene on
    preserve_norm: bool = True
    halt_on_uncertainty: bool = True  # if probe uncertain, stop intervening

    def validate(self):
        assert 0.0 <= self.projection_lambda <= 1.0
        assert 0.0 <= self.probe_confidence_threshold <= 1.0
