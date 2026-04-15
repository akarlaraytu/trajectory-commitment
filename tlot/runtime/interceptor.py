"""
TLoT Runtime Interceptor

This is the full system: Φ + π + ψ working together during inference.

Architecture:
    For each layer l in intervention_layers:
        1. ψ(h_l) → P(s | h_l)           # observe reasoning state
        2. s = argmax P(s) if conf > θ     # decide state (or UNDEFINED)
        3. Φ(s) → forbidden directions     # look up forbidden subspace
        4. h_l' = π_λ(h_l, Φ(s))          # project out forbidden directions
        5. normalize h_l' to ||h_l||       # preserve norm

    If ψ is uncertain (max_conf < θ):
        → either HALT or skip intervention (configurable)

This module provides the hook that integrates into TransformerLens.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TLoTState:
    """Tracks the runtime state of TLoT across layers."""
    current_phase: Optional[int] = None
    confidence: float = 0.0
    n_interventions: int = 0
    n_skipped_uncertain: int = 0
    n_halted: int = 0
    n_fallback_used: int = 0  # how often static Φ was used instead of state-dependent
    forbidden_removed: list[float] = field(default_factory=list)

    @property
    def fallback_ratio(self) -> float:
        """% of interventions that used fallback instead of ψ-guided Φ."""
        total = self.n_interventions + self.n_fallback_used
        return self.n_fallback_used / max(total, 1)

    @property
    def effective_psi(self) -> bool:
        """Is ψ actually contributing, or is the system running on static Φ?"""
        return self.fallback_ratio < 0.5


class TLoTInterceptor:
    """
    The full TLoT system as a TransformerLens hook manager.

    Components:
        phi_directions: dict mapping state_id → (k, d) tensor of forbidden directions
        probe_weights: (n_states, d) linear probe weight matrix
        probe_bias: (n_states,) linear probe bias

    This is what you get after running e00 (find Φ), e01 (validate π),
    and e02 (train ψ). This class ties them together.
    """

    def __init__(
        self,
        phi_directions: dict[int, torch.Tensor],  # state_id → (k, d) forbidden dirs
        probe_weights: torch.Tensor,  # (n_states, d) from trained ψ
        probe_bias: torch.Tensor,  # (n_states,)
        intervention_layers: list[int],
        projection_lambda: float = 0.8,
        confidence_threshold: float = 0.7,
        preserve_norm: bool = True,
        halt_on_uncertainty: bool = False,
        fallback_phi: Optional[torch.Tensor] = None,  # static Φ when ψ is uncertain
    ):
        self.phi_directions = phi_directions
        self.probe_weights = probe_weights
        self.probe_bias = probe_bias
        self.intervention_layers = intervention_layers
        self.lam = projection_lambda
        self.conf_threshold = confidence_threshold
        self.preserve_norm = preserve_norm
        self.halt_on_uncertainty = halt_on_uncertainty
        self.fallback_phi = fallback_phi

        # Runtime state tracking
        self.state = TLoTState()
        self.trace: list[dict] = []  # full execution trace for analysis

    def _probe(self, h: torch.Tensor) -> tuple[int, float, torch.Tensor]:
        """
        ψ(h) → (predicted_state, confidence, probabilities)

        h shape: (d,) — single token's hidden state
        """
        device = h.device
        W = self.probe_weights.to(device=device, dtype=h.dtype)
        b = self.probe_bias.to(device=device, dtype=h.dtype)

        logits = h @ W.T + b  # (n_states,)
        probs = F.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)

        return predicted.item(), confidence.item(), probs

    def _get_phi(self, state_id: int, device, dtype) -> Optional[torch.Tensor]:
        """Look up forbidden directions for a state."""
        if state_id in self.phi_directions:
            return self.phi_directions[state_id].to(device=device, dtype=dtype)
        return None

    def _project(
        self, h: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """
        π_λ(h, Φ) = normalize(h - λ · Proj_Φ(h))
        h shape: (..., d)
        phi shape: (k, d)
        """
        original_norm = h.norm(dim=-1, keepdim=True)

        # Orthonormalize Φ directions
        if phi.shape[0] > 1:
            Q, _ = torch.linalg.qr(phi.T)
            phi_orth = Q.T
        else:
            phi_orth = F.normalize(phi, dim=-1)

        # Project and remove
        h_proj = h @ phi_orth.T @ phi_orth
        h_safe = h - self.lam * h_proj

        if self.preserve_norm:
            safe_norm = h_safe.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            h_safe = h_safe * (original_norm / safe_norm)

        return h_safe

    def make_hook(self, layer_idx: int):
        """
        Create a TransformerLens hook function for a specific layer.

        Returns a closure that captures `self` and `layer_idx`.
        """
        def hook_fn(value, hook):
            """
            value shape: (batch, seq_len, d_model)
            We intervene on the LAST token position (autoregressive generation).
            """
            # Take last token's hidden state for probing
            h_last = value[:, -1, :]  # (batch, d_model)

            # For simplicity, probe on first batch element
            state_id, confidence, probs = self._probe(h_last[0])

            # Record trace
            trace_entry = {
                "layer": layer_idx,
                "predicted_state": state_id,
                "confidence": confidence,
                "probs": probs.detach().cpu().tolist(),
                "action": None,
            }

            if confidence < self.conf_threshold:
                self.state.n_skipped_uncertain += 1

                if self.halt_on_uncertainty:
                    self.state.n_halted += 1
                    trace_entry["action"] = "HALT"
                    self.trace.append(trace_entry)
                    return value  # no intervention

                if self.fallback_phi is not None:
                    # Use static Φ as fallback
                    phi = self.fallback_phi.to(device=value.device, dtype=value.dtype)
                    self.state.n_fallback_used += 1
                    trace_entry["action"] = "FALLBACK_PHI"
                else:
                    trace_entry["action"] = "SKIP"
                    self.trace.append(trace_entry)
                    return value
            else:
                phi = self._get_phi(state_id, value.device, value.dtype)
                if phi is None:
                    trace_entry["action"] = "NO_PHI_FOR_STATE"
                    self.trace.append(trace_entry)
                    return value

            # Apply projection to ALL token positions
            # (because attention can propagate forbidden information)
            value_safe = self._project(value, phi)
            self.state.n_interventions += 1

            # Measure removal
            h_last_safe = value_safe[:, -1, :]
            phi_norm = F.normalize(phi, dim=-1)
            component_before = torch.abs(h_last[0] @ phi_norm.T).mean().item()
            component_after = torch.abs(h_last_safe[0] @ phi_norm.T).mean().item()
            self.state.forbidden_removed.append(component_before - component_after)

            trace_entry["action"] = "PROJECT"
            trace_entry["forbidden_removed"] = component_before - component_after
            self.trace.append(trace_entry)

            self.state.current_phase = state_id
            self.state.confidence = confidence

            return value_safe

        return hook_fn

    def attach_to_model(self, model):
        """
        Attach TLoT hooks to a TransformerLens model.
        Call model.reset_hooks() before this if needed.
        """
        for layer_idx in self.intervention_layers:
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            model.add_hook(hook_name, self.make_hook(layer_idx))
        return model

    def reset(self):
        """Reset runtime state for a new inference run."""
        self.state = TLoTState()
        self.trace.clear()

    def get_summary(self) -> dict:
        """Summary of what happened during inference."""
        return {
            "n_interventions": self.state.n_interventions,
            "n_skipped_uncertain": self.state.n_skipped_uncertain,
            "n_halted": self.state.n_halted,
            "n_fallback_used": self.state.n_fallback_used,
            "fallback_ratio": self.state.fallback_ratio,
            "psi_effective": self.state.effective_psi,
            "mean_forbidden_removed": (
                float(np.mean(self.state.forbidden_removed))
                if self.state.forbidden_removed else 0.0
            ),
            "last_state": self.state.current_phase,
            "last_confidence": self.state.confidence,
            "trace_length": len(self.trace),
        }


def create_interceptor_from_experiments(
    phi_npy_path: str,
    phi_layer: int,
    probe_state_dict_path: Optional[str] = None,
    d_model: int = 2048,
    n_states: int = 4,
    projection_lambda: float = 0.8,
    confidence_threshold: float = 0.7,
) -> TLoTInterceptor:
    """
    Convenience: build a TLoTInterceptor from experiment outputs.

    If no probe is available yet, creates a dummy probe that always
    returns state=0 with confidence=1.0 (static Φ mode).
    """
    # Load Φ
    phi_dir = np.load(phi_npy_path)
    phi_tensor = torch.tensor(phi_dir, dtype=torch.float32).unsqueeze(0)  # (1, d)

    # For now: same Φ for all states (will be state-dependent after e02)
    phi_directions = {i: phi_tensor for i in range(n_states)}

    # Load or create dummy probe
    if probe_state_dict_path is not None:
        state_dict = torch.load(probe_state_dict_path, map_location="cpu")
        probe_weights = state_dict["linear.weight"]  # (n_states, d)
        probe_bias = state_dict["linear.bias"]  # (n_states,)
    else:
        # Dummy probe: always returns state 0 with high confidence
        probe_weights = torch.zeros(n_states, d_model)
        probe_weights[0, 0] = 10.0  # bias toward state 0
        probe_bias = torch.zeros(n_states)
        probe_bias[0] = 10.0

    return TLoTInterceptor(
        phi_directions=phi_directions,
        probe_weights=probe_weights,
        probe_bias=probe_bias,
        intervention_layers=[phi_layer],
        projection_lambda=projection_lambda,
        confidence_threshold=confidence_threshold,
        fallback_phi=phi_tensor,  # use static Φ as fallback
    )
