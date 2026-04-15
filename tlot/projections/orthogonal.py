"""
Orthogonal projection operations for TLoT.

These are the π functions that remove forbidden directions
from hidden state vectors during inference.
"""

import torch
import torch.nn.functional as F


def find_contrastive_direction(
    activations_positive: torch.Tensor,
    activations_negative: torch.Tensor,
    method: str = "mean_diff",
) -> torch.Tensor:
    """
    Find the direction in activation space that separates
    positive (correct) from negative (hallucinated) activations.

    This is how we DISCOVER Φ empirically.

    Args:
        activations_positive: (n_pos, d) — correct reasoning activations
        activations_negative: (n_neg, d) — hallucinated/wrong activations
        method: "mean_diff" or "pca"

    Returns:
        direction: (d,) unit vector pointing toward hallucination
    """
    if method == "mean_diff":
        # Simplest: the difference of means
        # This is what RepE (Zou et al. 2023) uses as baseline
        mean_pos = activations_positive.mean(dim=0)
        mean_neg = activations_negative.mean(dim=0)
        direction = mean_neg - mean_pos
        direction = F.normalize(direction, dim=0)
        return direction

    elif method == "pca":
        # PCA on the difference: find the principal axis of separation
        # More robust when the separation isn't purely linear
        diffs = activations_negative.mean(dim=0, keepdim=True) - activations_positive
        # SVD to find principal direction
        U, S, Vh = torch.linalg.svd(diffs, full_matrices=False)
        direction = Vh[0]  # first principal component
        return F.normalize(direction, dim=0)

    else:
        raise ValueError(f"Unknown method: {method}")


def soft_project(
    h: torch.Tensor,
    forbidden_directions: torch.Tensor,
    lam: float = 1.0,
    preserve_norm: bool = True,
) -> torch.Tensor:
    """
    π_λ(h) = normalize(h - λ · Proj_Φ(h))

    Args:
        h: hidden states, shape (..., d)
        forbidden_directions: (k, d) unit vectors spanning forbidden subspace
        lam: projection strength ∈ [0, 1]
        preserve_norm: if True, rescale to original norm

    Returns:
        h_safe: projected hidden states, same shape as h
    """
    original_shape = h.shape
    if h.dim() == 1:
        h = h.unsqueeze(0)

    original_norm = h.norm(dim=-1, keepdim=True)

    # Project h onto each forbidden direction and subtract
    # Proj_V(h) = V^T (V V^T)^{-1} V h
    # For orthonormal V: Proj_V(h) = V^T V h
    V = forbidden_directions  # (k, d)

    # Ensure V rows are orthonormal via QR
    if V.shape[0] > 1:
        Q, _ = torch.linalg.qr(V.T)  # Q is (d, k)
        V_orth = Q.T  # (k, d)
    else:
        V_orth = F.normalize(V, dim=-1)

    # h_proj = V^T @ V @ h^T → for each h, its component in forbidden subspace
    h_proj = h @ V_orth.T @ V_orth  # (..., d)
    h_safe = h - lam * h_proj

    if preserve_norm:
        safe_norm = h_safe.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        h_safe = h_safe * (original_norm / safe_norm)

    if len(original_shape) == 1:
        h_safe = h_safe.squeeze(0)

    return h_safe


def measure_forbidden_component(
    h: torch.Tensor,
    forbidden_directions: torch.Tensor,
) -> torch.Tensor:
    """
    Measure how much of h lies in the forbidden subspace.
    Returns scalar ∈ [0, 1] where 0 = fully orthogonal, 1 = fully aligned.

    This is our diagnostic: after projection, this should be ~0.
    """
    V = F.normalize(forbidden_directions, dim=-1)
    if h.dim() == 1:
        h = h.unsqueeze(0)

    h_norm = F.normalize(h, dim=-1)

    # For each forbidden direction, compute alignment
    # shape: (..., k)
    alignments = h_norm @ V.T
    # Total forbidden component = sqrt(sum of squared alignments)
    forbidden_magnitude = alignments.pow(2).sum(dim=-1).sqrt()

    return forbidden_magnitude
