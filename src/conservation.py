"""
Conservation Regularization Module — SuperInstance scientific refactor.

Implements conservation loss (Σ Δ_activations ≈ 0) and spectral
normalization with Cheeger-type gating for ternary-weight architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def conservation_loss(activations: Tensor, dim: int = -1) -> Tensor:
    """
    Conservation loss: Σ(Δ_activations) ≈ 0.

    For a sequence of activations along dimension `dim`, compute the
    sum of successive deltas and penalise any deviation from zero.
    This enforces a "closed gesture" — the computation returns to its
    origin, encouraging energy-preserving transformations.

    Args:
        activations: Tensor of shape (..., seq_len, ...).
        dim:         Dimension along which to compute deltas.

    Returns:
        A scalar loss tensor.
    """
    # Compute differences along the given dimension
    diffs = torch.diff(activations, dim=dim)  # (..., seq_len-1, ...)
    # Penalise the sum of deltas (should be ≈ 0)
    delta_sum = diffs.sum(dim=dim)
    loss = (delta_sum ** 2).mean()
    return loss


def spectral_normalize(
    weights: Tensor,
    cheeger_threshold: float = 0.5,
) -> Tensor:
    """
    Soft-gate weight magnitudes based on a Cheeger-type connectivity measure.

    Given a weight tensor, this computes the row-wise L2 norm as a proxy
    for spectral connectivity. Weights with norm below `cheeger_threshold`
    are softly gated (scaled down via a sigmoid), preserving only the
    strongly-connected components. This emulates the Cheeger constant
    idea from spectral graph theory: only well-connected paths survive.

    Args:
        weights:            Weight tensor of shape (out_features, in_features).
        cheeger_threshold:  Connectivity threshold for soft-gating.

    Returns:
        Normalised weight tensor with the same shape.
    """
    # Compute row-wise L2 norm (out_features,)
    row_norms = weights.norm(dim=1, keepdim=True)  # (out_features, 1)
    # Sigmoid gate: rows with norm >> threshold stay near 1;
    # rows with norm << threshold get suppressed.
    gate = torch.sigmoid(row_norms - cheeger_threshold)
    # Apply gating
    gated_weights = weights * gate
    return gated_weights


def ternary_quantize(weights: Tensor, scale: Optional[Tensor] = None) -> Tensor:
    """
    Quantise a weight tensor to ternary values {-1, 0, +1}.

    Uses a learnable scale factor; weights within the dead-zone [-0.5, 0.5]
    (after scaling) are set to 0, positive values become +1, negative -1.

    Args:
        weights: Full-precision weight tensor.
        scale:   Optional per-tensor scale factor. If None, uses 1.0.

    Returns:
        Ternary-quantised weight tensor.
    """
    if scale is None:
        scale = weights.abs().mean() + 1e-8
    normalised = weights / scale
    # Straight-through estimator: ternary sign with STE gradients
    ternary = torch.where(
        normalised.abs() < 0.5,
        torch.zeros_like(normalised),
        torch.sign(normalised),
    )
    # Straight-through estimator: pass gradient through (identity for grad,
    # ternary for forward)
    return ternary.detach() + weights - weights.detach()


class TernaryLinear(nn.Module):
    """
    A linear layer with ternary weights {-1, 0, +1} and optional scale.

    Maintains full-precision weights for gradient updates but uses
    ternary weights in the forward pass (straight-through estimator).

    Args:
        in_features:  Number of input features.
        out_features: Number of output features.
        bias:         Whether to include a bias term.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with ternary-quantised weights.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        w_ternary = ternary_quantize(self.weight, self.scale)
        out = F.linear(x, w_ternary, self.bias)
        return out
