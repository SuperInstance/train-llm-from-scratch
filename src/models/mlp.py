"""
Multi-Layer Perceptron (MLP) module.

Provides both a standard float-precision MLP (FloatMLP) and a ternary-weight
variant (TernaryMLP) with conservation regularization, confidence tracking,
and POLLN-style tile interfaces for the SuperInstance architecture.

The original ``MLP`` name is preserved as a backward-compatible alias that
defaults to ``FloatMLP``, with an optional ``ternary`` switch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, List, Optional, Any
from src.conservation import conservation_loss, ternary_quantize
from src.confidence import ConfidenceTile, get_zone, Zone


class FloatMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron with one hidden layer.

    This module is used within the Transformer block for feed-forward processing.
    It expands the input embedding size, applies a ReLU activation, and then projects it back
    to the original embedding size.

    Args:
        n_embed (int): The dimensionality of the input embedding.
    """
    def __init__(self, n_embed: int) -> None:
        """
        Initializes the FloatMLP module.

        Args:
            n_embed (int): The dimensionality of the input embedding.
        """
        super().__init__()
        self.hidden = nn.Linear(n_embed, 4 * n_embed)  # Linear layer to expand embedding size
        self.relu = nn.ReLU()                        # ReLU activation function
        self.proj = nn.Linear(4 * n_embed, n_embed)  # Linear layer to project back to original size

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the FloatMLP.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C), where B is batch size,
                              T is sequence length, and C is embedding size.

        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        x = self.forward_embedding(x)
        x = self.project_embedding(x)
        return x

    def forward_embedding(self, x: Tensor) -> Tensor:
        """
        Applies the hidden linear layer followed by ReLU activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the hidden layer and ReLU.
        """
        x = self.relu(self.hidden(x))
        return x

    def project_embedding(self, x: Tensor) -> Tensor:
        """
        Applies the projection linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the projection layer.
        """
        x = self.proj(x)
        return x


# --- Backward-compatible alias ---
MLP = FloatMLP


class TernaryMLP(nn.Module):
    """
    Ternary-Weighted MLP — SuperInstance scientific refactor.

    Replaces float Linear weights with ternary {-1, 0, +1} via scale/ternary split
    using a straight-through estimator. Adds:
      - Conservation regularisation (Σ Δ_activations ≈ 0)
      - ConfidenceTile interface for traceability and composition

    Args:
        n_embed (int): The dimensionality of the input embedding.
    """
    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.n_embed = n_embed
        # Store full-precision weights; forward uses ternary versions
        self.hidden_weight = nn.Parameter(torch.empty(4 * n_embed, n_embed))
        self.hidden_bias = nn.Parameter(torch.zeros(4 * n_embed))
        self.proj_weight = nn.Parameter(torch.empty(n_embed, 4 * n_embed))
        self.proj_bias = nn.Parameter(torch.zeros(n_embed))
        self.scale_hidden = nn.Parameter(torch.ones(1))
        self.scale_proj = nn.Parameter(torch.ones(1))
        self.relu = nn.ReLU()

        # Initialise weights with proper scaling
        nn.init.kaiming_uniform_(self.hidden_weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.proj_weight, a=5 ** 0.5)

        # Tile interface tracking
        self._last_input: Optional[Tensor] = None
        self._last_hidden: Optional[Tensor] = None
        self._last_output: Optional[Tensor] = None
        self._last_confidence: float = 1.0
        self._trace_steps: List[str] = []

    def _ternary_hidden(self, x: Tensor) -> Tensor:
        """Ternary-quantised hidden layer: W_hidden @ x + bias."""
        w = ternary_quantize(self.hidden_weight, self.scale_hidden)
        return F.linear(x, w, self.hidden_bias)

    def _ternary_proj(self, x: Tensor) -> Tensor:
        """Ternary-quantised projection layer: W_proj @ x + bias."""
        w = ternary_quantize(self.proj_weight, self.scale_proj)
        return F.linear(x, w, self.proj_bias)

    def forward_embedding(self, x: Tensor) -> Tensor:
        """
        Apply the ternary hidden layer with ReLU.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Hidden activations after ternary linear + ReLU.
        """
        self._last_input = x.detach()
        hidden = self._ternary_hidden(x)
        hidden = self.relu(hidden)
        self._last_hidden = hidden.detach()
        return hidden

    def project_embedding(self, x: Tensor) -> Tensor:
        """
        Apply the ternary projection layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output after ternary projection.
        """
        out = self._ternary_proj(x)
        self._last_output = out.detach()
        return out

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the ternary MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor of the same shape as input.
        """
        self._trace_steps = ['ternary_hidden', 'relu', 'ternary_proj']
        hidden = self.forward_embedding(x)
        out = self.project_embedding(hidden)

        # Compute confidence from activation magnitude distribution
        with torch.no_grad():
            act_norm = out.norm().item() / max(out.numel(), 1)
            self._last_confidence = min(1.0, max(0.0, act_norm / max(act_norm, 0.01)))

        return out

    def conservation_loss(self) -> Tensor:
        """
        Compute conservation regularisation on this MLP's activations.

        Returns:
            Scalar conservation loss (0 if no hidden activations recorded).
        """
        if self._last_hidden is None:
            return torch.tensor(0.0, device=self.hidden_weight.device)
        return conservation_loss(self._last_hidden, dim=-1)

    # --- ConfidenceTile interface ---

    @property
    def input_type(self) -> str:
        return f"embedding({self.n_embed})"

    @property
    def output_type(self) -> str:
        return f"embedding({self.n_embed})"

    @property
    def confidence(self) -> float:
        return self._last_confidence

    @property
    def zone(self) -> Zone:
        return get_zone(self._last_confidence)

    @property
    def trace(self) -> List[str]:
        return list(self._trace_steps)

    def as_tile(self) -> ConfidenceTile:
        """Wrap this TernaryMLP as a ConfidenceTile for composition."""
        return ConfidenceTile(
            input_type=self.input_type,
            output_type=self.output_type,
            function=self.forward,
            confidence=lambda _: self.confidence,
            trace=self.trace,
        )


# --- Convenience factory ---
def make_mlp(n_embed: int, ternary: bool = False) -> nn.Module:
    """
    Factory function — returns either a FloatMLP or a TernaryMLP.

    Args:
        n_embed: The embedding dimensionality.
        ternary: If True, return a ``TernaryMLP``; otherwise a ``FloatMLP``.

    Returns:
        An ``nn.Module`` instance of the requested type.
    """
    return TernaryMLP(n_embed) if ternary else FloatMLP(n_embed)


if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 3
    embedding_dim = 16
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    # Test FloatMLP (via the backward-compatible MLP alias)
    mlp_module = MLP(n_embed=embedding_dim)
    output_tensor = mlp_module(input_tensor)

    print("FloatMLP (alias MLP) Input Shape:", input_tensor.shape)
    print("FloatMLP (alias MLP) Output Shape:", output_tensor.shape)

    # Test TernaryMLP
    tmlp = TernaryMLP(n_embed=embedding_dim)
    out_t = tmlp(input_tensor)
    print("TernaryMLP Input Shape:", input_tensor.shape)
    print("TernaryMLP Output Shape:", out_t.shape)
    print("TeraryMLP confidence:", tmlp.confidence)
    print("TernaryMLP zone:", tmlp.zone)
    print("TernaryMLP trace:", tmlp.trace)
