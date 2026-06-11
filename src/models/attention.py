import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from src.confidence import Zone, get_zone

# Try to import conservation regularization (optional)
try:
    from src.conservation import conservation_loss, ternary_quantize, spectral_normalize
    _conservation_available = True
except ImportError:
    _conservation_available = False


class FloatHead(nn.Module):
    """
    A single attention head with full-precision (float) weights.

    This module calculates attention scores and applies them to the values.
    It includes key, query, and value projections, and uses causal masking
    to prevent attending to future tokens.

    Args:
        head_size (int): The dimensionality of the key, query, and value projections.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence, used for causal masking.
    """
    def __init__(self, head_size: int, n_embed: int, context_length: int) -> None:
        """
        Initializes the attention head.

        Args:
            head_size (int): The dimensionality of the key, query, and value projections.
            n_embed (int): The dimensionality of the input embedding.
            context_length (int): The maximum length of the input sequence.
        """
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)   # Key projection
        self.query = nn.Linear(n_embed, head_size, bias=False) # Query projection
        self.value = nn.Linear(n_embed, head_size, bias=False) # Value projection
        # Lower triangular matrix for causal masking
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B, T, C = x.shape
        head_size = self.key.out_features
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        scale_factor = 1 / math.sqrt(head_size)
        # Calculate attention weights: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        attn_weights = q @ k.transpose(-2, -1) * scale_factor
        # Apply causal masking
        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        v = self.value(x)   # (B, T, head_size)
        # Apply attention weights to values
        out = attn_weights @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


# Backward-compatible alias
Head = FloatHead


class TernaryHead(FloatHead):
    """
    A single attention head with ternary-quantized weights and confidence cascade.

    Extends FloatHead with:
      - Ternary quantization of key/query/value weights to {-1, 0, +1}
        using straight-through estimator from src/conservation.ternary_quantize.
      - Confidence cascade after softmax (GREEN >= 0.90, YELLOW >= 0.75, RED < 0.75).
      - trace_attention() method for explainability (top-k attended positions).
      - Conservation regularization via src.conservation.conservation_loss when available.

    Args:
        head_size (int): The dimensionality of the key, query, and value projections.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence, used for causal masking.
    """
    def __init__(self, head_size: int, n_embed: int, context_length: int) -> None:
        super().__init__(head_size, n_embed, context_length)
        self._last_attn_weights: Optional[torch.Tensor] = None
        self._last_confidence: float = 1.0
        self._trace: List[Tuple[int, int, float]] = []
        self._last_input: Optional[torch.Tensor] = None
        self._last_output: Optional[torch.Tensor] = None

    def _ternary_forward(self, x: torch.Tensor, weight: nn.Parameter) -> torch.Tensor:
        """Compute linear transformation with ternary-quantized weights."""
        if _conservation_available:
            w_ternary = ternary_quantize(weight)
        else:
            # Fallback: simple sign-based ternary quantization
            scale = weight.abs().mean() + 1e-8
            normalised = weight / scale
            w_ternary = torch.where(
                normalised.abs() < 0.5,
                torch.zeros_like(normalised),
                torch.sign(normalised),
            )
            w_ternary = w_ternary.detach() + weight - weight.detach()
        return F.linear(x, w_ternary, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ternary attention head with confidence scoring.

        Key, query, and value projections use ternary-quantized weights.
        After softmax, confidence is computed as the mean of maximum attention
        probabilities, then mapped to a Zone (GREEN/YELLOW/RED).

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B, T, C = x.shape
        head_size = self.key.out_features
        scale_factor = 1 / math.sqrt(head_size)

        # Ternary-quantized projections
        k = self._ternary_forward(x, self.key.weight)      # (B, T, head_size)
        q = self._ternary_forward(x, self.query.weight)     # (B, T, head_size)

        # Attention weights with causal masking
        attn_weights = q @ k.transpose(-2, -1) * scale_factor
        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Confidence cascade: mean of max attention probability per token
        max_attn = attn_weights.max(dim=-1).values  # (B, T)
        self._last_confidence = max_attn.mean().item()
        self._last_attn_weights = attn_weights.detach()

        # Ternary-quantized value projection
        v = self._ternary_forward(x, self.value.weight)     # (B, T, head_size)
        out = attn_weights @ v

        self._last_input = x.detach()
        self._last_output = out.detach()

        return out

    def trace_attention(self, top_k: int = 3) -> List[Tuple[int, int, float]]:
        """
        Record the top-k attended positions for each batch item.

        Args:
            top_k (int): Number of top attended positions to record (default: 3).

        Returns:
            List of (batch_index, position, attention_score) tuples.
        """
        if self._last_attn_weights is None:
            return []
        attn = self._last_attn_weights  # (B, T, T)
        B, T, _ = attn.shape
        trace: List[Tuple[int, int, float]] = []
        for b in range(B):
            # For the last query token, get top-k attended positions
            last_query_attn = attn[b, -1, :]  # (T,)
            values, indices = torch.topk(last_query_attn, min(top_k, T))
            for pos, score in zip(indices.tolist(), values.tolist()):
                trace.append((b, pos, score))
        self._trace = trace
        return trace

    def conservation_loss(self) -> torch.Tensor:
        """
        Compute conservation loss on this head's last forward pass activations.

        Returns a scalar tensor penalising deviation from Σ(Δ_activations) ≈ 0.
        Falls back to returning 0.0 if conservation module is unavailable or
        no forward pass has been recorded yet.

        Returns:
            torch.Tensor: Scalar conservation loss.
        """
        if not _conservation_available or self._last_output is None:
            return torch.tensor(0.0)
        return conservation_loss(self._last_output, dim=1)

    @property
    def confidence_score(self) -> Zone:
        """Return the confidence Zone for this head's last forward pass."""
        return get_zone(self._last_confidence)

    @property
    def raw_confidence(self) -> float:
        """Return the raw confidence score [0, 1]."""
        return self._last_confidence


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    This module combines multiple attention heads in parallel. The outputs of each head
    are concatenated and passed through a final linear projection to form the output.

    Use the ``ternary`` toggle to switch between FloatHead and TernaryHead.

    Args:
        n_head (int): The number of parallel attention heads.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence.
        ternary (bool): If True, use TernaryHead (ternary-quantized weights with
                        confidence cascade). Defaults to False.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int, ternary: bool = False) -> None:
        """
        Initializes the multi-head attention module.

        Args:
            n_head (int): The number of parallel attention heads.
            n_embed (int): The dimensionality of the input embedding.
            context_length (int): The maximum length of the input sequence.
            ternary (bool): If True, use TernaryHead instead of FloatHead.
        """
        super().__init__()
        head_class = TernaryHead if ternary else FloatHead
        self.heads = nn.ModuleList([head_class(n_embed // n_head, n_embed, context_length) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.head_size = n_embed // n_head
        self.ternary = ternary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after concatenating the heads and applying the output projection.
        """
        # Concatenate the output of each head along the last dimension (C)
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply final linear projection
        x = self.proj(x)
        return x


class TernaryMultiHeadAttention(MultiHeadAttention):
    """
    Multi-Head Attention using TernaryHead.

    Convenience subclass of MultiHeadAttention that always uses TernaryHead,
    providing confidence cascade and trace capabilities per head.

    Args:
        n_head (int): The number of parallel attention heads.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        super().__init__(n_head, n_embed, context_length, ternary=True)

    def confidence_distribution(self) -> dict:
        """
        Return the distribution of confidence zones across all heads.

        Returns:
            dict with keys 'GREEN', 'YELLOW', 'RED' counting heads.
        """
        zones = {'GREEN': 0, 'YELLOW': 0, 'RED': 0}
        for head in self.heads:
            zone = head.confidence_score
            zones[zone.name] = zones.get(zone.name, 0) + 1
        return zones

    def trace_all_heads(self, top_k: int = 3) -> List[List[Tuple[int, int, float]]]:
        """
        Trace attention across all heads.

        Args:
            top_k (int): Top-k positions to trace per head.

        Returns:
            List of traces, one per head.
        """
        return [h.trace_attention(top_k=top_k) for h in self.heads]


if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_len = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    multihead_attn = MultiHeadAttention(n_head=num_heads, n_embed=embedding_dim, context_length=context_len)
    output_tensor = multihead_attn(input_tensor)

    print("MultiHeadAttention Input Shape:", input_tensor.shape)
    print("MultiHeadAttention Output Shape:", output_tensor.shape)

    # Test ternary variant
    ternary_mha = TernaryMultiHeadAttention(n_head=num_heads, n_embed=embedding_dim, context_length=context_len)
    tern_output = ternary_mha(input_tensor)
    print("\nTernaryMultiHeadAttention Output Shape:", tern_output.shape)
    print("Confidence distribution:", ternary_mha.confidence_distribution())
    print("Head traces:", ternary_mha.trace_all_heads(top_k=2))
