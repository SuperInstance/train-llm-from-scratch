import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple
from src.confidence import Zone, get_zone


class Head(nn.Module):
    """
    A single attention head.

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

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    This module combines multiple attention heads in parallel. The outputs of each head
    are concatenated and passed through a final linear projection to form the output.

    Args:
        n_head (int): The number of parallel attention heads.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        """
        Initializes the multi-head attention module.

        Args:
            n_head (int): The number of parallel attention heads.
            n_embed (int): The dimensionality of the input embedding.
            context_length (int): The maximum length of the input sequence.
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed // n_head, n_embed, context_length) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.head_size = n_embed // n_head

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


# Alias: FloatHead for clarity alongside ternary alternatives.
FloatHead = Head


class TernaryHead(Head):
    """
    A single attention head with confidence cascade — SuperInstance refactor.

    Inherits from Head and adds:
      - Confidence scoring after softmax (GREEN ≥ 0.90, YELLOW ≥ 0.75, RED < 0.75)
      - Trace recording of top-k attended positions
      - confidence_score property returning the Zone

    Args:
        head_size (int): The dimensionality of the key, query, and value projections.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence, used for causal masking.
    """
    def __init__(self, head_size: int, n_embed: int, context_length: int) -> None:
        super().__init__(head_size, n_embed, context_length)
        self._last_attn_weights: Optional[torch.Tensor] = None
        self._last_confidence: float = 1.0
        self._trace: List[Tuple[int, int, float]] = []  # (batch, pos, score)
        self.register_buffer('_tril_indices', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ternary attention head with confidence scoring.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        B, T, C = x.shape
        head_size = self.key.out_features
        k = self.key(x)
        q = self.query(x)
        scale_factor = 1 / math.sqrt(head_size)
        attn_weights = q @ k.transpose(-2, -1) * scale_factor
        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Confidence cascade: average maximum attention weight
        max_attn = attn_weights.max(dim=-1).values  # (B, T)
        self._last_confidence = max_attn.mean().item()
        self._last_attn_weights = attn_weights.detach()

        v = self.value(x)
        out = attn_weights @ v
        return out

    def trace_attention(self, top_k: int = 3) -> List[Tuple[int, int, float]]:
        """
        Record the top-k attended positions for each batch item.

        Args:
            top_k (int): Number of top attended positions to record.

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

    @property
    def confidence_score(self) -> Zone:
        """Return the confidence Zone for this head's last forward pass."""
        return get_zone(self._last_confidence)

    @property
    def raw_confidence(self) -> float:
        """Return the raw confidence score [0, 1]."""
        return self._last_confidence


class TernaryMultiHeadAttention(MultiHeadAttention):
    """
    Multi-Head Attention using TernaryHead — SuperInstance refactor.

    Uses TernaryHead instances instead of standard Heads, providing
    confidence cascade and trace capabilities per head.

    Args:
        n_head (int): The number of parallel attention heads.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        super().__init__(n_head, n_embed, context_length)
        # Replace heads with TernaryHead instances
        head_size = n_embed // n_head
        self.heads = nn.ModuleList([
            TernaryHead(head_size, n_embed, context_length) for _ in range(n_head)
        ])

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
