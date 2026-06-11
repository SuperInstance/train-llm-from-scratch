import torch
import torch.nn as nn
from typing import List, Tuple
from src.models.attention import MultiHeadAttention, TernaryMultiHeadAttention
from src.models.mlp import MLP, TernaryMLP
from src.confidence import Zone, get_zone


class Block(nn.Module):
    """
    A single Transformer block.

    This block consists of a multi-head attention layer followed by an MLP,
    with layer normalization and residual connections.

    Args:
        n_head (int): The number of attention heads in the multi-head attention layer.
        n_embed (int): The dimensionality of the input embedding.
        context_length (int): The maximum length of the input sequence.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        """
        Initializes the Transformer block.

        Args:
            n_head (int): The number of attention heads.
            n_embed (int): The dimensionality of the embedding space.
            context_length (int): The maximum sequence length.
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_head, n_embed, context_length)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the block.
        """
        # Apply multi-head attention with residual connection
        x = x + self.attn(self.ln1(x))
        # Apply MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_embedding(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass focusing on the embedding and attention parts.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output after MLP embedding and the residual.
        """
        res = x + self.attn(self.ln1(x))
        x = self.mlp.forward_embedding(self.ln2(res))
        return x, res


class TernaryBlock(Block):
    """
    A Transformer block with ternary weights and confidence cascade.

    Uses TernaryMultiHeadAttention and TernaryMLP, providing per-block
    confidence scoring, traceability, and conservation regularisation.

    Args:
        n_head (int): The number of attention heads.
        n_embed (int): The dimensionality of the embedding space.
        context_length (int): The maximum sequence length.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        super().__init__(n_head, n_embed, context_length)
        # Override with ternary components
        self.attn = TernaryMultiHeadAttention(n_head, n_embed, context_length)
        self.mlp = TernaryMLP(n_embed)
        self._block_confidence: float = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with confidence tracking.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the block.
        """
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out

        # Overall block confidence: harmonic mean of attn head confidences
        with torch.no_grad():
            attn_conf = [h.raw_confidence for h in self.attn.heads]
            if attn_conf:
                # Mean of head confidences
                self._block_confidence = sum(attn_conf) / len(attn_conf)
            else:
                self._block_confidence = self.mlp.confidence

        return x

    @property
    def confidence(self) -> float:
        """Return the overall block confidence [0, 1]."""
        return self._block_confidence

    @property
    def zone(self) -> Zone:
        """Return the confidence Zone for this block."""
        return get_zone(self._block_confidence)

    @property
    def trace(self) -> List[str]:
        """Return the decision trace for this block."""
        traces = []
        for i, head in enumerate(self.attn.heads):
            h_trace = head.trace_attention(top_k=2)
            if h_trace:
                traces.append(f"head_{i}: zone={head.confidence_score.name}, conf={head.raw_confidence:.4f}")
        traces.append(f"mlp: zone={self.mlp.zone.name}, conf={self.mlp.confidence:.4f}")
        traces.append(f"block_zone={self.zone.name}, conf={self._block_confidence:.4f}")
        return traces


if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_len = 5
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    transformer_block = Block(n_head=num_heads, n_embed=embedding_dim, context_length=context_len)
    output_tensor = transformer_block(input_tensor)

    print("Transformer Block Input Shape:", input_tensor.shape)
    print("Transformer Block Output Shape:", output_tensor.shape)