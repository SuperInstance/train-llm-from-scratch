import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from src.models.transformer_block import Block, TernaryBlock


class Transformer(nn.Module):
    """
    The main Transformer model.

    This class combines token and position embeddings with a sequence of Transformer blocks
    and a final linear layer for language modeling.

    When ``ternary_weights=True``, each block uses ``TernaryMultiHeadAttention`` and
    ``TernaryMLP``, enabling per-block confidence scoring, conservation regularisation,
    and full decision tracing.

    Args:
        n_head (int): The number of attention heads in each transformer block.
        n_embed (int): The dimensionality of the embedding space.
        context_length (int): The maximum length of the input sequence.
        vocab_size (int): The size of the vocabulary.
        N_BLOCKS (int): The number of transformer blocks in the model.
        ternary_weights (bool): If True, use TernaryBlock (TernaryMultiHeadAttention + TernaryMLP).
    """
    def __init__(
        self,
        n_head: int,
        n_embed: int,
        context_length: int,
        vocab_size: int,
        N_BLOCKS: int,
        ternary_weights: bool = False,
    ) -> None:
        """
        Initializes the Transformer model.

        Args:
            n_head (int): Number of attention heads.
            n_embed (int): Embedding dimension.
            context_length (int): Maximum sequence length.
            vocab_size (int): Size of the vocabulary.
            N_BLOCKS (int): Number of transformer blocks.
            ternary_weights (bool): If True, use ternary components.
        """
        super().__init__()
        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS
        self.ternary_weights = ternary_weights
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(context_length, n_embed)
        if ternary_weights:
            self.attn_blocks = nn.ModuleList([
                TernaryBlock(n_head, n_embed, context_length) for _ in range(N_BLOCKS)
            ])
        else:
            self.attn_blocks = nn.ModuleList([
                Block(n_head, n_embed, context_length) for _ in range(N_BLOCKS)
            ])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer('pos_idxs', torch.arange(context_length))

    def _pre_attn_pass(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Combines token and position embeddings.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Sum of token and position embeddings.
        """
        B, T = idx.shape
        tok_embedding = self.token_embed(idx)
        pos_embedding = self.position_embed(self.pos_idxs[:T])
        return tok_embedding + pos_embedding

    def forward_hidden(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and return the final hidden states AFTER the final layer norm.

        This is exactly the tensor that ``lm_head`` consumes, so it is the right
        representation for auxiliary heads added during post-training (a scalar value
        head for PPO, a scalar reward head for the reward model). Keeping it as a
        separate method lets those heads reuse the backbone without duplicating the
        forward logic or rewriting ``forward``.

        Args:
            idx (torch.Tensor): Input token indices, shape (B, T).

        Returns:
            torch.Tensor: Final hidden states, shape (B, T, n_embed).
        """
        x = self._pre_attn_pass(idx)
        for block in self.attn_blocks:
            x = block(x)
        return self.layer_norm(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the Transformer.

        Args:
            idx (torch.Tensor): Input token indices.
            targets (torch.Tensor, optional): Target token indices for loss calculation. Defaults to None.

        Returns:
            tuple: Logits and loss (if targets are provided).
        """
        x = self.forward_hidden(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            flat_logits = logits.view(B * T, C)
            targets = targets.view(B * T).long()
            loss = F.cross_entropy(flat_logits, targets)
        return logits, loss

    def get_conservation_loss(self) -> torch.Tensor:
        """
        Compute conservation loss from all ternary blocks.

        Only valid when ternary_weights=True. Returns sum of per-block
        conservation losses across all MLP layers.

        Returns:
            Scalar conservation loss.
        """
        if not self.ternary_weights:
            return torch.tensor(0.0, device=self.lm_head.weight.device)
        total = torch.tensor(0.0, device=self.lm_head.weight.device)
        for block in self.attn_blocks:
            if hasattr(block.mlp, 'conservation_loss'):
                total = total + block.mlp.conservation_loss()
        return total

    # ------------------------------------------------------------------
    # Per-block confidence tracking (ternary_weights only)
    # ------------------------------------------------------------------

    @property
    def confidence(self) -> float:
        """
        Overall model confidence (mean across all blocks).

        When ``ternary_weights=False``, returns 1.0 (full confidence).
        When ``ternary_weights=True``, returns the mean of all per-block
        confidence scores from TernaryBlock.

        Returns:
            Average confidence across all transformer blocks [0, 1].
        """
        if not self.ternary_weights:
            return 1.0
        confidences = [b.confidence for b in self.attn_blocks if hasattr(b, 'confidence')]
        if not confidences:
            return 1.0
        return sum(confidences) / len(confidences)

    @property
    def block_confidences(self) -> List[float]:
        """
        Return per-block confidence scores as a list.

        Returns:
            List of float confidence values, one per block.
            Returns 1.0 for every block when ternary_weights=False.
        """
        if not self.ternary_weights:
            return [1.0] * self.N_BLOCKS
        return [b.confidence for b in self.attn_blocks if hasattr(b, 'confidence')]

    # ------------------------------------------------------------------
    # Decision trace (ternary_weights only)
    # ------------------------------------------------------------------

    def trace(self) -> List[List[str]]:
        """
        Return the full decision trace through all blocks.

        When ``ternary_weights=False``, returns a placeholder trace
        indicating standard (non-ternary) execution.

        Returns:
            List of per-block trace strings. Each inner list contains
            human-readable strings detailing attention-head zones,
            MLP zone, and overall block confidence.
        """
        full_trace: List[List[str]] = []
        for i, block in enumerate(self.attn_blocks):
            if hasattr(block, 'trace') and callable(block.trace):
                block_trace = block.trace     # calls TernaryBlock.trace @property
                full_trace.append([f"block_{i}: {t}" for t in block_trace])
            elif hasattr(block, 'trace') and isinstance(block.trace, list):
                full_trace.append([f"block_{i}: {t}" for t in block.trace])
            else:
                full_trace.append([f"block_{i}: standard (no ternary trace)"])
        return full_trace

    def confidence_distribution(self) -> Dict[str, int]:
        """
        Return the global zone distribution across all blocks.

        Returns:
            dict with keys 'GREEN', 'YELLOW', 'RED' counting the number
            of blocks in each confidence zone. When ternary_weights=False,
            all blocks are reported as GREEN.
        """
        zones: Dict[str, int] = {'GREEN': 0, 'YELLOW': 0, 'RED': 0}
        if not self.ternary_weights:
            zones['GREEN'] = self.N_BLOCKS
            return zones
        for block in self.attn_blocks:
            if hasattr(block, 'zone'):
                z = block.zone
                zones[z.name] = zones.get(z.name, 0) + 1
            else:
                zones['GREEN'] += 1
        return zones

    def head_zone_distribution(self) -> Dict[str, int]:
        """
        Return the zone distribution across ALL attention heads (not blocks).

        Returns:
            dict with keys 'GREEN', 'YELLOW', 'RED' counting individual
            attention heads. When ternary_weights=False, all heads are GREEN.
        """
        zones: Dict[str, int] = {'GREEN': 0, 'YELLOW': 0, 'RED': 0}
        if not self.ternary_weights:
            # No TernaryMultiHeadAttention → heads are standard, always GREEN
            zones['GREEN'] = self.N_BLOCKS * (self.lm_head.in_features // 8)  # rough head count
            return zones
        for block in self.attn_blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'confidence_distribution'):
                dz = block.attn.confidence_distribution()
                for k, v in dz.items():
                    zones[k] = zones.get(k, 0) + v
        return zones

    # ------------------------------------------------------------------
    # Legacy / convenience
    # ------------------------------------------------------------------

    def forward_embedding(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass focusing on the embedding and attention blocks.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            tuple: Output after attention blocks and the residual.
        """
        x = self._pre_attn_pass(idx)
        residual = x
        for block in self.attn_blocks:
            x, residual = block.forward_embedding(x)
        return x, residual

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens given a starting sequence.

        Args:
            idx (torch.Tensor): Initial sequence of token indices.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            torch.Tensor: The extended sequence of tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class TernaryTransformer(Transformer):
    """
    Ternary-Weighted Confidence-Cascading Transformer — SuperInstance refactor.

    A convenience subclass of Transformer that always uses ternary weights.
    Per-block confidence scoring, full decision tracing, and zone distribution
    are inherited from the base class.

    Args:
        n_head (int): Number of attention heads.
        n_embed (int): Embedding dimension.
        context_length (int): Maximum sequence length.
        vocab_size (int): Size of the vocabulary.
        N_BLOCKS (int): Number of transformer blocks.
    """
    def __init__(
        self,
        n_head: int,
        n_embed: int,
        context_length: int,
        vocab_size: int,
        N_BLOCKS: int,
    ) -> None:
        super().__init__(
            n_head=n_head,
            n_embed=n_embed,
            context_length=context_length,
            vocab_size=vocab_size,
            N_BLOCKS=N_BLOCKS,
            ternary_weights=True,
        )


if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    vocab_size = 100
    embedding_dim = 32
    num_heads = 4
    num_blocks = 2
    context_len = 5
    input_indices = torch.randint(0, vocab_size, (batch_size, sequence_length))

    transformer_model = Transformer(n_head=num_heads, n_embed=embedding_dim, context_length=context_len, vocab_size=vocab_size, N_BLOCKS=num_blocks)
    logits, loss = transformer_model(input_indices, targets=input_indices) # Using input as target for simplicity

    print("Transformer Logits Shape:", logits.shape)
    print("Transformer Loss:", loss)

    # Example of generating tokens
    start_indices = input_indices[:, :1]  # Take the first token of each sequence as start
    generated_tokens = transformer_model.generate(start_indices, max_new_tokens=5)
    print("Generated Tokens Shape:", generated_tokens.shape)

    # Test ternary variant with confidence / trace
    ternary_model = TernaryTransformer(
        n_head=num_heads, n_embed=embedding_dim,
        context_length=context_len, vocab_size=vocab_size,
        N_BLOCKS=num_blocks,
    )
    t_logits, t_loss = ternary_model(input_indices, targets=input_indices)
    print("\nTernaryTransformer Logits Shape:", t_logits.shape)
    print("TernaryTransformer Loss:", t_loss)
    print("TernaryTransformer Confidence:", ternary_model.confidence)
    print("TernaryTransformer Trace:", ternary_model.trace())
    print("TernaryTransformer Zone Distribution:", ternary_model.confidence_distribution())
    print("TernaryTransformer Head Zone Distribution:", ternary_model.head_zone_distribution())
    print("Conservation loss:", ternary_model.get_conservation_loss())
