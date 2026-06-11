"""
Confidence Cascade Module — SuperInstance scientific refactor.

Defines the Zone enum, ConfidenceTile interface (POLLN-style tile),
and composition operators for building confidence-cascading architectures.
"""

from enum import Enum, auto
from typing import Callable, List, Any, Optional


class Zone(Enum):
    """Confidence zones based on score thresholds."""
    GREEN = auto()   # ≥0.90 — high confidence, fully trusted
    YELLOW = auto()  # ≥0.75 — medium confidence, requires monitoring
    RED = auto()     # <0.75  — low confidence, degraded/uncertain


def get_zone(confidence: float) -> Zone:
    """Return the Zone corresponding to a numeric confidence score."""
    if confidence >= 0.90:
        return Zone.GREEN
    elif confidence >= 0.75:
        return Zone.YELLOW
    else:
        return Zone.RED


class ConfidenceTile:
    """
    A POLLN-style tile: (I, O, f, c, τ)

    Each tile carries:
      - input_type  (I):  a string describing the input domain
      - output_type (O):  a string describing the output domain
      - function    (f):  the callable transformation (Any -> Any)
      - confidence  (c):  a callable that returns a float ∈ [0, 1]
      - trace       (τ):  a list of human-readable decision steps

    Tiles can be composed sequentially (compose) or in parallel (parallel)
    to build complex architectures from simple building blocks.
    """

    def __init__(
        self,
        input_type: str,
        output_type: str,
        function: Callable,
        confidence: Optional[Callable[[Any], float]] = None,
        trace: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize a ConfidenceTile.

        Args:
            input_type:  Human-readable description of the input domain.
            output_type: Human-readable description of the output domain.
            function:    The core transformation (Any -> Any).
            confidence:  Optional callable that takes the output and returns
                         a float ∈ [0, 1]. Defaults to always returning 1.0.
            trace:       Optional list of decision strings. Defaults to empty.
        """
        self.input_type = input_type
        self.output_type = output_type
        self.function = function
        self._confidence_fn = confidence or (lambda _: 1.0)
        self.trace = trace or []
        self._last_confidence: float = 1.0
        self._last_input: Any = None
        self._last_output: Any = None

    @property
    def confidence(self) -> float:
        """Return the most recently computed confidence score."""
        return self._last_confidence

    @property
    def zone(self) -> Zone:
        """Return the Zone for the most recent confidence score."""
        return get_zone(self._last_confidence)

    def __call__(self, x: Any) -> Any:
        """
        Run the tile: store input, compute output and confidence.

        Args:
            x: Input value.

        Returns:
            Transformed output.
        """
        self._last_input = x
        self._last_output = self.function(x)
        self._last_confidence = self._confidence_fn(self._last_output)
        return self._last_output

    def compose(self, other: 'ConfidenceTile') -> 'ConfidenceTile':
        """
        Sequential composition: self ∘ other.

        Confidence multiplies: c_combined = c_self * c_other.

        Args:
            other: The tile to apply first (inner).

        Returns:
            A new ConfidenceTile representing the composition.
        """
        def composed_fn(x: Any) -> Any:
            intermediate = other(x)
            return self(intermediate)

        def composed_confidence(out: Any) -> float:
            # Use the stored confidences from the last call
            return self._last_confidence * other._last_confidence

        return ConfidenceTile(
            input_type=other.input_type,
            output_type=self.output_type,
            function=composed_fn,
            confidence=composed_confidence,
            trace=self.trace + [' ∘ '] + other.trace,
        )

    def parallel(
        self, others: List['ConfidenceTile'], weights: List[float]
    ) -> 'ConfidenceTile':
        """
        Parallel composition: weighted average.

        The combined output is Σ(w_i * tile_i(x)), and confidence is the
        weighted average of individual confidences.

        Args:
            others:  List of tiles to run in parallel with self.
            weights: Weights for each tile (must sum to 1, same length as
                     1 + len(others) where index 0 is self).

        Returns:
            A new ConfidenceTile representing the parallel composition.
        """
        all_tiles: List[ConfidenceTile] = [self] + list(others)
        assert len(all_tiles) == len(weights), (
            f"Expected {len(all_tiles)} weights, got {len(weights)}."
        )
        w_sum = sum(weights)
        assert abs(w_sum - 1.0) < 1e-6, f"Weights must sum to 1, got {w_sum}."

        def parallel_fn(x: Any) -> Any:
            results = [t(x) for t in all_tiles]
            # Weighted sum for tensor outputs; fall back to averaging
            if hasattr(results[0], 'mul') and hasattr(results[0], 'add'):
                out = sum(w * r for w, r in zip(weights, results))
            else:
                out = sum(w * r for w, r in zip(weights, results))
            return out

        def parallel_confidence(out: Any) -> float:
            return sum(
                w * t._last_confidence
                for w, t in zip(weights, all_tiles)
            )

        combined_trace: List[str] = []
        for i, t in enumerate(all_tiles):
            combined_trace.append(f"branch_{i}(w={weights[i]:.3f}): {t.trace}")
        combined_trace.append("∥ (parallel merge)")

        return ConfidenceTile(
            input_type=self.input_type,
            output_type=self.output_type,
            function=parallel_fn,
            confidence=parallel_confidence,
            trace=combined_trace,
        )

    def __repr__(self) -> str:
        return (
            f"ConfidenceTile({self.input_type} → {self.output_type}, "
            f"conf={self._last_confidence:.4f}, zone={self.zone.name})"
        )
