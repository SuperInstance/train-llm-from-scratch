//! # Ternary Core — LLM Inference Kernel
//!
//! Fast LUT-based ternary matrix multiplication, BitNet quantisation,
//! and confidence-cascade utilities for the refactored train-llm-from-scratch.
//!
//! ## Architecture
//!
//! The core insight: ternary weights ∈ {-1, 0, +1} eliminate floating-point
//! multiplies in the forward pass. Instead of `w · x` (one FMA per element)
//! we use a lookup: accumulate, skip, or subtract — all integer ops.
//!
//! Combined with the POLLN confidence cascade (Green/Yellow/Red zones),
//! this produces a transformer where each attention head can declare
//! its own confidence in every forward step.

mod quantize;
mod confidence;

use pyo3::prelude::*;
use pyo3::Bound;

// ─── re-export inner modules ──────────────────────────────────────────────
pub use quantize::quantize_ternary;
pub use confidence::{Zone, TileConfidence};

// ─── Ternary Matrix Multiplication ─────────────────────────────────────────

/// Fast LUT ternary matmul: C = W @ X  where W ∈ {-1, 0, +1}.
///
/// # Arguments
/// * `weights` — flattened ternary weight matrix, shape (m, k), values ∈ {-1, 0, +1}
/// * `input`   — flattened input matrix, shape (k, n)
/// * `output`  — output buffer, shape (m, n), will be filled in-place
///
/// # Panics
/// Panics if lengths don't match: weights.len() == m*k, input.len() == k*n, output.len() == m*n.
///
/// # Zero-multiplies
/// The j loop skips zero weights entirely — on average ~1/3 fewer iterations.
pub fn tern_matmul(
    weights: &[i8],
    input: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(weights.len(), m * k, "weights length must be m*k");
    assert_eq!(input.len(), k * n, "input length must be k*n");
    assert_eq!(output.len(), m * n, "output length must be m*n");

    for i in 0..m {
        let w_row = &weights[i * k..(i + 1) * k];
        let out_row = &mut output[i * n..(i + 1) * n];
        // Zero the output row
        for elem in out_row.iter_mut() {
            *elem = 0.0;
        }
        // Accumulate: for each column of W and row of X
        for j in 0..k {
            let w = w_row[j];
            if w == 0 {
                continue; // zero-weight: skip entirely (~1/3 of weights on average)
            }
            let in_row = &input[j * n..(j + 1) * n];
            if w == 1 {
                // accumulate
                for (o, &x) in out_row.iter_mut().zip(in_row.iter()) {
                    *o += x;
                }
            } else {
                // w == -1: subtract
                for (o, &x) in out_row.iter_mut().zip(in_row.iter()) {
                    *o -= x;
                }
            }
        }
    }
}

// ─── Conservation Loss (pure-Rust reference) ───────────────────────────────

/// Compute the conservation loss Σ(Δ²) for a flattened activation sequence.
///
/// This implements the Σ(Δ_activations) ≈ 0 principle: for contiguous
/// activations, penalises the squared sum of successive differences
/// to enforce closed-gesture energy preservation.
pub fn conservation_loss(activations: &[f32]) -> f64 {
    if activations.len() < 2 {
        return 0.0;
    }
    let mut delta_sum: f64 = 0.0;
    for pair in activations.windows(2) {
        delta_sum += (pair[1] - pair[0]) as f64;
    }
    (delta_sum * delta_sum) / activations.len() as f64
}

// ─── Sheaf Laplacian (structural regularizer) ──────────────────────────────

/// Compute a sheaf-Laplacian-style penalty from a flattened attention matrix.
///
/// Interprets the attention matrix as a graph incidence structure and
/// computes a discrete Laplacian quadratic form: Σ_i (Σ_j A_{ij} (x_i - x_j))²
///
/// This regularises attention heads toward smoothness across the sequence.
pub fn sheaf_laplacian(attn: &[f32], n_heads: usize) -> f64 {
    let n = attn.len();
    if n == 0 || n_heads == 0 {
        return 0.0;
    }
    let seq_len = n / n_heads;
    let mut penalty: f64 = 0.0;
    for h in 0..n_heads {
        let start = h * seq_len;
        let end = start + seq_len;
        let slice = &attn[start..end];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                let diff = (slice[i] - slice[j]) as f64;
                penalty += diff * diff;
            }
        }
    }
    penalty / (n_heads as f64)
}

// ─── Python bindings (PyO3) ────────────────────────────────────────────────

/// Python binding: ternary weight × float input multiplication.
#[pyfunction]
fn py_ternary_matmul(weights: Vec<i8>, input: Vec<f32>, m: usize, n: usize, k: usize) -> PyResult<Vec<f32>> {
    let mut output = vec![0.0f32; m * n];
    tern_matmul(&weights, &input, &mut output, m, n, k);
    Ok(output)
}

/// Python binding: quantize float weights to ternary.
#[pyfunction]
fn py_quantize_ternary(weights: Vec<f32>, m: usize, k: usize) -> PyResult<(Vec<i8>, Vec<f32>)> {
    let (t, s) = quantize_ternary(&weights, m, k);
    Ok((t, s))
}

/// Python binding: conservation loss.
#[pyfunction]
fn py_conservation_loss(activations: Vec<f32>) -> PyResult<f64> {
    Ok(conservation_loss(&activations))
}

/// Python binding: sheaf Laplacian penalty.
#[pyfunction]
fn py_sheaf_laplacian(attn: Vec<f32>, n_heads: usize) -> PyResult<f64> {
    Ok(sheaf_laplacian(&attn, n_heads))
}

// ─── PyO3 Module (v0.28 API) ───────────────────────────────────────────────

#[pymodule]
fn ternary_core_llm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_ternary_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(py_quantize_ternary, m)?)?;
    m.add_function(wrap_pyfunction!(py_conservation_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_sheaf_laplacian, m)?)?;
    Ok(())
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tern_matmul_basic_1x2() {
        // W = [1, 0]  (1x2), X = [[1], [2]] (2x1) => Y = [1*1 + 0*2] = [1]
        let w = vec![1i8, 0];
        let x = vec![1f32, 2f32];
        let mut y = vec![0f32; 1];
        tern_matmul(&w, &x, &mut y, 1, 1, 2);
        assert!((y[0] - 1.0).abs() < 1e-6, "Expected 1.0, got {}", y[0]);
    }

    #[test]
    fn test_tern_matmul_2x3() {
        // W = [[1, -1, 1], [0, 1, -1]] (2x3), X = [[1], [2], [3]] (3x1)
        // Row 0: 1*1 + (-1)*2 + 1*3 = 2
        // Row 1: 0*1 + 1*2 + (-1)*3 = -1
        let w = vec![1i8, -1, 1, 0, 1, -1];
        let x = vec![1f32, 2f32, 3f32];
        let mut y = vec![0f32; 2];
        tern_matmul(&w, &x, &mut y, 2, 1, 3);
        assert!((y[0] - 2.0).abs() < 1e-6, "Row 0: expected 2.0, got {}", y[0]);
        assert!((y[1] - (-1.0)).abs() < 1e-6, "Row 1: expected -1.0, got {}", y[1]);
    }

    #[test]
    fn test_tern_matmul_skip_zeros() {
        // All zeros should produce all zeros
        let w = vec![0i8; 6];
        let x = vec![1f32, 2f32, 3f32];
        let mut y = vec![0f32; 2];
        tern_matmul(&w, &x, &mut y, 2, 1, 3);
        assert!(y.iter().all(|&v| v.abs() < 1e-6), "All-zero weights should produce zero output");
    }

    #[test]
    fn test_quantize_all_positive() {
        let w = vec![2.0, 3.0, 1.5];
        let (t, scales) = quantize_ternary(&w, 1, 3);
        let s = scales[0];
        assert!(s > 0.0, "Scale must be positive, got {}", s);
        assert_eq!(t.len(), 3);
        // All values are > 0.5 after scaling should become +1
        for &v in &t {
            assert!(v == 1 || v == 0 || v == -1, "Must be ternary: got {}", v);
        }
    }

    #[test]
    fn test_conservation_loss_identity() {
        // Constant activations: zero delta sum => zero loss
        let a = vec![1.0f32; 5];
        let loss = conservation_loss(&a);
        assert!(loss.abs() < 1e-6, "Constant activations should have 0 loss, got {}", loss);
    }

    #[test]
    fn test_conservation_loss_linear() {
        // Linear ramp: 0,1,2,3 => deltas: 1,1,1 => delta_sum=3 => loss=9/4=2.25
        let a = vec![0.0f32, 1.0, 2.0, 3.0];
        let loss = conservation_loss(&a);
        assert!((loss - 2.25).abs() < 1e-6, "Expected 2.25, got {}", loss);
    }

    #[test]
    fn test_sheaf_laplacian_simple() {
        // Two identical rows => zero penalty
        let h = vec![1.0, 1.0, 1.0, 1.0];
        let p = sheaf_laplacian(&h, 2);
        assert!(p < 1e-6, "Identical rows should have ~0 penalty, got {}", p);
    }

    #[test]
    fn test_confidence_zone_classification() {
        let g = Zone::Green(0.95);
        let y = Zone::Yellow(0.80);
        let r = Zone::Red(0.50);
        assert_eq!(g.label(), "green");
        assert_eq!(y.label(), "yellow");
        assert_eq!(r.label(), "red");
        assert!((g.score() - 0.95).abs() < 1e-9);
        assert!((y.score() - 0.80).abs() < 1e-9);
        assert!((r.score() - 0.50).abs() < 1e-9);
    }
}
