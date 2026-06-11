/// BitNet-style ternary quantizer.
///
/// Quantizes float weight matrices to ternary values {-1, 0, +1}
/// using row-wise scaling: scale = mean(|w_i|), t_i = clamp(round(w_i / scale), -1, +1)
///
/// Also provides a straight-through estimator gradient placeholder.

/// Quantize a flattened 2D weight matrix (m rows, k cols) to ternary.
///
/// Returns (ternary_weights, scales) where:
/// - ternary_weights: Vec<i8> of shape (m, k) with values in {-1, 0, +1}
/// - scales: Vec<f32> of length m containing per-row scale factors
pub fn quantize_ternary(weights: &[f32], m: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(weights.len(), m * k, "weights length must be m*k");

    let mut ternary = vec![0i8; m * k];
    let mut scales = Vec::with_capacity(m);

    for row in 0..m {
        let start = row * k;
        let end = start + k;
        let row_slice = &weights[start..end];

        // scale = mean(|w_i|)
        let abs_sum: f32 = row_slice.iter().map(|v| v.abs()).sum();
        let scale = if k > 0 { abs_sum / k as f32 } else { 1.0f32 };
        let scale = if scale == 0.0 { 1.0f32 } else { scale };
        scales.push(scale);

        // t_i = clamp(round(w_i / scale), -1, +1)
        for (j, &w) in row_slice.iter().enumerate() {
            let q = (w / scale).round();
            let t = if q > 0.5 {
                1i8
            } else if q < -0.5 {
                -1i8
            } else {
                0i8
            };
            ternary[start + j] = t;
        }
    }

    (ternary, scales)
}

/// Straight-through estimator (STE) backward — surrogate gradient.
///
/// In a real training loop, this would route gradients around the quantizer.
/// For now, we return the "fake gradient" region: values whose weight
/// fell into the [-0.5, 0.5] dead zone get zero gradient, others get 1.0.
///
/// Returns a mask of the same shape as weights where:
/// - 1.0 means gradient passes through
/// - 0.0 means gradient is clipped (dead zone)
pub fn ste_gradient_mask(weights: &[f32], scales: &[f32], m: usize, k: usize) -> Vec<f32> {
    assert_eq!(weights.len(), m * k);
    assert_eq!(scales.len(), m);

    let mut mask = vec![0.0f32; m * k];

    for row in 0..m {
        let scale = scales[row];
        let start = row * k;
        for j in 0..k {
            let q = weights[start + j] / scale;
            // STE: if |q| < 0.5, the weight falls in the dead zone → no gradient
            mask[start + j] = if q.abs() < 0.5 { 0.0 } else { 1.0 };
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_all_positive() {
        let w = vec![3.0, 1.0, 0.5, 2.0];
        let (t, s) = quantize_ternary(&w, 1, 4);
        // scale = (3+1+0.5+2)/4 = 1.625
        // 3/1.625 ≈ 1.85 → round to 2 → clamp +1
        // 1/1.625 ≈ 0.615 → round to 1 → +1
        // 0.5/1.625 ≈ 0.308 → round to 0 → 0
        // 2/1.625 ≈ 1.23 → round to 1 → +1
        assert_eq!(t, vec![1, 1, 0, 1]);
        assert!((s[0] - 1.625).abs() < 1e-5);
    }

    #[test]
    fn test_quantize_all_negative() {
        let w = vec![-3.0, -1.0, -0.5, -2.0];
        let (t, _) = quantize_ternary(&w, 1, 4);
        assert_eq!(t, vec![-1, -1, 0, -1]);
    }

    #[test]
    fn test_quantize_ternary_vs_binary() {
        // All zeros should produce all zeros
        let w = vec![0.0; 6];
        let (t, s) = quantize_ternary(&w, 2, 3);
        assert_eq!(t, vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(s, vec![1.0, 1.0]); // scale defaults to 1.0 when all zeros
    }

    #[test]
    fn test_quantize_output_in_set() {
        let w = vec![0.1, -2.5, 3.7, -0.2, 0.8, -12.0];
        let (t, _) = quantize_ternary(&w, 2, 3);
        for &v in &t {
            assert!(v == -1 || v == 0 || v == 1, "value {} not in {{-1,0,1}}", v);
        }
    }

    #[test]
    fn test_ste_mask_dead_zone() {
        let w = vec![3.0, 0.1, -2.0, 0.0];
        let (_, s) = quantize_ternary(&w, 1, 4);
        let mask = ste_gradient_mask(&w, &s, 1, 4);
        // scale = (3+0.1+2+0)/4 = 1.275
        // 3/1.275 ≈ 2.35 → abs >= 0.5 → 1.0
        // 0.1/1.275 ≈ 0.078 → abs < 0.5 → 0.0
        // -2/1.275 ≈ -1.57 → abs >= 0.5 → 1.0
        // 0/1.275 = 0 → abs < 0.5 → 0.0
        assert_eq!(mask, vec![1.0, 0.0, 1.0, 0.0]);
    }
}
