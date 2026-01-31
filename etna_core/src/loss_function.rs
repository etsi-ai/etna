// # Loss functions (CrossEntropy, MSE)

/// Calculates the cross-entropy loss (for Classification)
pub fn cross_entropy(preds: &[Vec<f32>], y: &[Vec<f32>]) -> f32 {
    preds.iter().zip(y.iter())
        .map(|(p_row, y_row)| {
            y_row.iter().zip(p_row.iter())
                .map(|(y_val, p_val)| -y_val * (p_val + f32::EPSILON).ln())
                .sum::<f32>()
        })
        .sum::<f32>() / preds.len() as f32
}

/// Calculates Mean Squared Error (for Regression)
pub fn mse(preds: &[Vec<f32>], y: &[Vec<f32>]) -> f32 {
    let mut loss = 0.0;
    let n = preds.len() as f32;

    for (p_row, y_row) in preds.iter().zip(y.iter()) {
        for (p, t) in p_row.iter().zip(y_row.iter()) {
            loss += (p - t).powi(2);
        }
    }
    loss / n
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_loss_basic() {
        // preds = [[2.0]], target = [[1.0]]
        // (2 - 1)^2 = 1
        let preds = vec![vec![2.0]];
        let targets = vec![vec![1.0]];

        let loss = mse(&preds, &targets);

        assert_eq!(loss, 1.0);
    }

    #[test]
    fn mse_loss_multiple_values() {
        // ((2-1)^2 + (4-3)^2) / 1 = 2
        let preds = vec![vec![2.0, 4.0]];
        let targets = vec![vec![1.0, 3.0]];

        let loss = mse(&preds, &targets);

        assert_eq!(loss, 2.0);
    }

    #[test]
    fn cross_entropy_basic() {
        // One-hot target: class 0
        // -ln(0.8)
        let preds = vec![vec![0.8, 0.2]];
        let targets = vec![vec![1.0, 0.0]];

        let loss = cross_entropy(&preds, &targets);
        let expected = -0.8_f32.ln();

        assert!(
            (loss - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            loss
        );
    }
}
