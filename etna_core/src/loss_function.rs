// # Loss functions (CrossEntropy, MSE)

/// Calculates the cross-entropy loss (for Classification)
pub fn cross_entropy(preds: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>) -> f32 {
    preds.iter().zip(y.iter())
        .map(|(p_row, y_row)| {
            y_row.iter().zip(p_row.iter())
                .map(|(y_val, p_val)| -y_val * (p_val + 1e-9).ln())
                .sum::<f32>()
        })
        .sum::<f32>() / preds.len() as f32
}

/// Calculates Mean Squared Error (for Regression)
pub fn mse(preds: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>) -> f32 {
    let mut loss = 0.0;
    let n = preds.len() as f32;
    
    for (p_row, y_row) in preds.iter().zip(y.iter()) {
        for (p, t) in p_row.iter().zip(y_row.iter()) {
            loss += (p - t).powi(2);
        }
    }
    loss / n
}
