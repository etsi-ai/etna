// # Loss functions (CrossEntropy, MSE, etc.)

/// Calculates the cross-entropy loss between predicted probabilities and one-hot encoded labels
// pub fn cross_entropy(preds: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>) -> f32 {
//     let mut total_loss = 0.0;
//     let batch_size = preds.len();

//     for (pred, target) in preds.iter().zip(targets.iter()) {
//         for (p, t) in pred.iter().zip(target.iter()) {
//             if *t == 1.0 {
//                 total_loss -= (p + 1e-9).ln();  // Add epsilon to avoid log(0)
//             }
//         }
//     }

//     total_loss / batch_size as f32
// }


pub fn cross_entropy(preds: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>) -> f32 {
    preds.iter().zip(y.iter())
        .map(|(p_row, y_row)| y_row.iter().zip(p_row.iter()).map(|(y_val, p_val)| -y_val * (p_val + 1e-9).ln()).sum::<f32>())
        .sum::<f32>() / preds.len() as f32
}
