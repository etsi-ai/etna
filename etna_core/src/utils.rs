// # Utility functions (Matrix ops, initialization)


/// Converts class indices (e.g., 2, 0, 1) into one-hot vectors.
pub fn one_hot_encode(labels: &Vec<usize>, num_classes: usize) -> Vec<Vec<f32>> {
    labels
        .iter()
        .map(|&label| {
            let mut row = vec![0.0; num_classes];
            row[label] = 1.0;
            row
        })
        .collect()
}

/// Transposes a matrix (Vec<Vec<f32>>)
pub fn transpose(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    if matrix.is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    (0..cols)
        .map(|j| (0..rows).map(|i| matrix[i][j]).collect())
        .collect()
}

/// Returns vector of predicted class indices from probabilities
pub fn argmax_rows(matrix: &Vec<Vec<f32>>) -> Vec<usize> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

/// Calculates accuracy between predicted and true labels
pub fn accuracy(preds: &Vec<usize>, labels: &Vec<usize>) -> f32 {
    let correct = preds
        .iter()
        .zip(labels)
        .filter(|(p, l)| p == l)
        .count();

    correct as f32 / preds.len() as f32
}
