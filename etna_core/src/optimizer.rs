// # Optimizers (SGD, Adam, etc.)

/// Simple Stochastic Gradient Descent (SGD) optimizer

pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }

    // Update weights and biases in-place
    pub fn step(
        &self,
        weights: &mut Vec<Vec<f32>>,
        weight_grads: &Vec<Vec<f32>>,
        bias: &mut Vec<f32>,
        bias_grads: &Vec<f32>,
    ) {
        for i in 0..weights.len() {
            for j in 0..weights[0].len() {
                weights[i][j] -= self.learning_rate * weight_grads[i][j];
            }
            bias[i] -= self.learning_rate * bias_grads[i];
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_updates_weight_correctly() {
        let optimizer = SGD::new(0.1);

        let mut weights = vec![vec![1.0]];
        let weight_grads = vec![vec![0.1]];

        let mut bias = vec![0.0];
        let bias_grads = vec![0.0];

        optimizer.step(
            &mut weights,
            &weight_grads,
            &mut bias,
            &bias_grads,
        );

        // Floating point math is slightly dishonest, so use tolerance
        let expected = 0.99;
        let actual = weights[0][0];

        assert!(
            (actual - expected).abs() < 1e-6,
            "Expected weight to be {}, got {}",
            expected,
            actual
        );
    }
}
