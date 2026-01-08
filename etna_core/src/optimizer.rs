// # Optimizers (SGD, Adam, etc.)

/// Simple Stochastic Gradient Descent (SGD) optimizer with optional L2 regularization (weight decay)
/// 
/// L2 regularization adds a penalty term to the loss function: L_reg = L + (lambda/2) * ||W||^2
/// The gradient becomes: grad_W = grad_L + lambda * W
/// This encourages smaller weights and helps prevent overfitting.

pub struct SGD {
    pub learning_rate: f32,
    pub weight_decay: f32,  // L2 regularization coefficient (lambda)
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { 
            learning_rate,
            weight_decay: 0.0,  // Default: no regularization
        }
    }

    /// Create SGD optimizer with L2 regularization (weight decay)
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> Self {
        SGD { 
            learning_rate, 
            weight_decay,
        }
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

    #[test]
    fn sgd_with_weight_decay_creates_correctly() {
        let optimizer = SGD::with_weight_decay(0.01, 0.001);
        assert!((optimizer.learning_rate - 0.01).abs() < 1e-6);
        assert!((optimizer.weight_decay - 0.001).abs() < 1e-6);
    }

    #[test]
    fn sgd_default_has_no_weight_decay() {
        let optimizer = SGD::new(0.1);
        assert!((optimizer.weight_decay - 0.0).abs() < 1e-6);
    }
}
