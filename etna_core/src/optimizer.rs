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
