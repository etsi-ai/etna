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

/// Adam optimizer with adaptive learning rates
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    // First moment estimates (m) for weights and biases
    m_weights: Vec<Vec<f32>>,
    m_bias: Vec<f32>,
    // Second moment estimates (v) for weights and biases
    v_weights: Vec<Vec<f32>>,
    v_bias: Vec<f32>,
    // Time step counter
    t: usize,
}

impl Adam {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m_weights: vec![],
            m_bias: vec![],
            v_weights: vec![],
            v_bias: vec![],
            t: 0,
        }
    }

    /// Initialize moment estimates based on weight/bias dimensions
    pub fn initialize(&mut self, weight_shape: (usize, usize), bias_len: usize) {
        self.m_weights = vec![vec![0.0; weight_shape.1]; weight_shape.0];
        self.m_bias = vec![0.0; bias_len];
        self.v_weights = vec![vec![0.0; weight_shape.1]; weight_shape.0];
        self.v_bias = vec![0.0; bias_len];
        self.t = 0;
    }

    /// Update weights and biases using Adam algorithm
    pub fn step(
        &mut self,
        weights: &mut Vec<Vec<f32>>,
        weight_grads: &Vec<Vec<f32>>,
        bias: &mut Vec<f32>,
        bias_grads: &Vec<f32>,
    ) {
        // Initialize on first step if not already initialized
        if self.m_weights.is_empty() {
            let input_size = if !weights.is_empty() && !weights[0].is_empty() {
                weights[0].len()
            } else {
                return; // Can't initialize with empty weights
            };
            self.initialize((weights.len(), input_size), bias.len());
        }

        self.t += 1;
        let t_f32 = self.t as f32;

        // Update weights - ensure dimensions match
        let output_size = weights.len().min(weight_grads.len()).min(self.m_weights.len());
        for i in 0..output_size {
            let input_size = weights[i].len().min(weight_grads[i].len()).min(self.m_weights[i].len());
            for j in 0..input_size {
                // Update biased first moment estimate
                self.m_weights[i][j] = self.beta1 * self.m_weights[i][j] + (1.0 - self.beta1) * weight_grads[i][j];
                
                // Update biased second raw moment estimate
                self.v_weights[i][j] = self.beta2 * self.v_weights[i][j] + (1.0 - self.beta2) * weight_grads[i][j] * weight_grads[i][j];
                
                // Compute bias-corrected first moment estimate
                let m_hat = self.m_weights[i][j] / (1.0 - self.beta1.powf(t_f32));
                
                // Compute bias-corrected second raw moment estimate
                let v_hat = self.v_weights[i][j] / (1.0 - self.beta2.powf(t_f32));
                
                // Update weights
                weights[i][j] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }

        // Update biases
        for i in 0..bias.len() {
            // Update biased first moment estimate
            self.m_bias[i] = self.beta1 * self.m_bias[i] + (1.0 - self.beta1) * bias_grads[i];
            
            // Update biased second raw moment estimate
            self.v_bias[i] = self.beta2 * self.v_bias[i] + (1.0 - self.beta2) * bias_grads[i] * bias_grads[i];
            
            // Compute bias-corrected first moment estimate
            let m_hat = self.m_bias[i] / (1.0 - self.beta1.powf(t_f32));
            
            // Compute bias-corrected second raw moment estimate
            let v_hat = self.v_bias[i] / (1.0 - self.beta2.powf(t_f32));
            
            // Update bias
            bias[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

/// Optimizer enum to support both SGD and Adam
#[derive(Clone, Copy)]
pub enum OptimizerType {
    SGD,
    Adam,
}
