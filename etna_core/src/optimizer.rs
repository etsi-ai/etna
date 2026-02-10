//! Optimizers (Sgd, Adam)
//!
//! Simple Stochastic Gradient Descent (Sgd) optimizer with optional L2 regularization (weight decay)
//!
//! L2 regularization adds a penalty term to the loss function: L_reg = L + (lambda/2) * ||W||^2
//! The gradient becomes: grad_W = grad_L + lambda * W
//! This encourages smaller weights and helps prevent overfitting.

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Sgd {
    pub learning_rate: f32,
    pub weight_decay: f32,  // L2 regularization coefficient (lambda)
}

impl Sgd {
    pub fn new(learning_rate: f32) -> Self {
        Sgd {
            learning_rate,
            weight_decay: 0.0,  // Default: no regularization
        }
    }

    /// Create Sgd optimizer with L2 regularization (weight decay)
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> Self {
        Sgd {
            learning_rate,
            weight_decay,
        }
    }
}

/// Adam optimizer
#[derive(Serialize, Deserialize)]
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,

    // First and second moment estimates
    m_w: Vec<Vec<f32>>,
    v_w: Vec<Vec<f32>>,
    m_b: Vec<f32>,
    v_b: Vec<f32>,
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            t: 0,
            m_w: Vec::new(),
            v_w: Vec::new(),
            m_b: Vec::new(),
            v_b: Vec::new(),
        }
    }

    pub fn step(
        &mut self,
        weights: &mut [Vec<f32>],
        grad_w: &[Vec<f32>],
        bias: &mut [f32],
        grad_b: &[f32],
    ) {
        self.t += 1;

        // Lazy initialization (once per layer)
        if self.m_w.is_empty() {
            self.m_w = vec![vec![0.0; weights[0].len()]; weights.len()];
            self.v_w = vec![vec![0.0; weights[0].len()]; weights.len()];
            self.m_b = vec![0.0; bias.len()];
            self.v_b = vec![0.0; bias.len()];
        }

        let t_f = self.t as f32;

        // Update weights
        for i in 0..weights.len() {
            for j in 0..weights[0].len() {
                self.m_w[i][j] =
                    self.beta1 * self.m_w[i][j] + (1.0 - self.beta1) * grad_w[i][j];
                self.v_w[i][j] =
                    self.beta2 * self.v_w[i][j] + (1.0 - self.beta2) * grad_w[i][j].powi(2);

                let m_hat = self.m_w[i][j] / (1.0 - self.beta1.powf(t_f));
                let v_hat = self.v_w[i][j] / (1.0 - self.beta2.powf(t_f));

                weights[i][j] -=
                    self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }

        // Update bias
        for i in 0..bias.len() {
            self.m_b[i] =
                self.beta1 * self.m_b[i] + (1.0 - self.beta1) * grad_b[i];
            self.v_b[i] =
                self.beta2 * self.v_b[i] + (1.0 - self.beta2) * grad_b[i].powi(2);

            let m_hat = self.m_b[i] / (1.0 - self.beta1.powf(t_f));
            let v_hat = self.v_b[i] / (1.0 - self.beta2.powf(t_f));

            bias[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_with_weight_decay_creates_correctly() {
        let optimizer = Sgd::with_weight_decay(0.01, 0.001);
        assert!((optimizer.learning_rate - 0.01).abs() < 1e-6);
        assert!((optimizer.weight_decay - 0.001).abs() < 1e-6);
    }

    #[test]
    fn sgd_default_has_no_weight_decay() {
        let optimizer = Sgd::new(0.1);
        assert!((optimizer.weight_decay - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_adam_step_updates() {
        // Initialize Adam with specific constants for predictable calculation
        // lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

        let mut weights = vec![vec![0.5]]; // Single weight
        let grad_w = vec![vec![0.1]];      // Constant gradient
        let mut bias = vec![0.1];          // Single bias
        let grad_b = vec![0.1];            // Constant gradient

        // Perform one step
        optimizer.step(&mut weights, &grad_w, &mut bias, &grad_b);

        // 1. Verify time step increment
        assert_eq!(optimizer.t, 1, "Time step should be 1 after first update");

        // 2. Verify numerical update (Manual Calculation)
        // t = 1
        // m = 0.9*0 + 0.1*0.1 = 0.01
        // v = 0.999*0 + 0.001*(0.1^2) = 0.00001
        // m_hat = 0.01 / (1 - 0.9) = 0.1
        // v_hat = 0.00001 / (1 - 0.999) = 0.01
        // step = lr * m_hat / (sqrt(v_hat) + eps)
        // step = 0.001 * 0.1 / (0.1 + 1e-8) ≈ 0.001

        // Expected Weight: 0.5 - 0.001 = 0.499
        let expected_weight = 0.499;
        assert!((weights[0][0] - expected_weight).abs() < 1e-5,
            "Weight update incorrect. Got {}, expected approx {}", weights[0][0], expected_weight);

        // Expected Bias: 0.1 - 0.001 = 0.099
        let expected_bias = 0.099;
        assert!((bias[0] - expected_bias).abs() < 1e-5,
            "Bias update incorrect. Got {}, expected approx {}", bias[0], expected_bias);
    }
}