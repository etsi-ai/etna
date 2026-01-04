// Optimizers (SGD, Adam)

/// Supported optimizer types
#[derive(Clone, Copy)]
pub enum OptimizerType {
    SGD,
    Adam,
}

/// Simple Stochastic Gradient Descent (SGD) optimizer
pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            learning_rate: lr,
        }
    }
}

/// Adam optimizer
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
        weights: &mut Vec<Vec<f32>>,
        grad_w: &Vec<Vec<f32>>,
        bias: &mut Vec<f32>,
        grad_b: &Vec<f32>,
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
