use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::optimizer::{SGD, Adam};

/// Weight initialization strategy
/// - Xavier (Glorot): For layers followed by Sigmoid/Softmax. std = sqrt(2 / (n_in + n_out))
/// - Kaiming (He): For layers followed by ReLU/LeakyReLU. std = sqrt(2 / n_in)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitStrategy {
    /// Xavier/Glorot initialization - best for Sigmoid, Tanh, Softmax
    Xavier,
    /// Kaiming/He initialization - best for ReLU, LeakyReLU
    Kaiming,
}

/// Fully connected layer: y = Wx + b
#[derive(Serialize, Deserialize)]
pub struct Linear {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    grad_weights: Vec<Vec<f32>>,
    grad_bias: Vec<f32>,
    cached_input: Vec<Vec<f32>>,
}

impl Linear {
    /// Create a new Linear layer with legacy initialization (backward compatible)
    /// Create a new Linear layer with specified initialization strategy
    /// 
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `output_size` - Number of output features    
    /// * `init` - Initialization strategy (Xavier, Kaiming, or Legacy)
    pub fn new_with_init(input_size: usize, output_size: usize, init: InitStrategy) -> Self {
        // UPDATED: Use rand::rng() (replacing thread_rng)
        let mut rng = rand::rng();
        
        let weights = match init {
            InitStrategy::Xavier => {
                // Xavier/Glorot: std = sqrt(2 / (n_in + n_out))
                let std = (2.0 / (input_size + output_size) as f32).sqrt();
                (0..output_size)
                    .map(|_| {
                        (0..input_size)
                            .map(|_| Self::sample_normal(&mut rng) * std)
                            .collect()
                    })
                    .collect()
            },
            InitStrategy::Kaiming => {
                // Kaiming/He: std = sqrt(2 / n_in)
                let std = (2.0 / input_size as f32).sqrt();
                (0..output_size)
                    .map(|_| {
                        (0..input_size)
                            .map(|_| Self::sample_normal(&mut rng) * std)
                            .collect()
                    })
                    .collect()
            },
            
        };
            
        // Initialize gradients as 0.0
        Self {
            weights,
            bias: vec![0.0; output_size],
            grad_weights: vec![vec![0.0; input_size]; output_size],
            grad_bias: vec![0.0; output_size],
            cached_input: vec![],
        }
    }

    /// Sample from standard normal distribution using Box-Muller transform
    fn sample_normal<R: Rng>(rng: &mut R) -> f32 {
        // Box-Muller transform for normal distribution
        // UPDATED: Use random_range() (replacing gen_range)
        let u1: f32 = rng.random_range(0.0001..1.0); // Avoid log(0)
        let u2: f32 = rng.random_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn forward(&mut self, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        self.cached_input = input.clone();

        input.iter().map(|x| {
            self.weights.iter().enumerate().map(|(i, w)| {
                w.iter().zip(x.iter()).map(|(w, x)| w * x).sum::<f32>() + self.bias[i]
            }).collect()
        }).collect()
    }

    pub fn backward(&mut self, grad_output: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let batch_size = self.cached_input.len();
        let input_size = self.cached_input[0].len();

        let mut grad_input = vec![vec![0.0; input_size]; batch_size];

        for i in 0..batch_size {
            for j in 0..self.weights.len() {
                self.grad_bias[j] += grad_output[i][j];
                for k in 0..input_size {
                    self.grad_weights[j][k] += grad_output[i][j] * self.cached_input[i][k];
                    grad_input[i][k] += grad_output[i][j] * self.weights[j][k];
                }
            }
        }
        grad_input
    }

    /// Update weights using SGD
    pub fn update_sgd(&mut self, optimizer: &SGD) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[0].len() {
                // L2 regularization: add lambda * weight to gradient
                let l2_term = optimizer.weight_decay * self.weights[i][j];
                self.weights[i][j] -= optimizer.learning_rate * (self.grad_weights[i][j] + l2_term);
                self.grad_weights[i][j] = 0.0;
            }
            // Note: We don't apply weight decay to biases (standard practice)
            self.bias[i] -= optimizer.learning_rate * self.grad_bias[i];
            self.grad_bias[i] = 0.0;
        }
    }

    /// Update weights using Adam
    pub fn update_adam(&mut self, optimizer: &mut Adam) {
        // Use Adam's step method which handles the update internally
        optimizer.step(
            &mut self.weights,
            &self.grad_weights,
            &mut self.bias,
            &self.grad_bias,
        );

        // Clear gradients after update
        self.grad_weights.iter_mut().for_each(|r| r.iter_mut().for_each(|v| *v = 0.0));
        self.grad_bias.iter_mut().for_each(|v| *v = 0.0);
    }
}

/// =======================
/// ReLU
/// =======================
#[derive(Serialize, Deserialize)]
pub struct ReLU;

impl ReLU {
    pub fn forward(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        input.iter().map(|x| x.iter().map(|&v| v.max(0.0)).collect()).collect()
    }

    pub fn backward(grad: &Vec<Vec<f32>>, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        grad.iter().zip(input.iter())
            .map(|(g, i)| g.iter().zip(i.iter()).map(|(g, v)| if *v > 0.0 { *g } else { 0.0 }).collect())
            .collect()
    }
}

/// =======================
/// Leaky ReLU
/// Leaky ReLU activation: max(0.01 * x, x)
/// =======================
#[derive(Serialize, Deserialize)]
pub struct LeakyReLU;

impl LeakyReLU {
    const ALPHA: f32 = 0.01;

    pub fn forward(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        input.iter().map(|x| {
            x.iter().map(|&v| if v > 0.0 { v } else { Self::ALPHA * v }).collect()
        }).collect()
    }

    pub fn backward(grad: &Vec<Vec<f32>>, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        grad.iter().zip(input.iter())
            .map(|(g, i)| g.iter().zip(i.iter())
                .map(|(g, v)| if *v > 0.0 { *g } else { Self::ALPHA * *g })
                .collect())
            .collect()
    }
}

/// =======================
/// Sigmoid
/// =======================
#[derive(Serialize, Deserialize)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn forward(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        input.iter().map(|x| {
            x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
        }).collect()
    }

    pub fn backward(grad: &Vec<Vec<f32>>, out: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        grad.iter().zip(out.iter())
            .map(|(g, o)| g.iter().zip(o.iter()).map(|(g_val, o_val)| g_val * o_val * (1.0 - o_val)).collect())
            .collect()
    }
}

/// =======================
/// Activation Enum
/// =======================
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    LeakyReLU,
    Sigmoid,
}

impl Activation {
    pub fn forward(&self, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        match self {
            Self::ReLU => ReLU::forward(input),
            Self::LeakyReLU => LeakyReLU::forward(input),
            Self::Sigmoid => Sigmoid::forward(input),
        }
    }

    pub fn backward(&self, grad: &Vec<Vec<f32>>, cache: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        match self {
            Self::ReLU => ReLU::backward(grad, cache),
            Self::LeakyReLU => LeakyReLU::backward(grad, cache),
            Self::Sigmoid => Sigmoid::backward(grad, cache),
        }
    }

    /// Get the recommended weight initialization strategy for this activation
    /// - ReLU/LeakyReLU: Kaiming (He) initialization
    /// - Sigmoid: Xavier (Glorot) initialization
    pub fn init_strategy(&self) -> InitStrategy {
        match self {
            Activation::ReLU | Activation::LeakyReLU => InitStrategy::Kaiming,
            Activation::Sigmoid => InitStrategy::Xavier,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaky_relu_forward_positive() {
        let input = vec![vec![1.0, 2.0, 3.0]];
        let output = LeakyReLU::forward(&input);
        assert_eq!(output, vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_leaky_relu_forward_negative() {
        let input = vec![vec![-1.0, -2.0, -3.0]];
        let output = LeakyReLU::forward(&input);
        assert_eq!(output, vec![vec![-0.01, -0.02, -0.03]]);
    }

    #[test]
    fn test_leaky_relu_forward_mixed() {
        let input = vec![vec![-2.0, 0.0, 2.0]];
        let output = LeakyReLU::forward(&input);
        assert_eq!(output, vec![vec![-0.02, 0.0, 2.0]]);
    }

    #[test]
    fn test_leaky_relu_backward_positive() {
        let grad_output = vec![vec![1.0, 1.0, 1.0]];
        let input = vec![vec![1.0, 2.0, 3.0]];
        let grad = LeakyReLU::backward(&grad_output, &input);
        assert_eq!(grad, vec![vec![1.0, 1.0, 1.0]]);
    }

    #[test]
    fn test_leaky_relu_backward_negative() {
        let grad_output = vec![vec![1.0, 1.0, 1.0]];
        let input = vec![vec![-1.0, -2.0, -3.0]];
        let grad = LeakyReLU::backward(&grad_output, &input);
        assert_eq!(grad, vec![vec![0.01, 0.01, 0.01]]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let input = vec![vec![0.0]];
        let output = Sigmoid::forward(&input);
        assert!((output[0][0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_forward_large_positive() {
        let input = vec![vec![100.0]];
        let output = Sigmoid::forward(&input);
        assert!((output[0][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_forward_large_negative() {
        let input = vec![vec![-100.0]];
        let output = Sigmoid::forward(&input);
        assert!(output[0][0].abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_backward() {
        // At x=0, sigmoid = 0.5, derivative = 0.5 * 0.5 = 0.25
        let grad_output = vec![vec![1.0]];
        let sigmoid_output = vec![vec![0.5]];
        let grad = Sigmoid::backward(&grad_output, &sigmoid_output);
        assert!((grad[0][0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_backward_near_saturation() {
        // Near saturation (output close to 1), derivative should be close to 0
        let grad_output = vec![vec![1.0]];
        let sigmoid_output = vec![vec![0.99]];
        let grad = Sigmoid::backward(&grad_output, &sigmoid_output);
        assert!((grad[0][0] - 0.0099).abs() < 1e-4);
    }

    #[test]
    fn test_relu_forward() {
        let input = vec![vec![-1.0, 0.0, 1.0]];
        let output = ReLU::forward(&input);
        assert_eq!(output, vec![vec![0.0, 0.0, 1.0]]);
    }

    #[test]
    fn test_relu_backward() {
        let grad_output = vec![vec![1.0, 1.0, 1.0]];
        let input = vec![vec![-1.0, 0.0, 1.0]];
        let grad = ReLU::backward(&grad_output, &input);
        assert_eq!(grad, vec![vec![0.0, 0.0, 1.0]]);
    }

    #[test]
    fn test_activation_enum_relu_forward() {
        let act = Activation::ReLU;
        let input = vec![vec![-1.0, 0.0, 1.0]];
        let output = act.forward(&input);
        assert_eq!(output, vec![vec![0.0, 0.0, 1.0]]);
    }

    #[test]
    fn test_activation_enum_leaky_relu_forward() {
        let act = Activation::LeakyReLU;
        let input = vec![vec![-1.0, 0.0, 1.0]];
        let output = act.forward(&input);
        assert_eq!(output, vec![vec![-0.01, 0.0, 1.0]]);
    }

    #[test]
    fn test_activation_enum_sigmoid_forward() {
        let act = Activation::Sigmoid;
        let input = vec![vec![0.0]];
        let output = act.forward(&input);
        assert!((output[0][0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_activation_enum_relu_backward() {
        let act = Activation::ReLU;
        let grad_output = vec![vec![1.0, 1.0, 1.0]];
        let input = vec![vec![-1.0, 0.0, 1.0]];
        let grad = act.backward(&grad_output, &input);
        assert_eq!(grad, vec![vec![0.0, 0.0, 1.0]]);
    }

    #[test]
    fn test_activation_enum_leaky_relu_backward() {
        let act = Activation::LeakyReLU;
        let grad_output = vec![vec![1.0, 1.0, 1.0]];
        let input = vec![vec![-1.0, 0.0, 1.0]];
        let grad = act.backward(&grad_output, &input);
        assert_eq!(grad, vec![vec![0.01, 0.01, 1.0]]);
    }

    #[test]
    fn test_activation_enum_sigmoid_backward() {
        let act = Activation::Sigmoid;
        let grad_output = vec![vec![1.0]];
        let sigmoid_output = vec![vec![0.5]];
        let grad = act.backward(&grad_output, &sigmoid_output);
        assert!((grad[0][0] - 0.25).abs() < 1e-6);
    }

    // Weight initialization tests
    #[test]
    fn test_kaiming_init_variance() {
        // Kaiming init: std = sqrt(2/n_in), variance = 2/n_in
        let input_size = 100;
        let output_size = 50;
        let layer = Linear::new_with_init(input_size, output_size, InitStrategy::Kaiming);
        
        // Calculate variance of weights
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let count = (input_size * output_size) as f32;
        
        for row in &layer.weights {
            for &w in row {
                sum += w;
                sum_sq += w * w;
            }
        }
        
        let mean = sum / count;
        let variance = sum_sq / count - mean * mean;
        let expected_variance = 2.0 / input_size as f32;
        
        // Allow 50% tolerance due to random sampling
        assert!(
            (variance - expected_variance).abs() < expected_variance * 0.5,
            "Kaiming variance {} should be close to {}", variance, expected_variance
        );
    }

    #[test]
    fn test_xavier_init_variance() {
        // Xavier init: std = sqrt(2/(n_in + n_out)), variance = 2/(n_in + n_out)
        let input_size = 100;
        let output_size = 50;
        let layer = Linear::new_with_init(input_size, output_size, InitStrategy::Xavier);
        
        // Calculate variance of weights
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let count = (input_size * output_size) as f32;
        
        for row in &layer.weights {
            for &w in row {
                sum += w;
                sum_sq += w * w;
            }
        }
        
        let mean = sum / count;
        let variance = sum_sq / count - mean * mean;
        let expected_variance = 2.0 / (input_size + output_size) as f32;
        
        // Allow 50% tolerance due to random sampling
        assert!(
            (variance - expected_variance).abs() < expected_variance * 0.5,
            "Xavier variance {} should be close to {}", variance, expected_variance
        );
    }


    #[test]
    fn test_activation_init_strategy() {
        assert_eq!(Activation::ReLU.init_strategy(), InitStrategy::Kaiming);
        assert_eq!(Activation::LeakyReLU.init_strategy(), InitStrategy::Kaiming);
        assert_eq!(Activation::Sigmoid.init_strategy(), InitStrategy::Xavier);
    }

}