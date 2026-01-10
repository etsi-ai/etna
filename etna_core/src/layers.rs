// # Layers (Linear, ReLU, Softmax, etc.)


use rand::Rng;
use crate::optimizer::SGD;
use serde::{Serialize, Deserialize};

/// Weight initialization strategy
/// - Xavier (Glorot): For layers followed by Sigmoid/Softmax. std = sqrt(2 / (n_in + n_out))
/// - Kaiming (He): For layers followed by ReLU/LeakyReLU. std = sqrt(2 / n_in)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitStrategy {
    /// Xavier/Glorot initialization - best for Sigmoid, Tanh, Softmax
    Xavier,
    /// Kaiming/He initialization - best for ReLU, LeakyReLU
    Kaiming,
    /// Legacy random initialization (-0.1 to 0.1)
    Legacy,
}

/// Fully connected layer: y = Wx + b
// Linear Layer (implementing forward, backward, and update)

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Linear {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    input_size: usize,
    output_size: usize,
    grad_weights: Vec<Vec<f32>>,  // Gradient for weights
    grad_bias: Vec<f32>,          // Gradient for biases
    cached_input: Vec<Vec<f32>>,  // Cache input for backward pass
}

impl Linear {
    /// Create a new Linear layer with legacy initialization (backward compatible)
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self::new_with_init(input_size, output_size, InitStrategy::Legacy)
    }

    /// Create a new Linear layer with specified initialization strategy
    /// 
    /// # Arguments
    /// * `input_size` - Number of input features
    /// * `output_size` - Number of output features  
    /// * `init` - Initialization strategy (Xavier, Kaiming, or Legacy)
    pub fn new_with_init(input_size: usize, output_size: usize, init: InitStrategy) -> Self {
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
            InitStrategy::Legacy => {
                // Legacy: uniform random between -0.1 and 0.1
                (0..output_size)
                    .map(|_| (0..input_size).map(|_| rng.random_range(-0.1..0.1)).collect())
                    .collect()
            },
        };
            
        let bias = vec![0.0; output_size];
        
        // Initialize gradients as 0.0
        let grad_weights = vec![vec![0.0; input_size]; output_size];
        let grad_bias = vec![0.0; output_size];
        let cached_input = vec![];

        Linear { weights, bias, input_size, output_size, grad_weights, grad_bias, cached_input }
    }

    /// Sample from standard normal distribution using Box-Muller transform
    fn sample_normal<R: Rng>(rng: &mut R) -> f32 {
        // Box-Muller transform for normal distribution
        let u1: f32 = rng.random_range(0.0001..1.0); // Avoid log(0)
        let u2: f32 = rng.random_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn forward(&mut self, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        self.cached_input = input.clone();
    
        input
            .iter()
            .map(|x| {
                self.weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        w.iter()
                            .zip(x.iter())
                            .map(|(w_val, x_val)| w_val * x_val)
                            .sum::<f32>()
                            + self.bias[i]
                    })
                    .collect::<Vec<f32>>()
            })
            .collect()
    }
    

    pub fn backward(&mut self, grad_output: &Vec<Vec<f32>>, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut grad_input = vec![vec![0.0; self.input_size]; input.len()];

        // Compute gradients for weights and biases
        for (i, grad) in grad_output.iter().enumerate() {
            for (j, grad_val) in grad.iter().enumerate() {
                self.grad_bias[j] += grad_val; // Sum gradients for bias
                for (k, &input_val) in input[i].iter().enumerate() {
                    self.grad_weights[j][k] += grad_val * input_val; // Gradient for weights
                    grad_input[i][k] += grad_val * self.weights[j][k]; // Gradient for input
                }
            }
        }
        grad_input
    }

    pub fn update(&mut self, optimizer: &mut SGD) {
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                // L2 regularization: add lambda * weight to gradient
                // grad_total = grad_loss + lambda * weight
                let l2_term = optimizer.weight_decay * self.weights[i][j];
                self.weights[i][j] -= optimizer.learning_rate * (self.grad_weights[i][j] + l2_term);
                self.grad_weights[i][j] = 0.0;
            }
            // Note: We don't apply weight decay to biases (standard practice)
            self.bias[i] -= optimizer.learning_rate * self.grad_bias[i];
            self.grad_bias[i] = 0.0;
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ReLU;

impl ReLU {
    pub fn forward(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        input.iter().map(|x| x.iter().map(|&v| v.max(0.0)).collect()).collect()
    }

    pub fn backward(grad_output: &Vec<Vec<f32>>, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        grad_output.iter().zip(input.iter())
            .map(|(grad, in_val)| grad.iter().zip(in_val.iter()).map(|(g, i)| if *i > 0.0 { *g } else { 0.0 }).collect())
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct Softmax;

impl Softmax {
    pub fn forward(logits: &Vec<f32>) -> Vec<f32> {
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|x| x / sum_exp).collect()
    }

    pub fn backward(preds: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        preds.iter().zip(y.iter())
            .map(|(p, t)| p.iter().zip(t.iter()).map(|(a, b)| a - b).collect())
            .collect()
    }    
}

/// Leaky ReLU activation: max(0.01 * x, x)
#[derive(Serialize, Deserialize)]
pub struct LeakyReLU;

impl LeakyReLU {
    const ALPHA: f32 = 0.01;

    pub fn forward(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        input.iter()
            .map(|x| x.iter().map(|&v| if v > 0.0 { v } else { Self::ALPHA * v }).collect())
            .collect()
    }

    pub fn backward(grad_output: &Vec<Vec<f32>>, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        grad_output.iter().zip(input.iter())
            .map(|(grad, in_val)| {
                grad.iter().zip(in_val.iter())
                    .map(|(g, i)| if *i > 0.0 { *g } else { Self::ALPHA * *g })
                    .collect()
            })
            .collect()
    }
}

/// Sigmoid activation: 1 / (1 + e^(-x))
#[derive(Serialize, Deserialize)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn forward(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        input.iter()
            .map(|x| x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect())
            .collect()
    }

    /// Backward pass for Sigmoid
    /// Derivative: sigmoid(x) * (1 - sigmoid(x))
    /// For efficiency, we use the output of forward pass: output * (1 - output)
    pub fn backward(grad_output: &Vec<Vec<f32>>, sigmoid_output: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        grad_output.iter().zip(sigmoid_output.iter())
            .map(|(grad, out)| {
                grad.iter().zip(out.iter())
                    .map(|(g, o)| g * o * (1.0 - o))
                    .collect()
            })
            .collect()
    }
}

/// Configurable activation function enum
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    ReLU,
    LeakyReLU,
    Sigmoid,
}

impl Activation {
    /// Apply forward pass using the selected activation
    pub fn forward(&self, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        match self {
            Activation::ReLU => ReLU::forward(input),
            Activation::LeakyReLU => LeakyReLU::forward(input),
            Activation::Sigmoid => Sigmoid::forward(input),
        }
    }

    /// Apply backward pass using the selected activation
    /// For Sigmoid, pass the cached output from forward pass
    pub fn backward(&self, grad_output: &Vec<Vec<f32>>, input_or_output: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        match self {
            Activation::ReLU => ReLU::backward(grad_output, input_or_output),
            Activation::LeakyReLU => LeakyReLU::backward(grad_output, input_or_output),
            Activation::Sigmoid => {
                // For sigmoid, we need to pass the output, not input
                // But since the backward is called with hidden_cache which is the output of forward,
                // we can use it directly
                Sigmoid::backward(grad_output, input_or_output)
            },
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
    fn test_legacy_init_range() {
        // Legacy init: uniform between -0.1 and 0.1
        let layer = Linear::new_with_init(50, 30, InitStrategy::Legacy);
        
        for row in &layer.weights {
            for &w in row {
                assert!(w >= -0.1 && w <= 0.1, "Legacy weight {} out of range", w);
            }
        }
    }

    #[test]
    fn test_activation_init_strategy() {
        assert_eq!(Activation::ReLU.init_strategy(), InitStrategy::Kaiming);
        assert_eq!(Activation::LeakyReLU.init_strategy(), InitStrategy::Kaiming);
        assert_eq!(Activation::Sigmoid.init_strategy(), InitStrategy::Xavier);
    }

    #[test]
    fn test_linear_new_uses_legacy() {
        // Linear::new() should use legacy initialization for backward compatibility
        let layer = Linear::new(10, 5);
        
        for row in &layer.weights {
            for &w in row {
                assert!(w >= -0.1 && w <= 0.1, "Default init weight {} out of range", w);
            }
        }
    }
}
