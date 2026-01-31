use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::optimizer::{Adam, Sgd};

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

struct Softmax;

impl Softmax {
    fn forward(logits: &[f32]) -> Vec<f32> {
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|x| x / sum_exp).collect()
    }

    fn backward(preds: &[Vec<f32>], y: &[Vec<f32>]) -> Vec<Vec<f32>> {
        preds.iter().zip(y.iter())
            .map(|(p, t)| p.iter().zip(t.iter()).map(|(a, b)| a - b).collect())
            .collect()
    }
}

/// Unified trait for all model layers
pub trait Layer {
    /// Forward pass through the layer
    fn forward(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Backward pass through the layer (backpropagation)
    fn backward(&mut self, grad_output: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Update layer parameters using Sgd
    fn update_sgd(&mut self, _optimizer: &Sgd) {}

    /// Update layer parameters using Adam
    fn update_adam(&mut self, _optimizer: &mut Adam) {}
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

    pub fn forward(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.cached_input = input.to_owned();

        input.iter().map(|x| {
            self.weights.iter().enumerate().map(|(i, w)| {
                w.iter().zip(x.iter()).map(|(w, x)| w * x).sum::<f32>() + self.bias[i]
            }).collect()
        }).collect()
    }

    pub fn backward(&mut self, grad_output: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let batch_size = self.cached_input.len();
        let input_size = self.cached_input[0].len();

        let mut grad_input = vec![vec![0.0; input_size]; batch_size];

        for (i, grad_in_row) in grad_input.iter_mut().enumerate() {
            let go_row = &grad_output[i];
            let cached_row = &self.cached_input[i];
            for (j, (grad_w_row, weight_row)) in self
                .grad_weights
                .iter_mut()
                .zip(self.weights.iter())
                .enumerate()
            {
                let g = go_row[j];
                self.grad_bias[j] += g;
                for (k, (gw, w)) in grad_w_row.iter_mut().zip(weight_row.iter()).enumerate() {
                    *gw += g * cached_row[k];
                    grad_in_row[k] += g * w;
                }
            }
        }
        grad_input
    }

    /// Update weights using Sgd
    pub fn update_sgd(&mut self, optimizer: &Sgd) {
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

impl Layer for Linear {
    fn forward(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.forward(input)
    }

    fn backward(&mut self, grad_output: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.backward(grad_output)
    }

    fn update_sgd(&mut self, optimizer: &Sgd) {
        self.update_sgd(optimizer)
    }

    fn update_adam(&mut self, optimizer: &mut Adam) {
        self.update_adam(optimizer)
    }
}

/// =======================
/// ReLU
/// =======================
#[derive(Serialize, Deserialize)]
pub struct ReLU;

impl ReLU {
    pub fn forward(input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input.iter().map(|x| x.iter().map(|&v| v.max(0.0)).collect()).collect()
    }

    pub fn backward(grad: &[Vec<f32>], input: &[Vec<f32>]) -> Vec<Vec<f32>> {
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

    pub fn forward(input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input.iter().map(|x| {
            x.iter().map(|&v| if v > 0.0 { v } else { Self::ALPHA * v }).collect()
        }).collect()
    }

    pub fn backward(grad: &[Vec<f32>], input: &[Vec<f32>]) -> Vec<Vec<f32>> {
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
    pub fn forward(input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        input.iter().map(|x| {
            x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
        }).collect()
    }

    pub fn backward(grad: &[Vec<f32>], out: &[Vec<f32>]) -> Vec<Vec<f32>> {
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
    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        match self {
            Self::ReLU => ReLU::forward(input),
            Self::LeakyReLU => LeakyReLU::forward(input),
            Self::Sigmoid => Sigmoid::forward(input),
        }
    }

    pub fn backward(&self, grad: &[Vec<f32>], cache: &[Vec<f32>]) -> Vec<Vec<f32>> {
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

/// Wrapper for activations to implement the Layer trait
#[derive(Serialize, Deserialize)]
pub struct ActivationLayer {
    pub activation: Activation,
    cached_data: Vec<Vec<f32>>,
}

impl ActivationLayer {
    pub fn new(activation: Activation) -> Self {
        Self {
            activation,
            cached_data: vec![],
        }
    }
}

impl Layer for ActivationLayer {
    fn forward(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let output = self.activation.forward(input);
        // Sigmoid backward needs output, ReLU/LeakyReLU need input
        match self.activation {
            Activation::Sigmoid => self.cached_data = output.clone(),
            _ => self.cached_data = input.to_owned(),
        }
        output
    }

    fn backward(&mut self, grad_output: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.activation.backward(grad_output, &self.cached_data)
    }
}

/// Wrapper for Softmax to implement the Layer trait
#[derive(Serialize, Deserialize, Default)]
pub struct SoftmaxLayer {
    cached_output: Vec<Vec<f32>>,
}

impl SoftmaxLayer {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Layer for SoftmaxLayer {
    fn forward(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let output = input.iter().map(|l| Softmax::forward(l)).collect();
        self.cached_output = output;
        self.cached_output.clone()
    }

    fn backward(&mut self, grad_output: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Note: For the output layer, grad_output is expected to be the targets (Y)
        // due to the specific implementation of Softmax::backward returning (preds - targets)
        Softmax::backward(&self.cached_output, grad_output)
    }
}

/// Polymorphic wrapper for all layer types to support serialization
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayerWrapper {
    Linear(Linear),
    Activation(ActivationLayer),
    Softmax(SoftmaxLayer),
}

impl Layer for LayerWrapper {
    fn forward(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        match self {
            Self::Linear(l) => l.forward(input),
            Self::Activation(a) => a.forward(input),
            Self::Softmax(s) => s.forward(input),
        }
    }

    fn backward(&mut self, grad_output: &[Vec<f32>]) -> Vec<Vec<f32>> {
        match self {
            Self::Linear(l) => l.backward(grad_output),
            Self::Activation(a) => a.backward(grad_output),
            Self::Softmax(s) => s.backward(grad_output),
        }
    }

    fn update_sgd(&mut self, optimizer: &Sgd) {
        if let Self::Linear(l) = self {
            l.update_sgd(optimizer);
        }
    }

    fn update_adam(&mut self, optimizer: &mut Adam) {
        if let Self::Linear(l) = self {
            l.update_adam(optimizer);
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
