// # Layers (Linear, ReLU, Softmax, etc.)


use rand::Rng;
use crate::optimizer::SGD;
use serde::{Serialize, Deserialize};

/// Fully connected layer: y = Wx + b
// Linear Layer (implementing forward, backward, and update)
// Linear Layer (implementing forward, backward, and update)

#[derive(Serialize, Deserialize)]
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
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        // Initialize weights with small random values (e.g., between -0.1 and 0.1)
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
            
        let bias = vec![0.0; output_size];
        
        // Initialize gradients as 0.0
        let grad_weights = vec![vec![0.0; input_size]; output_size];
        let grad_bias = vec![0.0; output_size];
        let cached_input = vec![];

        Linear { weights, bias, input_size, output_size, grad_weights, grad_bias, cached_input }
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
                self.weights[i][j] -= optimizer.learning_rate * self.grad_weights[i][j];
                self.grad_weights[i][j] = 0.0;
            }
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
}
