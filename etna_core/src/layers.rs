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
        self.cached_input = input.clone(); // Cache input for backpropagation
        input.iter().map(|x| {
            self.weights.iter()
                .map(|w| w.iter().zip(x.iter()).map(|(w_val, x_val)| w_val * x_val).sum::<f32>() + self.bias.iter().sum::<f32>())
                .collect::<Vec<f32>>()
        }).collect()
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
