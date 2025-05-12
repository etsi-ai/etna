// # Layers (Linear, ReLU, Softmax, etc.)


// use rand::Rng;

use crate::optimizer::SGD;

/// Fully connected layer: y = Wx + b
// Linear Layer (implementing forward, backward, and update)
pub struct Linear {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = vec![vec![0.0; input_size]; output_size];
        let bias = vec![0.0; output_size];
        Linear { weights, bias, input_size, output_size }
    }

    pub fn forward(&self, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        input.iter().map(|x| {
            self.weights.iter()
                .map(|w| w.iter().zip(x.iter()).map(|(w_val, x_val)| w_val * x_val).sum::<f32>() + self.bias.iter().sum::<f32>())
                .collect::<Vec<f32>>()
        }).collect()
    }

    pub fn backward(&self, grad_output: &Vec<Vec<f32>>, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        grad_output.iter().map(|grad| {
            self.weights.iter().map(|weight| grad.iter().zip(weight.iter()).map(|(g, w)| g * w).sum::<f32>())
                .collect::<Vec<f32>>()
        }).collect()
    }

    pub fn update(&mut self, optimizer: &mut SGD) {
        for (w, b) in self.weights.iter_mut().zip(self.bias.iter_mut()) {
            for weight in w {
                *weight -= optimizer.learning_rate; // Simple gradient descent
            }
            *b -= optimizer.learning_rate;
        }
    }
}

// ReLU Layer (implementing forward and backward)
pub struct ReLU;

impl ReLU {
    pub fn forward(input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|x| x.max(0.0)).collect()
    }

    pub fn backward(grad_output: &Vec<f32>, input: &Vec<f32>) -> Vec<f32> {
        grad_output.iter().zip(input.iter())
            .map(|(g, i)| if *i > 0.0 { *g } else { 0.0 })
            .collect()
    }
}

// Softmax Layer (implementing forward and backward)
pub struct Softmax;

impl Softmax {
    pub fn forward(logits: &Vec<f32>) -> Vec<f32> {
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|x| x / sum_exp).collect()
    }

    pub fn backward(preds: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
        preds.iter().zip(y.iter()).map(|(p, t)| p - t).collect()
    }
}
