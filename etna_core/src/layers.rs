use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::optimizer::{SGD, Adam};

/// =======================
/// Linear Layer
/// =======================
#[derive(Serialize, Deserialize)]
pub struct Linear {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    grad_weights: Vec<Vec<f32>>,
    grad_bias: Vec<f32>,
    cached_input: Vec<Vec<f32>>,
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        Self {
            weights,
            bias: vec![0.0; output_size],
            grad_weights: vec![vec![0.0; input_size]; output_size],
            grad_bias: vec![0.0; output_size],
            cached_input: vec![],
        }
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

    pub fn update_sgd(&mut self, opt: &SGD) {
        for i in 0..self.weights.len() {
            for j in 0..self.weights[0].len() {
                self.weights[i][j] -= opt.learning_rate * self.grad_weights[i][j];
                self.grad_weights[i][j] = 0.0;
            }
            self.bias[i] -= opt.learning_rate * self.grad_bias[i];
            self.grad_bias[i] = 0.0;
        }
    }

    pub fn update_adam(&mut self, opt: &mut Adam) {
        opt.step(&mut self.weights, &self.grad_weights, &mut self.bias, &self.grad_bias);
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
            .map(|(g, o)| g.iter().zip(o.iter()).map(|(g, o)| g * o * (1.0 - o)).collect())
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
}
