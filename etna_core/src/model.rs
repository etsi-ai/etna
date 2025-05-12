// Model training/prediction logic
//

// model.rs - Full Working Version (Neural Network in Rust)

use crate::layers::{Linear, ReLU, Softmax};
use crate::loss_function::cross_entropy;
use crate::optimizer::SGD;

pub struct SimpleNN {
    linear1: Linear,
    relu: ReLU,
    linear2: Linear,
    softmax: Softmax,
    input_cache: Vec<Vec<f32>>,
    hidden_cache: Vec<Vec<f32>>,
    logits_cache: Vec<Vec<f32>>,
    probs_cache: Vec<Vec<f32>>,
}

impl SimpleNN {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            linear1: Linear::new(input_dim, hidden_dim),
            relu: ReLU,
            linear2: Linear::new(hidden_dim, output_dim),
            softmax: Softmax,
            input_cache: vec![],
            hidden_cache: vec![],
            logits_cache: vec![],
            probs_cache: vec![],
        }
    }

    // Forward pass for the entire batch (Vec<Vec<f32>>)
    pub fn forward(&mut self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut hidden_post: Vec<Vec<f32>> = vec![];

        // For each input (row) in the batch
        for input in x {
            let hidden_pre = self.linear1.forward(&vec![input.clone()]); // Forward pass for one sample
            let hidden_post_sample = ReLU::forward(&hidden_pre[0]); // Apply ReLU for one sample
            hidden_post.push(hidden_post_sample);
        }

        let mut logits: Vec<Vec<f32>> = vec![];
        for sample in hidden_post.iter() {
            let logit = self.linear2.forward(&vec![sample.clone()]);
            logits.push(logit[0]); // Assuming a batch size of 1 per sample
        }

        let mut probs: Vec<Vec<f32>> = vec![];
        for logit in logits.iter() {
            let prob = Softmax::forward(logit);
            probs.push(prob);
        }

        // Cache values
        self.input_cache = x.clone();
        self.hidden_cache = hidden_post.clone();
        self.logits_cache = logits.clone();
        self.probs_cache = probs.clone();

        probs
    }

    // Training function, adjust for batch processing
    pub fn train(&mut self, x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>, epochs: usize, lr: f32) {
        let mut optimizer = SGD::new(lr);

        for epoch in 0..epochs {
            let preds = self.forward(x);
            let loss = cross_entropy(&preds, y);

            let mut grad_softmax: Vec<Vec<f32>> = vec![];
            for (i, pred) in preds.iter().enumerate() {
                let grad = Softmax::backward(pred, &y[i]);
                grad_softmax.push(grad);
            }

            let mut grad_linear2: Vec<Vec<f32>> = vec![];
            for (i, grad) in grad_softmax.iter().enumerate() {
                let grad_layer2 = self.linear2.backward(grad, &self.hidden_cache[i]);
                grad_linear2.push(grad_layer2);
            }

            let mut grad_relu: Vec<Vec<f32>> = vec![];
            for (i, grad) in grad_linear2.iter().enumerate() {
                let grad_relu_layer = ReLU::backward(grad, &self.hidden_cache[i]);
                grad_relu.push(grad_relu_layer);
            }

            let mut grad_linear1: Vec<Vec<f32>> = vec![];
            for (i, grad) in grad_relu.iter().enumerate() {
                let grad_layer1 = self.linear1.backward(grad, &self.input_cache[i]);
                grad_linear1.push(grad_layer1);
            }

            // Update the weights after the backward pass
            self.linear1.update(&mut optimizer);
            self.linear2.update(&mut optimizer);

            println!("Epoch {}/{} - Loss: {:.4}", epoch + 1, epochs, loss);
        }
    }

    // Prediction function for the entire batch
    pub fn predict(&mut self, x: &Vec<Vec<f32>>) -> Vec<usize> {
        let probs = self.forward(x);
        probs.iter()
            .map(|row| row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0))
            .collect()
    }
}
