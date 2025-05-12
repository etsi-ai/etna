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

    pub fn forward(&mut self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let hidden_pre = self.linear1.forward(x);
        let hidden_post = ReLU::forward(&hidden_pre);
        let logits = self.linear2.forward(&hidden_post);
        let probs = logits.iter().map(|logit| Softmax::forward(logit)).collect::<Vec<Vec<f32>>>();

        // Cache values
        self.input_cache = x.clone();
        self.hidden_cache = hidden_post;
        self.logits_cache = logits;
        self.probs_cache = probs.clone();

        probs
    }

    pub fn train(&mut self, x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>, epochs: usize, lr: f32) {
        let mut optimizer = SGD::new(lr);

        for epoch in 0..epochs {
            let preds = self.forward(x);
            let loss = cross_entropy(&preds, y);

            let grad_softmax = Softmax::backward(&preds, y);
            let grad_linear2 = self.linear2.backward(&grad_softmax, &self.hidden_cache);
            let grad_relu = ReLU::backward(&grad_linear2, &self.hidden_cache);
            let _grad_linear1 = self.linear1.backward(&grad_relu, &self.input_cache);

            self.linear1.update(&mut optimizer);
            self.linear2.update(&mut optimizer);

            println!("Epoch {}/{} - Loss: {:.4}", epoch + 1, epochs, loss);
        }
    }

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
