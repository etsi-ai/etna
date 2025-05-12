// Model training/prediction logic
use crate::layers::{Linear, ReLU, Softmax};
use crate::loss_function::cross_entropy;
use crate::optimizer::SGD;


pub struct SimpleNN {
    linear1: Linear,
    relu: ReLU,
    linear2: Linear,
    softmax: Softmax,
    input_cache: Vec<Vec<f32>>,  // Cache input for backprop
    hidden_cache: Vec<Vec<f32>>, // Cache hidden layer values
    logits_cache: Vec<Vec<f32>>, // Cache logits
    probs_cache: Vec<Vec<f32>>, // Cache probabilities
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
        let hidden_pre = self.linear1.forward(x); // Forward pass for linear1
        let hidden_post = ReLU::forward(&hidden_pre); // ReLU activation
        let logits = self.linear2.forward(&hidden_post); // Forward pass for linear2
        let probs = logits.iter().map(|logit| Softmax::forward(logit)).collect::<Vec<Vec<f32>>>(); // Softmax activation

        // Cache intermediate values
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

            let grad_softmax = Softmax::backward(&preds, y); // Softmax backward pass
            let grad_linear2 = self.linear2.backward(&grad_softmax, &self.hidden_cache); // Linear2 backward pass
            let grad_relu = ReLU::backward(&grad_linear2, &self.hidden_cache); // ReLU backward pass
            let _grad_linear1 = self.linear1.backward(&grad_relu, &self.input_cache); // Linear1 backward pass

            // Update parameters using stored gradients
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
