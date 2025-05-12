// Model training/prediction logic
//

use crate::layers::{Linear, ReLU, Softmax};
use crate::loss::cross_entropy;
use crate::optim::SGD;

pub struct SimpleNN {
    linear1: Linear,
    relu: ReLU,
    linear2: Linear,
    softmax: Softmax,
}

impl SimpleNN {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            linear1: Linear::new(input_dim, hidden_dim),
            relu: ReLU::new(),
            linear2: Linear::new(hidden_dim, output_dim),
            softmax: Softmax::new(),
        }
    }

    pub fn forward(&mut self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let out1 = self.linear1.forward(x);
        let out2 = self.relu.forward(&out1);
        let out3 = self.linear2.forward(&out2);
        self.softmax.forward(&out3)
    }

    pub fn train(
        &mut self,
        x: &Vec<Vec<f32>>,
        y: &Vec<Vec<f32>>,
        epochs: usize,
        lr: f32,
    ) {
        let mut optimizer = SGD::new(lr);

        for epoch in 0..epochs {
            // Forward pass
            let out1 = self.linear1.forward(x);
            let out2 = self.relu.forward(&out1);
            let out3 = self.linear2.forward(&out2);
            let preds = self.softmax.forward(&out3);

            // Loss + gradient
            let loss = cross_entropy(&preds, y);
            let grad = self.softmax.backward(y);

            // Backward pass
            let grad2 = self.linear2.backward(&grad, &out2);
            let grad1 = self.relu.backward(&grad2);
            let _ = self.linear1.backward(&grad1, x);

            // Update
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
