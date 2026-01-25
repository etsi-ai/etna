// Model training / prediction logic
// Implements a simple 2-layer neural network with mandatory mini-batch training

use crate::layers::{Activation, InitStrategy, Linear, Layer, LayerWrapper, ActivationLayer, SoftmaxLayer};
use crate::loss_function::{cross_entropy, mse};
use crate::optimizer::{Adam, SGD};

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

// Required for shuffling training data each epoch
use rand::rng;
use rand::seq::SliceRandom;

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    Classification,
    Regression,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum OptimizerType {
    SGD,
    Adam,
}

#[derive(Serialize, Deserialize)]
pub struct SimpleNN {
    layers: Vec<LayerWrapper>,
    task_type: TaskType,
    optimizers: Vec<Option<LayerOptimizer>>,
}

#[derive(Serialize, Deserialize)]
enum LayerOptimizer {
    SGD(SGD),
    Adam(Adam),
}

impl SimpleNN {
    pub fn new(
        input_dim: usize,
        hidden_layers: Vec<usize>,
        output_dim: usize,
        task_code: usize,
        activation: Activation,
    ) -> Self {
        let task_type = if task_code == 1 { TaskType::Regression } else { TaskType::Classification };
        let mut layers = Vec::new();
        let mut current_in = input_dim;

        // Build the dynamic layer stack
        for &hidden_dim in &hidden_layers {
            layers.push(LayerWrapper::Linear(Linear::new_with_init(
                current_in,
                hidden_dim,
                activation.init_strategy(),
            )));
            layers.push(LayerWrapper::Activation(ActivationLayer::new(activation)));
            current_in = hidden_dim;
        }

        // Output Layer
        layers.push(LayerWrapper::Linear(Linear::new_with_init(
            current_in,
            output_dim,
            InitStrategy::Xavier,
        )));

        if task_type == TaskType::Classification {
            layers.push(LayerWrapper::Softmax(SoftmaxLayer::new()));
        }

        Self { layers, task_type, optimizers: vec![] }
    }

    /// Forward pass (used by both training and prediction)
    pub fn forward(&mut self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut current_input = x.clone();

        for layer in &mut self.layers {
            current_input = layer.forward(&current_input);
        }

        current_input
    }

    /// Train the network using mandatory mini-batch training
    pub fn train(
        &mut self,
        x: &Vec<Vec<f32>>,
        y: &Vec<Vec<f32>>,
        epochs: usize,
        lr: f32,
        weight_decay: f32,
        optimizer_type: OptimizerType,
        batch_size: usize,
    ) -> Vec<f32> {
        let mut loss_history = Vec::new();

        // Initialize optimizers ONLY if they don't exist yet
        if self.optimizers.is_empty() {
            self.optimizers = self.layers.iter().map(|l| {
                if let LayerWrapper::Linear(_) = l {
                    match optimizer_type {
                        OptimizerType::SGD => Some(LayerOptimizer::SGD(if weight_decay > 0.0 {
                            SGD::with_weight_decay(lr, weight_decay)
                        } else {
                            SGD::new(lr)
                        })),
                        OptimizerType::Adam => Some(LayerOptimizer::Adam(Adam::new(lr, 0.9, 0.999, 1e-8))),
                    }
                } else {
                    None
                }
            }).collect();
        }

        for epoch in 0..epochs {
            // ---- Shuffle data at the start of each epoch ----
            let mut indices: Vec<usize> = (0..x.len()).collect();
            indices.shuffle(&mut rng());

            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Iterate over shuffled data using fixed-size mini-batches

            for batch_start in (0..x.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(x.len());
                let batch_indices = &indices[batch_start..batch_end];

                let x_batch: Vec<Vec<f32>> = batch_indices.iter().map(|&i| x[i].clone()).collect();
                let y_batch: Vec<Vec<f32>> = batch_indices.iter().map(|&i| y[i].clone()).collect();

                let preds = self.forward(&x_batch);

                let (loss, grad_output) = match self.task_type {
                    TaskType::Classification => {
                        let loss_val = cross_entropy(&preds, &y_batch);
                        // For classification, the last layer is Softmax,
                        // which expects targets for backward pass to compute (preds - target)
                        (loss_val, y_batch)
                    }
                    TaskType::Regression => {
                        let loss_val = mse(&preds, &y_batch);
                        // For regression, dL/dpreds = preds - target
                        let grad = preds.iter().zip(y_batch.iter())
                            .map(|(p_row, y_row)| p_row.iter().zip(y_row.iter()).map(|(p, t)| p - t).collect())
                            .collect();
                        (loss_val, grad)
                    }
                };

                // ---- Backward pass (Reverse iteration) ----
                let mut current_grad = grad_output;
                for layer in self.layers.iter_mut().rev() {
                    current_grad = layer.backward(&current_grad);
                }

                // ---- Parameter update ----
                for (layer, opt) in self.layers.iter_mut().zip(self.optimizers.iter_mut()) {
                    if let Some(ref mut o) = opt {
                        match o {
                            LayerOptimizer::SGD(s) => layer.update_sgd(s),
                            LayerOptimizer::Adam(a) => layer.update_adam(a),
                        }
                    }
                }

                epoch_loss += loss;
                batch_count += 1;
            }

            let avg_loss = epoch_loss / batch_count as f32;
            loss_history.push(avg_loss);

            if epoch % 10 == 0 {
                println!("Epoch {}/{} - Loss: {:.4}", epoch, epochs, avg_loss);
            }
        }

        loss_history
    }

    /// Train the network with a progress callback
    /// The callback is called after each epoch with (epoch, total_epochs, loss)
    pub fn train_with_callback<F>(
        &mut self,
        x: &Vec<Vec<f32>>,
        y: &Vec<Vec<f32>>,
        epochs: usize,
        lr: f32,
        weight_decay: f32,
        optimizer_type: OptimizerType,
        batch_size: usize,
        progress_callback: F,
    ) -> Vec<f32>
    where
        F: Fn(usize, usize, f32),
    {
        let mut loss_history = Vec::new();

        // Initialize optimizers ONLY if they don't exist yet
        if self.optimizers.is_empty() {
            self.optimizers = self.layers.iter().map(|l| {
                if let LayerWrapper::Linear(_) = l {
                    match optimizer_type {
                        OptimizerType::SGD => Some(LayerOptimizer::SGD(if weight_decay > 0.0 {
                            SGD::with_weight_decay(lr, weight_decay)
                        } else {
                            SGD::new(lr)
                        })),
                        OptimizerType::Adam => Some(LayerOptimizer::Adam(Adam::new(lr, 0.9, 0.999, 1e-8))),
                    }
                } else {
                    None
                }
            }).collect();
        }

        for epoch in 0..epochs {
            // ---- Shuffle data at the start of each epoch ----
            let mut indices: Vec<usize> = (0..x.len()).collect();
            indices.shuffle(&mut rng());

            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // Iterate over shuffled data using fixed-size mini-batches
            for batch_start in (0..x.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(x.len());
                let batch_indices = &indices[batch_start..batch_end];

                let x_batch: Vec<Vec<f32>> = batch_indices.iter().map(|&i| x[i].clone()).collect();
                let y_batch: Vec<Vec<f32>> = batch_indices.iter().map(|&i| y[i].clone()).collect();

                let preds = self.forward(&x_batch);

                let (loss, grad_output) = match self.task_type {
                    TaskType::Classification => {
                        let loss_val = cross_entropy(&preds, &y_batch);
                        (loss_val, y_batch)
                    }
                    TaskType::Regression => {
                        let loss_val = mse(&preds, &y_batch);
                        let grad = preds.iter().zip(y_batch.iter())
                            .map(|(p_row, y_row)| p_row.iter().zip(y_row.iter()).map(|(p, t)| p - t).collect())
                            .collect();
                        (loss_val, grad)
                    }
                };

                // ---- Backward pass (Reverse iteration) ----
                let mut current_grad = grad_output;
                for layer in self.layers.iter_mut().rev() {
                    current_grad = layer.backward(&current_grad);
                }

                // ---- Parameter update ----
                for (layer, opt) in self.layers.iter_mut().zip(self.optimizers.iter_mut()) {
                    if let Some(ref mut o) = opt {
                        match o {
                            LayerOptimizer::SGD(s) => layer.update_sgd(s),
                            LayerOptimizer::Adam(a) => layer.update_adam(a),
                        }
                    }
                }

                epoch_loss += loss;
                batch_count += 1;
            }

            let avg_loss = epoch_loss / batch_count as f32;
            loss_history.push(avg_loss);

            // Call the progress callback instead of printing
            progress_callback(epoch, epochs, avg_loss);
        }

        loss_history
    }

    /// Run inference on input data (no gradient tracking)

    pub fn predict(&mut self, x: &Vec<Vec<f32>>) -> Vec<f32> {
        let output = self.forward(x);

        match self.task_type {
            TaskType::Classification => output
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as f32)
                        .unwrap_or(0.0)
                })
                .collect(),
            TaskType::Regression => output.iter().map(|row| row[0]).collect(),
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let serialized = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let model: SimpleNN = serde_json::from_str(&contents)?;
        Ok(model)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that mini-batch training reduces loss over time.
    ///
    /// This test uses a tiny XOR-like dataset and ensures that
    /// average training loss decreases when using mini-batches.
    #[test]
    fn training_loss_decreases_with_minibatch() {
        let mut model = SimpleNN::new(
            2, // input_dim
            vec![4], // hidden_layers
            1, // output_dim
            1, // regression task
            Activation::ReLU,
        );

        let x = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];

        let y = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let losses = model.train(
            &x,
            &y,
            50,                 // epochs
            0.1,                // learning rate
            0.0,                // weight decay
            OptimizerType::SGD, // optimizer
            2,                  // batch size
        );

        // Ensure training progresses in the right direction
        assert!(
            losses.last().unwrap() < losses.first().unwrap(),
            "Expected training loss to decrease with mini-batch training"
        );
    }
}
