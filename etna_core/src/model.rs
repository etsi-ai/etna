// Model training / prediction logic
// Implements a simple 2-layer neural network with mandatory mini-batch training

use crate::layers::{Linear, Activation, InitStrategy};
use crate::softmax::Softmax;
use crate::loss_function::{cross_entropy, mse};
use crate::optimizer::{SGD, Adam};

use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};

// Required for shuffling training data each epoch
use rand::seq::SliceRandom;
use rand::rng;


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
    linear1: Linear,
    activation: Activation,
    linear2: Linear,
    task_type: TaskType,

    // Caches used during backpropagation
    input_cache: Vec<Vec<f32>>,
    hidden_cache: Vec<Vec<f32>>,
    logits_cache: Vec<Vec<f32>>,
    probs_cache: Vec<Vec<f32>>,
}

impl SimpleNN {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        task_code: usize,
        activation: Activation,
    ) -> Self {
        let task_type = if task_code == 1 {
            TaskType::Regression
        } else {
            TaskType::Classification
        };

        // Initialization strategy depends on the activation function
        let hidden_init = activation.init_strategy();
        let output_init = InitStrategy::Xavier;

        Self {
            linear1: Linear::new_with_init(input_dim, hidden_dim, hidden_init),
            activation,
            linear2: Linear::new_with_init(hidden_dim, output_dim, output_init),
            task_type,
            input_cache: vec![],
            hidden_cache: vec![],
            logits_cache: vec![],
            probs_cache: vec![],
        }
    }

    /// Forward pass (used by both training and prediction)
    pub fn forward(&mut self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let hidden_pre = self.linear1.forward(x);
        let hidden_post = self.activation.forward(&hidden_pre);
        let logits = self.linear2.forward(&hidden_post);

        let output = match self.task_type {
            TaskType::Classification => logits
                .iter()
                .map(|l| Softmax::forward(l))
                .collect(),
            TaskType::Regression => logits.clone(),
        };

        // Cache values needed for backprop
        self.input_cache = x.clone();
        self.hidden_cache = hidden_post;
        self.logits_cache = logits;
        self.probs_cache = output.clone();

        output
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

        // Separate optimizers per layer (important for Adam state)
        let mut sgd_l1 = match optimizer_type {
            OptimizerType::SGD => Some(if weight_decay > 0.0 {
                SGD::with_weight_decay(lr, weight_decay)
            } else {
                SGD::new(lr)
            }),
            OptimizerType::Adam => None,
        };

        let mut sgd_l2 = match optimizer_type {
            OptimizerType::SGD => Some(if weight_decay > 0.0 {
                SGD::with_weight_decay(lr, weight_decay)
            } else {
                SGD::new(lr)
            }),
            OptimizerType::Adam => None,
        };

        let mut adam_l1 = match optimizer_type {
            OptimizerType::SGD => None,
            OptimizerType::Adam => Some(Adam::new(lr, 0.9, 0.999, 1e-8)),
        };

        let mut adam_l2 = match optimizer_type {
            OptimizerType::SGD => None,
            OptimizerType::Adam => Some(Adam::new(lr, 0.9, 0.999, 1e-8)),
        };

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

                let x_batch: Vec<Vec<f32>> =
                    batch_indices.iter().map(|&i| x[i].clone()).collect();
                let y_batch: Vec<Vec<f32>> =
                    batch_indices.iter().map(|&i| y[i].clone()).collect();

                let preds = self.forward(&x_batch);

                let (loss, grad_output) = match self.task_type {
                    TaskType::Classification => {
                        let loss_val = cross_entropy(&preds, &y_batch);
                        let grad = Softmax::backward(&preds, &y_batch);
                        (loss_val, grad)
                    }
                    TaskType::Regression => {
                        let loss_val = mse(&preds, &y_batch);
                        let grad = preds
                            .iter()
                            .zip(y_batch.iter())
                            .map(|(p_row, y_row)| {
                                p_row
                                    .iter()
                                    .zip(y_row.iter())
                                    .map(|(p, t)| p - t)
                                    .collect()
                            })
                            .collect();
                        (loss_val, grad)
                    }
                };

                // ---- Backward pass ----
                let grad_linear2 = self.linear2.backward(&grad_output);
                let grad_activation =
                    self.activation.backward(&grad_linear2, &self.hidden_cache);
                let _grad_linear1 = self.linear1.backward(&grad_activation);

                // ---- Parameter update ----
                match optimizer_type {
                    OptimizerType::SGD => {
                        if let Some(ref mut opt) = sgd_l1 {
                            self.linear1.update_sgd(opt);
                        }
                        if let Some(ref mut opt) = sgd_l2 {
                            self.linear2.update_sgd(opt);
                        }
                    }
                    OptimizerType::Adam => {
                        if let Some(ref mut opt) = adam_l1 {
                            self.linear1.update_adam(opt);
                        }
                        if let Some(ref mut opt) = adam_l2 {
                            self.linear2.update_adam(opt);
                        }
                    }
                }

                epoch_loss += loss;
                batch_count += 1;
            }

            let avg_loss = epoch_loss / batch_count as f32;
            loss_history.push(avg_loss);

            if epoch % 10 == 0 {
                println!(
                    "Epoch {}/{} - Loss: {:.4}",
                    epoch, epochs, avg_loss
                );
            }
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

    #[test]
    fn training_loss_decreases_with_minibatch() {
        let mut model = SimpleNN::new(
            2, 4, 1, 1, Activation::ReLU
        );

        let x = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];

        let y = vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0],
        ];

        let losses = model.train(
            &x,
            &y,
            50,
            0.1,
            0.0,
            OptimizerType::SGD,
            2, // batch_size
        );

    assert!(losses.last().unwrap() < losses.first().unwrap());
    }
}
