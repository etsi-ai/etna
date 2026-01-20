// Model training/prediction logic
use crate::layers::{Linear, Activation, InitStrategy};
use crate::softmax::Softmax;
use crate::loss_function::{cross_entropy, mse};
use crate::optimizer::{SGD, Adam};
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};

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
    input_cache: Vec<Vec<f32>>,
    hidden_cache: Vec<Vec<f32>>,
    logits_cache: Vec<Vec<f32>>,
    probs_cache: Vec<Vec<f32>>,
}

impl SimpleNN {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, task_code: usize, activation: Activation) -> Self {
        let task_type = if task_code == 1 { TaskType::Regression } else { TaskType::Classification };
        
        // Use appropriate initialization based on activation function:
        // - linear1 is followed by the activation, so use its recommended init
        // - linear2 is followed by Softmax (classification) or nothing (regression)
        //   For Softmax, Xavier is appropriate; for regression output, Xavier is also fine
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

    pub fn forward(&mut self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let hidden_pre = self.linear1.forward(x);
        let hidden_post = self.activation.forward(&hidden_pre);
        let logits = self.linear2.forward(&hidden_post);
        
        // Only apply Softmax for Classification
        let output = match self.task_type {
            TaskType::Classification => logits.iter().map(|l| Softmax::forward(l)).collect(),
            TaskType::Regression => logits.clone(),
        };

        self.input_cache = x.clone();
        self.hidden_cache = hidden_post;
        self.logits_cache = logits; 
        self.probs_cache = output.clone();

        output
    }

    pub fn train(&mut self, x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>, epochs: usize, lr: f32, weight_decay: f32, optimizer_type: OptimizerType, batch_size: usize, x_val: Option<&Vec<Vec<f32>>>, y_val: Option<&Vec<Vec<f32>>>) -> (Vec<f32>, Vec<f32>) {
        let mut loss_history = Vec::new(); // Training loss history
        let mut val_loss_history = Vec::new(); // Validation loss history

        // Create separate optimizer instances for each layer (persistent across epochs)
        // This is critical for Adam, as each layer has different dimensions
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

        let n_samples = x.len();
        let effective_batch_size = if batch_size == 0 { n_samples } else { batch_size.min(n_samples) };

        for epoch in 0..epochs {
            // Training: process in batches
            let mut epoch_loss = 0.0;
            let mut num_batches = 0;

            for batch_start in (0..n_samples).step_by(effective_batch_size) {
                let batch_end = (batch_start + effective_batch_size).min(n_samples);
                let x_batch: Vec<Vec<f32>> = x[batch_start..batch_end].to_vec();
                let y_batch: Vec<Vec<f32>> = y[batch_start..batch_end].to_vec();

                let preds = self.forward(&x_batch);
                
                let (loss, grad_output) = match self.task_type {
                    TaskType::Classification => {
                        let loss_val = cross_entropy(&preds, &y_batch);
                        let grad = Softmax::backward(&preds, &y_batch); 
                        (loss_val, grad)
                    },
                    TaskType::Regression => {
                        let loss_val = mse(&preds, &y_batch);
                        // Gradient of MSE: (pred - target)
                        let grad = preds.iter().zip(y_batch.iter())
                            .map(|(p_row, y_row)| {
                                p_row.iter().zip(y_row.iter()).map(|(p, t)| p - t).collect()
                            }).collect();
                        (loss_val, grad)
                    }
                };

                let grad_linear2 = self.linear2.backward(&grad_output);
                let grad_activation = self.activation.backward(&grad_linear2, &self.hidden_cache);
                let _grad_linear1 = self.linear1.backward(&grad_activation);

                // Update layers based on optimizer type (optimizers persist across batches)
                match optimizer_type {
                    OptimizerType::SGD => {
                        if let Some(ref mut opt) = sgd_l1 {
                            self.linear1.update_sgd(opt);
                        }
                        if let Some(ref mut opt) = sgd_l2 {
                            self.linear2.update_sgd(opt);
                        }
                    },
                    OptimizerType::Adam => {
                        if let Some(ref mut opt) = adam_l1 {
                            self.linear1.update_adam(opt);
                        }
                        if let Some(ref mut opt) = adam_l2 {
                            self.linear2.update_adam(opt);
                        }
                    },
                }

                epoch_loss += loss;
                num_batches += 1;
            }

            let avg_loss = epoch_loss / num_batches as f32;
            loss_history.push(avg_loss);

            // Calculate validation loss if validation data is provided
            if let (Some(x_val_data), Some(y_val_data)) = (x_val, y_val) {
                if !x_val_data.is_empty() && !y_val_data.is_empty() {
                    let val_preds = self.forward(x_val_data);
                    let val_loss = match self.task_type {
                        TaskType::Classification => cross_entropy(&val_preds, y_val_data),
                        TaskType::Regression => mse(&val_preds, y_val_data),
                    };
                    val_loss_history.push(val_loss);
                    
                    if epoch % 10 == 0 {
                        println!("Epoch {}/{} - Loss: {:.4}, Val Loss: {:.4}", epoch, epochs, avg_loss, val_loss);
                    }
                } else {
                    val_loss_history.push(f32::INFINITY);
                    if epoch % 10 == 0 {
                        println!("Epoch {}/{} - Loss: {:.4}", epoch, epochs, avg_loss);
                    }
                }
            } else {
                if epoch % 10 == 0 {
                    println!("Epoch {}/{} - Loss: {:.4}", epoch, epochs, avg_loss);
                }
            }
        }
        
        (loss_history, val_loss_history) // Return both training and validation loss histories
    }

    pub fn predict(&mut self, x: &Vec<Vec<f32>>) -> Vec<f32> {
        let output = self.forward(x);
        
        match self.task_type {
            TaskType::Classification => {
                // Return class indices as floats
                output.iter()
                    .map(|row| row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i as f32)
                        .unwrap_or(0.0))
                    .collect()
            },
            TaskType::Regression => {
                // Return value
                output.iter().map(|row| row[0]).collect()
            }
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let serialized = serde_json::to_string(self)?; // Convert struct to JSON string
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