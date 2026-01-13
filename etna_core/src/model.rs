// Model training/prediction logic
use crate::layers::{Linear, Activation, Softmax, InitStrategy};
use crate::loss_function::{cross_entropy, mse};
use crate::optimizer::SGD;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Write, Read};

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
    Classification,
    Regression,
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

    pub fn train(
        &mut self,
        x: &Vec<Vec<f32>>,
        y: &Vec<Vec<f32>>,
        x_val: Option<&Vec<Vec<f32>>>,
        y_val: Option<&Vec<Vec<f32>>>,
        epochs: usize,
        lr: f32,
        weight_decay: f32,
        early_stopping: bool,
        patience: usize,
    ) -> Vec<f32> {

        let mut optimizer = SGD::with_weight_decay(lr, weight_decay);
        let mut loss_history = Vec::new(); // Create list
        let mut best_loss = f32::INFINITY;
        let mut epochs_without_improve = 0;
        let mut best_linear1: Option<Linear> = None;
        let mut best_linear2: Option<Linear> = None;

        for epoch in 0..epochs {
            let preds = self.forward(x);
            
            let (loss, grad_output) = match self.task_type {
                TaskType::Classification => {
                    let loss_val = cross_entropy(&preds, y);
                    let grad = Softmax::backward(&preds, y); 
                    (loss_val, grad)
                },
                TaskType::Regression => {
                    let loss_val = mse(&preds, y);
                    // Gradient of MSE: (pred - target)
                    let grad = preds.iter().zip(y.iter())
                        .map(|(p_row, y_row)| {
                            p_row.iter().zip(y_row.iter()).map(|(p, t)| p - t).collect()
                        }).collect();
                    (loss_val, grad)
                }
            };

            let grad_linear2 = self.linear2.backward(&grad_output, &self.hidden_cache);
            let grad_activation = self.activation.backward(&grad_linear2, &self.hidden_cache);
            let _grad_linear1 = self.linear1.backward(&grad_activation, &self.input_cache);

            self.linear1.update(&mut optimizer);
            self.linear2.update(&mut optimizer);

            loss_history.push(loss); // Store loss
// ---------- EARLY STOPPING CHECK ----------
            let monitor_loss = if let (Some(xv), Some(yv)) = (x_val, y_val) {
                let val_preds = self.forward(xv);
                match self.task_type {
                    TaskType::Classification => cross_entropy(&val_preds, yv),
                    TaskType::Regression => mse(&val_preds, yv),
                }
            } else {
                loss
            };

            if early_stopping {
                if monitor_loss < best_loss {
                    best_loss = monitor_loss;
                    epochs_without_improve = 0;

                    // 🔥 STORE BEST WEIGHTS IN RAM
                    best_linear1 = Some(self.linear1.clone());
                    best_linear2 = Some(self.linear2.clone());
                } else {
                    epochs_without_improve += 1;
                }

                if epochs_without_improve >= patience {
                    println!(
                        "Early stopping at epoch {} (best val loss {:.4})",
                        epoch, best_loss
                    );
                    break;
                }
            }
            // ----------------------------------------


            if epoch % 10 == 0 {
                println!("Epoch {}/{} - Loss: {:.4}", epoch, epochs, loss);
            }
        }
        if early_stopping {
            if let (Some(l1), Some(l2)) = (best_linear1, best_linear2) {
                self.linear1 = l1;
                self.linear2 = l2;
            }
        }

        loss_history // Return list
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
