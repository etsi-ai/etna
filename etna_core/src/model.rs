// Model training/prediction logic
use crate::layers::{Linear, ReLU, Softmax};
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
    relu: ReLU,
    linear2: Linear,
    task_type: TaskType,
    input_cache: Vec<Vec<f32>>,
    hidden_cache: Vec<Vec<f32>>,
    logits_cache: Vec<Vec<f32>>,
    probs_cache: Vec<Vec<f32>>,
}

impl SimpleNN {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, task_code: usize) -> Self {
        let task_type = if task_code == 1 { TaskType::Regression } else { TaskType::Classification };
        
        Self {
            linear1: Linear::new(input_dim, hidden_dim),
            relu: ReLU,
            linear2: Linear::new(hidden_dim, output_dim),
            task_type, // 0 = Classification, 1 = Regression
            input_cache: vec![],
            hidden_cache: vec![],
            logits_cache: vec![],
            probs_cache: vec![],
        }
    }

    pub fn train(&mut self, x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>, epochs: usize, lr: f32) -> Vec<f32> {
        self.train_with_progress(x, y, epochs, lr, None)
    }

    pub fn forward(&mut self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let hidden_pre = self.linear1.forward(x);
        let hidden_post = ReLU::forward(&hidden_pre);
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

    pub fn train_with_progress(
        &mut self,
        x: &Vec<Vec<f32>>,
        y: &Vec<Vec<f32>>,
        epochs: usize,
        lr: f32,
        mut progress: Option<&mut dyn FnMut(usize, f32)>,
    ) -> Vec<f32> {
        let mut optimizer = SGD::new(lr);
        let mut loss_history = Vec::new();

        for _epoch in 0..epochs {
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
            let grad_relu = ReLU::backward(&grad_linear2, &self.hidden_cache);
            let _grad_linear1 = self.linear1.backward(&grad_relu, &self.input_cache);

            self.linear1.update(&mut optimizer);
            self.linear2.update(&mut optimizer);

            loss_history.push(loss);

            if let Some(cb) = progress.as_mut() {
                cb(loss_history.len(), loss);
            }
        }

        loss_history
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
