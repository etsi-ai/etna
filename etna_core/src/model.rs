// Model training/prediction logic
use crate::layers::{Linear, ReLU, Softmax};
use crate::loss_function::{cross_entropy, mse};
use crate::optimizer::SGD;

#[derive(Clone, Copy, PartialEq)]
pub enum TaskType {
    Classification,
    Regression,
}

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

    pub fn train(&mut self, x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>, epochs: usize, lr: f32) {
        let mut optimizer = SGD::new(lr);

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
            let grad_relu = ReLU::backward(&grad_linear2, &self.hidden_cache);
            let _grad_linear1 = self.linear1.backward(&grad_relu, &self.input_cache);

            self.linear1.update(&mut optimizer);
            self.linear2.update(&mut optimizer);

            if epoch % 10 == 0 {
                println!("Epoch {}/{} - Loss: {:.4}", epoch, epochs, loss);
            }
        }
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
}
