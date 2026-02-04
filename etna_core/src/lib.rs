// Rust-Python bridge (pyo3)

#![allow(dead_code)]

mod model;
mod layers;
mod loss_function;
mod optimizer;


use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::model::{SimpleNN, OptimizerType};
use crate::layers::Activation;

/// Safe conversion helper
fn pylist_to_vec2(pylist: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f32>>> {
    pylist.iter()
        .map(|item| item.extract::<Vec<f32>>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyErr::from(e))
}

/// Python Class Wrapper
#[pyclass]
struct EtnaModel {
    inner: SimpleNN,
}

#[pymethods]
impl EtnaModel {
    #[new]
    #[pyo3(signature = (input_dim, hidden_layers, output_dim, task_type, activation=None))]
    fn new(
        input_dim: usize,
        hidden_layers: Vec<usize>,
        output_dim: usize,
        task_type: usize,
        activation: Option<String>,
    ) -> Self {
        // Parse activation string (default: ReLU)
        let act = match activation.as_deref().unwrap_or("relu") {
            "leaky_relu" => Activation::LeakyReLU,
            "sigmoid" => Activation::Sigmoid,
            _ => Activation::ReLU,
        };

        EtnaModel {
            inner: SimpleNN::new(input_dim, hidden_layers, output_dim, task_type, act),
        }
    }

    #[pyo3(signature = (x, y, epochs, lr, batch_size=32, weight_decay=0.0, optimizer="sgd", x_val=None, y_val=None))]
    fn train(
        &mut self,
        x: &Bound<'_, PyList>,
        y: &Bound<'_, PyList>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        weight_decay: f32,
        optimizer: &str,
        x_val: Option<&Bound<'_, PyList>>,
        y_val: Option<&Bound<'_, PyList>>,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let x_vec = pylist_to_vec2(x)?;
        let y_vec = pylist_to_vec2(y)?;

        // Parse optimizer string (default to SGD if not specified or invalid)
        let optimizer_type = match optimizer {
            "adam" => OptimizerType::Adam,
            _ => OptimizerType::SGD,  // Default to SGD for backward compatibility
        };

        // Convert optional validation data
        let x_val_opt = match x_val {
            Some(v) => Some(pylist_to_vec2(v)?),
            None => None,
        };
        let y_val_opt = match y_val {
            Some(v) => Some(pylist_to_vec2(v)?),
            None => None,
        };

        // Capture the history returned by Rust (both train and val losses)
        let (train_history, val_history) = self.inner.train(
            &x_vec, 
            &y_vec, 
            epochs, 
            lr, 
            weight_decay, 
            optimizer_type,
            batch_size,
            x_val_opt.as_ref(),
            y_val_opt.as_ref()
        );
        
        // Return both histories to Python
        Ok((train_history, val_history))
    }

    fn predict(&mut self, x: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
        let x_vec = pylist_to_vec2(x)?;
        Ok(self.inner.predict(&x_vec))
    }

    /// Get raw forward pass outputs (probabilities for classification, values for regression)
    /// This is useful for calculating validation loss
    fn forward(&mut self, x: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f32>>> {
        let x_vec = pylist_to_vec2(x)?;
        Ok(self.inner.forward(&x_vec))
    }

    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to save model: {}", e))
        })?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let inner = SimpleNN::load(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load model: {}", e))
        })?;
        Ok(Self { inner })
    }
}

#[pymodule]
fn _etna_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EtnaModel>()?;
    Ok(())
}

