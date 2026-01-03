// Rust-Python bridge (pyo3)

#![allow(dead_code)]

mod model;
mod layers;
mod loss_function;
mod optimizer;
mod utils; 

use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::model::SimpleNN;
use crate::layers::Activation;

/// Helper: Convert Python list to Rust Vec
fn pylist_to_vec2(pylist: &Bound<'_, PyList>) -> Vec<Vec<f32>> {
    pylist.iter()
        .map(|item| item.extract::<Vec<f32>>().expect("Expected list of floats"))
        .collect()
}

/// Python Class Wrapper
#[pyclass]
struct EtnaModel {
    inner: SimpleNN,
}

#[pymethods]
impl EtnaModel {
    #[new]
    #[pyo3(signature = (input_dim, hidden_dim, output_dim, task_type, activation=None))]
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, task_type: usize, activation: Option<String>) -> Self {
        // Parse activation string, default to ReLU
        let act = match activation.as_deref().unwrap_or("relu") {
            "leaky_relu" => Activation::LeakyReLU,
            "sigmoid" => Activation::Sigmoid,
            _ => Activation::ReLU,
        };
        
        EtnaModel {
            inner: SimpleNN::new(input_dim, hidden_dim, output_dim, task_type, act),
        }
    }

    fn train(&mut self, x: &Bound<'_, PyList>, y: &Bound<'_, PyList>, epochs: usize, lr: f32) -> PyResult<Vec<f32>> {
        let x_vec = pylist_to_vec2(x);
        let y_vec = pylist_to_vec2(y);
        
        // Capture the history returned by Rust
        let history = self.inner.train(&x_vec, &y_vec, epochs, lr);
        
        // Return it to Python
        Ok(history)
    }

    fn predict(&mut self, x: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
        let x_vec = pylist_to_vec2(x);
        let preds = self.inner.predict(&x_vec);
        Ok(preds)
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
        Ok(EtnaModel { inner })
    }
}

#[pymodule]
fn _etna_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EtnaModel>()?;
    Ok(())
}
