// Rust-Python bridge (pyo3)

mod model;
mod layers;
mod loss_function;
mod optimizer;

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::layers::Activation;
use crate::model::SimpleNN;
use crate::model::OptimizerType;

/// Safe conversion helper
fn pylist_to_vec2(pylist: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f32>>> {
    pylist.iter()
        .map(|item| item.extract::<Vec<f32>>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e)
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
        let act = match activation.as_deref().unwrap_or("relu") {
            "leaky_relu" => Activation::LeakyReLU,
            "sigmoid" => Activation::Sigmoid,
            _ => Activation::ReLU,
        };

        EtnaModel {
            inner: SimpleNN::new(input_dim, hidden_layers, output_dim, task_type, act),
        }
    }

    #[pyo3(signature = (x, y, epochs, lr, batch_size=32, weight_decay=0.0, optimizer="sgd"))]
    fn train(
        &mut self,
        x: &Bound<'_, PyList>,
        y: &Bound<'_, PyList>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        weight_decay: f32,
        optimizer: &str,
    ) -> PyResult<Vec<f32>> {
        let x_vec = pylist_to_vec2(x)?;
        let y_vec = pylist_to_vec2(y)?;
        let optimizer_type = match optimizer {
            "adam" => OptimizerType::Adam,
            _ => OptimizerType::SGD,
        };

    // Capture the history returned by Rust
    let history = self.inner.train(
        &x_vec,
        &y_vec,
        epochs,
        lr,
        weight_decay,
        optimizer_type,
        batch_size,
    );

        // Return it to Python
        Ok(history)
    }

    fn predict(&mut self, x: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
        let x_vec = pylist_to_vec2(x)?;
        Ok(self.inner.predict(&x_vec))
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
