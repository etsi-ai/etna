// Rust-Python bridge (pyo3)

mod model;
mod layers;
mod loss_function;
mod optimizer;

use pyo3::prelude::*;
use pyo3::Py;
use numpy::PyReadonlyArray2;

use crate::layers::Activation;
use crate::model::{OptimizerType, SimpleNN};

/// Zero-copy conversion: reads directly from NumPy's contiguous buffer
/// instead of iterating over Python list objects one element at a time.
fn ndarray_to_vec2(arr: PyReadonlyArray2<'_, f32>) -> Vec<Vec<f32>> {
    let array = arr.as_array();
    array
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
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

    #[pyo3(signature = (x, y, epochs, lr, batch_size=32, weight_decay=0.0, optimizer="sgd", early_stopping=false, patience=10, restore_best=true, progress_callback=None))]
    #[allow(clippy::too_many_arguments)]
    fn train(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f32>,
        y: PyReadonlyArray2<'_, f32>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        weight_decay: f32,
        optimizer: &str,
        early_stopping: bool,
        patience: usize,
        restore_best: bool,
        progress_callback: Option<Py<PyAny>>,
    ) -> PyResult<Vec<f32>> {
        let x_vec = ndarray_to_vec2(x);
        let y_vec = ndarray_to_vec2(y);
        let optimizer_type = match optimizer {
            "adam" => OptimizerType::Adam,
            _ => OptimizerType::Sgd,
        };

        // Create a closure that calls the Python callback if provided.
        // The callback is executed synchronously inside this function.
        let callback = |epoch: usize, total: usize, loss: f32| {
            if let Some(ref cb) = progress_callback {
                let _ = cb.call1(py, (epoch, total, loss));
            }
        };

        let history = self.inner.train_with_callback(
            &x_vec,
            &y_vec,
            epochs,
            lr,
            weight_decay,
            optimizer_type,
            batch_size,
            early_stopping,
            patience,
            restore_best,
            callback,
        );

        Ok(history)
    }

    fn predict(&mut self, x: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<f32>> {
        let x_vec = ndarray_to_vec2(x);
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
