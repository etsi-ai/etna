// Rust-Python bridge (pyo3)

#![allow(dead_code)]

mod model;
mod layers;
mod loss_function;
mod optimizer;
mod utils; 

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use pyo3::{Py, Python};
use crate::model::SimpleNN;

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
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, task_type: usize) -> Self {
        EtnaModel {
            inner: SimpleNN::new(input_dim, hidden_dim, output_dim, task_type),
        }
    }

    #[pyo3(signature = (x, y, epochs, lr, progress=None))]
    fn train(&mut self, x: &Bound<'_, PyList>, y: &Bound<'_, PyList>, epochs: usize, lr: f32, progress: Option<Py<PyAny>>) -> PyResult<Vec<f32>> {
        let x_vec = pylist_to_vec2(x);
        let y_vec = pylist_to_vec2(y);

        let history = if let Some(cb) = progress {
            let mut callback = move |epoch: usize, loss: f32| {
                Python::with_gil(|py| {
                    if let Err(err) = cb.call1(py, (epoch, loss)) {
                        err.print(py);
                    }
                });
            };

            self.inner.train_with_progress(&x_vec, &y_vec, epochs, lr, Some(&mut callback))
        } else {
            self.inner.train(&x_vec, &y_vec, epochs, lr)
        };

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
