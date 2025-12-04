// Rust-Python bridge (pyo3)

mod model;
mod layers;
mod loss_function;
mod optimizer;
mod utils; 

use pyo3::prelude::*;
use pyo3::types::PyList;
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

    fn train(&mut self, x: &Bound<'_, PyList>, y: &Bound<'_, PyList>, epochs: usize, lr: f32) -> PyResult<()> {
        let x_vec = pylist_to_vec2(x);
        let y_vec = pylist_to_vec2(y);
        self.inner.train(&x_vec, &y_vec, epochs, lr);
        Ok(())
    }

    fn predict(&mut self, x: &Bound<'_, PyList>) -> PyResult<Vec<f32>> {
        let x_vec = pylist_to_vec2(x);
        let preds = self.inner.predict(&x_vec);
        Ok(preds)
    }
}

#[pymodule]
fn _etna_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EtnaModel>()?;
    Ok(())
}
