// Rust-Python bridge (pyo3)


mod model;
mod layers;
mod loss_function;
mod optimizer;


use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

use crate::model::SimpleNN;

/// Convert Python list of lists to Rust Vec<Vec<f32>>
fn pylist_to_vec2(pylist: &PyList) -> Vec<Vec<f32>> {
    pylist.iter()
        .map(|row| row.extract::<Vec<f32>>().expect("Expected list of floats"))
        .collect()
}

/// Convert target indices to one-hot vectors
fn one_hot_encode(targets: Vec<usize>, num_classes: usize) -> Vec<Vec<f32>> {
    targets.into_iter().map(|i| {
        let mut row = vec![0.0; num_classes];
        row[i] = 1.0;
        row
    }).collect()
}

/// Train the neural network
#[pyfunction]
fn train(x: &PyList, y: &PyList, epochs: usize, lr: f32) -> PyResult<()> {
    let x_vec = pylist_to_vec2(x);
    let y_raw: Vec<usize> = y.extract()?; // e.g., [0, 1, 1, 0]
    let num_classes = *y_raw.iter().max().unwrap_or(&0) + 1;
    let y_vec = one_hot_encode(y_raw, num_classes);

    let input_dim = x_vec[0].len();
    let hidden_dim = 16;
    let output_dim = num_classes;

    let mut model = SimpleNN::new(input_dim, hidden_dim, output_dim);
    model.train(&x_vec, &y_vec, epochs, lr);
    Ok(())
}

/// Predict labels from input
#[pyfunction]
fn predict(x: &PyList) -> PyResult<Vec<usize>> {
    let x_vec = pylist_to_vec2(x);
    let input_dim = x_vec[0].len();
    let hidden_dim = 16;
    let output_dim = 2; // Default output class count

    let mut model = SimpleNN::new(input_dim, hidden_dim, output_dim);
    let preds = model.predict(&x_vec);
    Ok(preds)
}

/// Register functions to the module
#[pymodule]
fn _etna_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    Ok(())
}
