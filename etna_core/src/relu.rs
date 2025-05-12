// ReLU Layer (implementing forward and backward)

pub struct ReLU;

impl ReLU {
    pub fn forward(input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        input.iter().map(|x| x.iter().map(|&v| v.max(0.0)).collect()).collect()
    }

    pub fn backward(grad_output: &Vec<Vec<f32>>, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        grad_output.iter().zip(input.iter())
            .map(|(grad, in_val)| grad.iter().zip(in_val.iter()).map(|(g, i)| if *i > 0.0 { *g } else { 0.0 }).collect())
            .collect()
    }
}
