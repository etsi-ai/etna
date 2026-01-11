// Linear Layer (implementing forward, backward, and update)
#[derive(Clone)]
pub struct Linear {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    input_size: usize,
    output_size: usize,
    grad_weights: Vec<Vec<f32>>,  // Gradient for weights
    grad_bias: Vec<f32>,          // Gradient for biases
    cached_input: Vec<Vec<f32>>,  // Cache input for backward pass
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = vec![vec![0.0; input_size]; output_size];
        let bias = vec![0.0; output_size];
        let grad_weights = vec![vec![0.0; input_size]; output_size];
        let grad_bias = vec![0.0; output_size];
        let cached_input = vec![];

        Linear { weights, bias, input_size, output_size, grad_weights, grad_bias, cached_input }
    }

    pub fn forward(&mut self, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        self.cached_input = input.clone(); // Cache input for backpropagation
        input.iter().map(|x| {
            self.weights.iter()
                .map(|w| 
                    w.iter()
                    .zip(x.iter())
                    .map(|(w_val, x_val)| w_val * x_val)
                    .sum::<f32>() 
                    + self.bias[j]
                )
                .collect::<Vec<f32>>()
        }).collect()
    }

    pub fn backward(&mut self, grad_output: &Vec<Vec<f32>>, input: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let mut grad_input = vec![vec![0.0; self.input_size]; input.len()];

        // Compute gradients for weights and biases
        for (i, grad) in grad_output.iter().enumerate() {
            for (j, grad_val) in grad.iter().enumerate() {
                self.grad_bias[j] += grad_val; // Sum gradients for bias
                for (k, &input_val) in input[i].iter().enumerate() {
                    self.grad_weights[j][k] += grad_val * input_val; // Gradient for weights
                    grad_input[i][k] += grad_val * self.weights[j][k]; // Gradient for input
                }
            }
        }
        grad_input
    }

    pub fn update(&mut self, optimizer: &mut SGD) {
        for i in 0..self.output_size {
            for j in 0..self.input_size {
                self.weights[i][j] -= optimizer.learning_rate * self.grad_weights[i][j];
                self.grad_weights[i][j] = 0.0;
            }
            self.bias[i] -= optimizer.learning_rate * self.grad_bias[i];
            self.grad_bias[i] = 0.0;
        }
    }
}
