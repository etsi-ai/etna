// # Layers (Linear, ReLU, Softmax, etc.)


use rand::Rng;

/// Fully connected layer: y = Wx + b
pub struct Linear {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<Vec<f32>>, // shape: [output_size][input_size]
    pub bias: Vec<f32>,         // shape: [output_size]
}

impl Linear {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-0.01..0.01)) // Random init
                    .collect()
            })
            .collect();

        let bias = vec![0.0; output_size];

        Linear {
            input_size,
            output_size,
            weights,
            bias,
        }
    }

    /// Forward pass: output = weights * input + bias
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = vec![0.0; self.output_size];

        for i in 0..self.output_size {
            for j in 0..self.input_size {
                output[i] += self.weights[i][j] * input[j];
            }
            output[i] += self.bias[i];
        }

        output
    }
}

/// ReLU activation function
pub struct ReLU;

impl ReLU {
    pub fn forward(input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|x| x.max(0.0)).collect()
    }
}

/// Softmax activation function (output layer)
pub struct Softmax;

impl Softmax {
    pub fn forward(input: &Vec<f32>) -> Vec<f32> {
        let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.iter().map(|x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();

        exps.iter().map(|x| x / sum).collect()
    }
}
