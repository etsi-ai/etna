# [Core] Add LeakyReLU & Sigmoid Activations

## Description

Added two new activation functions to make the library more versatile beyond standard ReLU.

## Changes

### New Activations in `etna_core/src/layers.rs`

**LeakyReLU:**
- Forward: `max(0.01 * x, x)` - allows small gradients for negative values
- Backward: derivative is `1` for positive inputs, `0.01` for negative

**Sigmoid:**
- Forward: `1 / (1 + e^(-x))` - squashes output to (0, 1) range
- Backward: `sigmoid(x) * (1 - sigmoid(x))` using cached output for efficiency

### Updated `etna_core/src/model.rs`

Registered new activations in imports for model usage.

## Tests Added

Added 12 unit tests in `layers.rs`:

| Test | Description |
|------|-------------|
| `test_leaky_relu_forward_positive` | Positive values pass through unchanged |
| `test_leaky_relu_forward_negative` | Negative values scaled by 0.01 |
| `test_leaky_relu_forward_mixed` | Mixed positive/negative/zero inputs |
| `test_leaky_relu_backward_positive` | Gradient = 1 for positive inputs |
| `test_leaky_relu_backward_negative` | Gradient = 0.01 for negative inputs |
| `test_sigmoid_forward` | sigmoid(0) = 0.5 |
| `test_sigmoid_forward_large_positive` | sigmoid(100) ≈ 1.0 |
| `test_sigmoid_forward_large_negative` | sigmoid(-100) ≈ 0.0 |
| `test_sigmoid_backward` | Derivative at x=0 is 0.25 |
| `test_sigmoid_backward_near_saturation` | Gradient vanishes near saturation |
| `test_relu_forward` | Coverage for existing ReLU |
| `test_relu_backward` | Coverage for existing ReLU |

## Verification

```
cargo test
running 12 tests
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Checklist

- [x] New activation structs exist in `layers.rs`
- [x] Forward and backward methods implemented correctly
- [x] Activations registered in `model.rs`
- [x] Unit tests added and passing
