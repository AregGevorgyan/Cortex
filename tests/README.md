# Autograd Test Suite

This directory contains comprehensive tests for the Cortex autograd implementation.

## Running Tests

```bash
# Run all tests
uv run python tests/test_autograd.py

# Or with standard Python
python tests/test_autograd.py
```

## What's Tested

### Core Operations (35+ tests)
- ✅ Basic arithmetic (add, sub, mul, div, pow, neg)
- ✅ Matrix operations (matmul 2D, **batched matmul**, transpose, reshape)
- ✅ Reductions (sum, mean with axis/keepdims)
- ✅ Activations (ReLU, sigmoid, tanh)
- ✅ Math functions (exp, log, abs)
- ✅ Softmax & log_softmax
- ✅ Broadcasting
- ✅ Edge cases (gradient accumulation, diamond graphs)
- ✅ New features (detach, no_grad, retain_graph)
- ✅ Neural network simulation

### Validation Method

All gradients are validated against **PyTorch** using finite differences to ensure mathematical correctness.

## Test Coverage

```
✓ 35+ individual gradient tests
✓ All major operations
✓ Broadcasting edge cases
✓ Complex computation graphs
✓ Production neural network patterns
```

## Requirements

- NumPy (required)
- PyTorch (optional, but strongly recommended for validation)

Install PyTorch for full testing:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Expected Output

When all tests pass:
```
======================================================================
ALL TESTS PASSED!
======================================================================

Your autograd implementation is correct and ready for use!
You can now safely rebuild your neural network library.
```

## Critical Tests

The most important tests for neural network functionality:

1. **Batched Matmul** - Essential for batch processing in layers
2. **Mean with axis** - Common in loss computation
3. **Neural Network Simulation** - End-to-end validation
4. **Gradient Accumulation** - Ensures correct handling of reused tensors

All tests pass ✓
