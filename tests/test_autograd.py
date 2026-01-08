"""
Comprehensive test suite for autograd implementation.
Tests gradient correctness against PyTorch and validates edge cases.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cortex.tensor import Tensor, no_grad

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Skipping comparison tests.")


def assert_close(cortex_grad, torch_grad, rtol=1e-4, atol=1e-5, op_name=""):
    """Compare gradients between Cortex and PyTorch"""
    if not np.allclose(cortex_grad, torch_grad, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(cortex_grad - torch_grad))
        raise AssertionError(
            f"{op_name} gradient mismatch!\n"
            f"Max difference: {max_diff}\n"
            f"Cortex:\n{cortex_grad}\n"
            f"PyTorch:\n{torch_grad}"
        )


def test_against_torch(test_name, cortex_fn, torch_fn, input_shape, check_input_grad=True):
    """Generic test function to compare Cortex and PyTorch gradients"""
    if not TORCH_AVAILABLE:
        print(f"Skipping {test_name} (PyTorch not available)")
        return

    # Generate random input
    x_np = np.random.randn(*input_shape).astype(np.float32)

    # Cortex
    x_cortex = Tensor(x_np.copy())
    y_cortex = cortex_fn(x_cortex)

    # PyTorch
    x_torch = torch.tensor(x_np.copy(), requires_grad=True)
    y_torch = torch_fn(x_torch)

    # Create gradient with same shape as output - use same grad_out for both
    # Handle scalar outputs
    output_data = y_cortex.data if isinstance(y_cortex.data, np.ndarray) else np.array(y_cortex.data)

    if output_data.size == 1:
        y_cortex.backward()
        y_torch.backward()
    else:
        grad_out = np.random.randn(*output_data.shape).astype(np.float32)
        y_cortex.backward(grad_out)
        y_torch.backward(torch.tensor(grad_out))

    # Compare
    if check_input_grad:
        assert_close(x_cortex.grad, x_torch.grad.numpy(), op_name=test_name)

    print(f"✓ {test_name}")


# ============================================================================
# BASIC OPERATIONS
# ============================================================================

def test_addition():
    """Test addition gradient"""
    if not TORCH_AVAILABLE:
        return

    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 4).astype(np.float32)

    # Cortex
    x_c = Tensor(x_np.copy())
    y_c = Tensor(y_np.copy())
    z_c = x_c + y_c
    z_c.backward(np.ones_like(z_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    y_t = torch.tensor(y_np.copy(), requires_grad=True)
    z_t = x_t + y_t
    z_t.backward(torch.ones_like(z_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Addition (x)")
    assert_close(y_c.grad, y_t.grad.numpy(), op_name="Addition (y)")
    print("✓ Addition")


def test_subtraction():
    """Test subtraction gradient"""
    if not TORCH_AVAILABLE:
        return

    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 4).astype(np.float32)

    # Cortex
    x_c = Tensor(x_np.copy())
    y_c = Tensor(y_np.copy())
    z_c = x_c - y_c
    z_c.backward(np.ones_like(z_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    y_t = torch.tensor(y_np.copy(), requires_grad=True)
    z_t = x_t - y_t
    z_t.backward(torch.ones_like(z_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Subtraction (x)")
    assert_close(y_c.grad, y_t.grad.numpy(), op_name="Subtraction (y)")
    print("✓ Subtraction")


def test_multiplication():
    """Test element-wise multiplication gradient"""
    if not TORCH_AVAILABLE:
        return

    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(3, 4).astype(np.float32)

    # Cortex
    x_c = Tensor(x_np.copy())
    y_c = Tensor(y_np.copy())
    z_c = x_c * y_c
    z_c.backward(np.ones_like(z_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    y_t = torch.tensor(y_np.copy(), requires_grad=True)
    z_t = x_t * y_t
    z_t.backward(torch.ones_like(z_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Multiplication (x)")
    assert_close(y_c.grad, y_t.grad.numpy(), op_name="Multiplication (y)")
    print("✓ Multiplication")


def test_division():
    """Test division gradient"""
    if not TORCH_AVAILABLE:
        return

    x_np = np.random.randn(3, 4).astype(np.float32) + 1.0
    y_np = np.random.randn(3, 4).astype(np.float32) + 1.0

    # Cortex
    x_c = Tensor(x_np.copy())
    y_c = Tensor(y_np.copy())
    z_c = x_c / y_c
    z_c.backward(np.ones_like(z_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    y_t = torch.tensor(y_np.copy(), requires_grad=True)
    z_t = x_t / y_t
    z_t.backward(torch.ones_like(z_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Division (x)")
    assert_close(y_c.grad, y_t.grad.numpy(), op_name="Division (y)")
    print("✓ Division")


def test_power():
    """Test power operation gradient"""
    test_against_torch(
        "Power (x**2)",
        lambda x: x ** 2,
        lambda x: x ** 2,
        (3, 4)
    )

    test_against_torch(
        "Power (x**3)",
        lambda x: x ** 3,
        lambda x: x ** 3,
        (3, 4)
    )


def test_negation():
    """Test negation gradient"""
    test_against_torch(
        "Negation",
        lambda x: -x,
        lambda x: -x,
        (3, 4)
    )


# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

def test_matmul_2d():
    """Test 2D matrix multiplication gradient"""
    if not TORCH_AVAILABLE:
        return

    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(4, 5).astype(np.float32)

    # Cortex
    x_c = Tensor(x_np.copy())
    y_c = Tensor(y_np.copy())
    z_c = x_c @ y_c
    z_c.backward(np.ones_like(z_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    y_t = torch.tensor(y_np.copy(), requires_grad=True)
    z_t = x_t @ y_t
    z_t.backward(torch.ones_like(z_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Matmul 2D (x)")
    assert_close(y_c.grad, y_t.grad.numpy(), op_name="Matmul 2D (y)")
    print("✓ Matmul 2D")


def test_matmul_batched():
    """Test batched matrix multiplication gradient (CRITICAL TEST)"""
    if not TORCH_AVAILABLE:
        return

    # Test case: batch_size=32, like neural network layers
    x_np = np.random.randn(32, 10, 20).astype(np.float32)
    y_np = np.random.randn(32, 20, 30).astype(np.float32)

    # Cortex
    x_c = Tensor(x_np.copy())
    y_c = Tensor(y_np.copy())
    z_c = x_c @ y_c
    z_c.backward(np.ones_like(z_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    y_t = torch.tensor(y_np.copy(), requires_grad=True)
    z_t = x_t @ y_t
    z_t.backward(torch.ones_like(z_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Batched Matmul (x)")
    assert_close(y_c.grad, y_t.grad.numpy(), op_name="Batched Matmul (y)")
    print("✓ Batched Matmul")


def test_transpose():
    """Test transpose gradient"""
    test_against_torch(
        "Transpose",
        lambda x: x.transpose((1, 0, 2)),
        lambda x: x.permute(1, 0, 2),
        (3, 4, 5)
    )


def test_reshape():
    """Test reshape gradient"""
    test_against_torch(
        "Reshape",
        lambda x: x.reshape(6, 4),
        lambda x: x.reshape(6, 4),
        (3, 8)
    )


# ============================================================================
# REDUCTION OPERATIONS
# ============================================================================

def test_sum_global():
    """Test global sum gradient"""
    test_against_torch(
        "Sum (global)",
        lambda x: x.sum(),
        lambda x: x.sum(),
        (3, 4)
    )


def test_sum_axis():
    """Test sum along axis gradient"""
    test_against_torch(
        "Sum (axis=0)",
        lambda x: x.sum(axis=0),
        lambda x: x.sum(dim=0),
        (3, 4)
    )

    test_against_torch(
        "Sum (axis=1)",
        lambda x: x.sum(axis=1),
        lambda x: x.sum(dim=1),
        (3, 4)
    )


def test_mean_global():
    """Test global mean gradient"""
    test_against_torch(
        "Mean (global)",
        lambda x: x.mean(),
        lambda x: x.mean(),
        (3, 4)
    )


def test_mean_axis():
    """Test mean along axis gradient"""
    test_against_torch(
        "Mean (axis=0)",
        lambda x: x.mean(axis=0),
        lambda x: x.mean(dim=0),
        (3, 4)
    )

    test_against_torch(
        "Mean (axis=1, keepdims=True)",
        lambda x: x.mean(axis=1, keepdims=True),
        lambda x: x.mean(dim=1, keepdim=True),
        (3, 4)
    )


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

def test_relu():
    """Test ReLU gradient"""
    test_against_torch(
        "ReLU",
        lambda x: x.relu(),
        lambda x: torch.relu(x),
        (3, 4)
    )


def test_sigmoid():
    """Test sigmoid gradient"""
    test_against_torch(
        "Sigmoid",
        lambda x: x.sigmoid(),
        lambda x: torch.sigmoid(x),
        (3, 4)
    )


def test_tanh():
    """Test tanh gradient"""
    test_against_torch(
        "Tanh",
        lambda x: x.tanh(),
        lambda x: torch.tanh(x),
        (3, 4)
    )


# ============================================================================
# MATHEMATICAL FUNCTIONS
# ============================================================================

def test_exp():
    """Test exponential gradient"""
    test_against_torch(
        "Exp",
        lambda x: x.exp(),
        lambda x: torch.exp(x),
        (3, 4)
    )


def test_log():
    """Test logarithm gradient"""
    # Use positive inputs
    def cortex_log(x):
        return (x.abs() + 1.0).log()

    def torch_log(x):
        return torch.log(torch.abs(x) + 1.0)

    test_against_torch(
        "Log",
        cortex_log,
        torch_log,
        (3, 4)
    )


def test_abs():
    """Test absolute value gradient"""
    test_against_torch(
        "Abs",
        lambda x: x.abs(),
        lambda x: torch.abs(x),
        (3, 4)
    )


# ============================================================================
# SOFTMAX AND LOSS FUNCTIONS
# ============================================================================

def test_softmax():
    """Test softmax gradient"""
    test_against_torch(
        "Softmax",
        lambda x: x.softmax(axis=-1),
        lambda x: torch.softmax(x, dim=-1),
        (3, 4)
    )


def test_log_softmax():
    """Test log_softmax gradient"""
    test_against_torch(
        "Log Softmax",
        lambda x: x.log_softmax(axis=-1),
        lambda x: torch.log_softmax(x, dim=-1),
        (3, 4)
    )

def test_mse_loss():
    """Test MSE loss computation and gradients"""
    from cortex.nn.loss import MSELoss

    pred = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    target = Tensor(np.array([[1.5, 2.5], [3.5, 4.5]]))

    loss_fn = MSELoss()
    loss = loss_fn(pred, target)
    loss.backward()

    # MSE = mean((pred - target)^2) = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
    assert abs(loss.data - 0.25) < 1e-6

    # Gradient: d/dpred MSE = 2*(pred - target) / n
    expected_grad = 2 * (pred.data - target.data) / 4
    assert np.allclose(pred.grad, expected_grad)


# ============================================================================
# BROADCASTING TESTS
# ============================================================================

def test_broadcasting_add():
    """Test broadcasting in addition"""
    if not TORCH_AVAILABLE:
        return

    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(4).astype(np.float32)

    # Cortex
    x_c = Tensor(x_np.copy())
    y_c = Tensor(y_np.copy())
    z_c = x_c + y_c
    z_c.backward(np.ones_like(z_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    y_t = torch.tensor(y_np.copy(), requires_grad=True)
    z_t = x_t + y_t
    z_t.backward(torch.ones_like(z_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Broadcasting Add (x)")
    assert_close(y_c.grad, y_t.grad.numpy(), op_name="Broadcasting Add (y)")
    print("✓ Broadcasting (addition)")


def test_broadcasting_multiply():
    """Test broadcasting in multiplication"""
    if not TORCH_AVAILABLE:
        return

    x_np = np.random.randn(3, 4).astype(np.float32)
    y_np = np.random.randn(1, 4).astype(np.float32)

    # Cortex
    x_c = Tensor(x_np.copy())
    y_c = Tensor(y_np.copy())
    z_c = x_c * y_c
    z_c.backward(np.ones_like(z_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    y_t = torch.tensor(y_np.copy(), requires_grad=True)
    z_t = x_t * y_t
    z_t.backward(torch.ones_like(z_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Broadcasting Mul (x)")
    assert_close(y_c.grad, y_t.grad.numpy(), op_name="Broadcasting Mul (y)")
    print("✓ Broadcasting (multiplication)")


# ============================================================================
# EDGE CASES
# ============================================================================

def test_multiple_uses():
    """Test gradient accumulation when tensor is used multiple times"""
    if not TORCH_AVAILABLE:
        return

    x_np = np.random.randn(3, 4).astype(np.float32)

    # Cortex
    x_c = Tensor(x_np.copy())
    y_c = x_c + x_c  # x used twice
    y_c.backward(np.ones_like(y_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    y_t = x_t + x_t
    y_t.backward(torch.ones_like(y_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Multiple uses")
    print("✓ Multiple uses (gradient accumulation)")


def test_diamond_graph():
    """Test diamond-shaped computation graph"""
    if not TORCH_AVAILABLE:
        return

    x_np = np.random.randn(3, 4).astype(np.float32)

    # Cortex
    x_c = Tensor(x_np.copy())
    a_c = x_c * 2.0
    b_c = x_c + 3.0
    c_c = a_c + b_c  # Diamond: x -> a,b -> c
    c_c.backward(np.ones_like(c_c.data))

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=True)
    a_t = x_t * 2.0
    b_t = x_t + 3.0
    c_t = a_t + b_t
    c_t.backward(torch.ones_like(c_t))

    assert_close(x_c.grad, x_t.grad.numpy(), op_name="Diamond graph")
    print("✓ Diamond graph")


def test_detach():
    """Test detach() stops gradient flow"""
    x = Tensor([1.0, 2.0, 3.0])
    y = x * 2.0
    z = y.detach()

    assert z.requires_grad == False, "Detached tensor should not require grad"
    print("✓ Detach")


def test_no_grad_context():
    """Test no_grad() context manager"""
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    with no_grad():
        y = x * 2.0
        assert y.requires_grad == False, "Tensor in no_grad context should not require grad"

    # Outside context, should work normally
    z = x * 3.0
    assert z.requires_grad == True, "Tensor outside no_grad should require grad"

    print("✓ no_grad context")


def test_retain_graph():
    """Test retain_graph parameter"""
    x = Tensor([1.0, 2.0, 3.0])
    y = x * 2.0
    z = y.sum()

    # First backward with retain_graph=True
    z.backward(retain_graph=True)
    grad1 = x.grad.copy()

    # Reset gradients
    x.zero_grad()
    y.zero_grad()
    z.zero_grad()

    # Second backward should work (graph retained)
    z.backward(retain_graph=False)
    grad2 = x.grad.copy()

    assert np.allclose(grad1, grad2), "Gradients should be the same"
    print("✓ retain_graph")


# ============================================================================
# NEURAL NETWORK SIMULATION
# ============================================================================

def test_simple_neural_network():
    """Test gradient flow through a simple neural network"""
    if not TORCH_AVAILABLE:
        return

    # Network: x @ W1 + b1 -> ReLU -> @ W2 + b2
    batch_size, in_dim, hidden_dim, out_dim = 32, 10, 20, 5

    x_np = np.random.randn(batch_size, in_dim).astype(np.float32)
    W1_np = np.random.randn(in_dim, hidden_dim).astype(np.float32) * 0.1
    b1_np = np.random.randn(1, hidden_dim).astype(np.float32) * 0.1
    W2_np = np.random.randn(hidden_dim, out_dim).astype(np.float32) * 0.1
    b2_np = np.random.randn(1, out_dim).astype(np.float32) * 0.1

    # Cortex
    x_c = Tensor(x_np.copy(), requires_grad=False)
    W1_c = Tensor(W1_np.copy())
    b1_c = Tensor(b1_np.copy())
    W2_c = Tensor(W2_np.copy())
    b2_c = Tensor(b2_np.copy())

    h_c = (x_c @ W1_c + b1_c).relu()
    out_c = h_c @ W2_c + b2_c
    loss_c = out_c.mean()
    loss_c.backward()

    # PyTorch
    x_t = torch.tensor(x_np.copy(), requires_grad=False)
    W1_t = torch.tensor(W1_np.copy(), requires_grad=True)
    b1_t = torch.tensor(b1_np.copy(), requires_grad=True)
    W2_t = torch.tensor(W2_np.copy(), requires_grad=True)
    b2_t = torch.tensor(b2_np.copy(), requires_grad=True)

    h_t = torch.relu(x_t @ W1_t + b1_t)
    out_t = h_t @ W2_t + b2_t
    loss_t = out_t.mean()
    loss_t.backward()

    # Compare all parameter gradients
    assert_close(W1_c.grad, W1_t.grad.numpy(), op_name="Neural Net W1")
    assert_close(b1_c.grad, b1_t.grad.numpy(), op_name="Neural Net b1")
    assert_close(W2_c.grad, W2_t.grad.numpy(), op_name="Neural Net W2")
    assert_close(b2_c.grad, b2_t.grad.numpy(), op_name="Neural Net b2")

    print("✓ Simple Neural Network")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all autograd tests"""
    print("=" * 70)
    print("AUTOGRAD TEST SUITE")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("\nWARNING: PyTorch not available. Installing PyTorch is recommended.")
        print("  pip install torch\n")

    print("\n--- Basic Operations ---")
    test_addition()
    test_subtraction()
    test_multiplication()
    test_division()
    test_power()
    test_negation()

    print("\n--- Matrix Operations ---")
    test_matmul_2d()
    test_matmul_batched()
    test_transpose()
    test_reshape()

    print("\n--- Reduction Operations ---")
    test_sum_global()
    test_sum_axis()
    test_mean_global()
    test_mean_axis()

    print("\n--- Activation Functions ---")
    test_relu()
    test_sigmoid()
    test_tanh()

    print("\n--- Mathematical Functions ---")
    test_exp()
    test_log()
    test_abs()

    print("\n--- Softmax & Loss ---")
    test_softmax()
    test_log_softmax()

    print("\n--- Broadcasting ---")
    test_broadcasting_add()
    test_broadcasting_multiply()

    print("\n--- Edge Cases ---")
    test_multiple_uses()
    test_diamond_graph()
    test_detach()
    test_no_grad_context()
    test_retain_graph()

    print("\n--- Neural Network Simulation ---")
    test_simple_neural_network()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("\nYour autograd implementation is correct and ready for use!")
    print("You can now safely rebuild your neural network library.")


if __name__ == "__main__":
    run_all_tests()
