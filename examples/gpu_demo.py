"""
GPU Backend Demo for Cortex

This script demonstrates how to use GPU acceleration with Cortex via CuPy.
"""

import cortex
import numpy as np


def demo_backend_switching():
    """Demonstrate switching between CPU and GPU backends."""
    print("=" * 70)
    print("DEMO 1: Backend Switching")
    print("=" * 70)

    # Default backend is NumPy (CPU)
    print(f"\n1. Default backend: {cortex.get_backend_name()}")

    x = cortex.Tensor([1.0, 2.0, 3.0])
    print(f"   Tensor created on {cortex.get_backend_name()}: {x.data}")

    # Try switching to GPU
    try:
        cortex.set_backend('gpu')
        print(f"\n2. Switched to GPU backend: {cortex.get_backend_name()}")

        y = cortex.Tensor([4.0, 5.0, 6.0])
        print(f"   Tensor created on {cortex.get_backend_name()}: {y.data}")

        # Switch back to CPU
        cortex.set_backend('cpu')
        print(f"\n3. Switched back to CPU: {cortex.get_backend_name()}")

    except ImportError as e:
        print(f"\n2. GPU backend not available (CuPy not installed)")
        print(f"   Install with: pip install cupy-cuda12x")
        print(f"   Continuing with CPU backend...")


def demo_autograd_on_gpu():
    """Demonstrate autograd working on GPU."""
    print("\n" + "=" * 70)
    print("DEMO 2: Autograd on GPU")
    print("=" * 70)

    try:
        cortex.set_backend('gpu')
        print(f"\nBackend: {cortex.get_backend_name()}")
    except ImportError:
        cortex.set_backend('cpu')
        print(f"\nBackend: {cortex.get_backend_name()} (GPU not available)")

    # Create tensors
    x = cortex.Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = cortex.Tensor([[5.0, 6.0], [7.0, 8.0]])

    # Forward pass
    z = (x @ y) + 10.0
    result = z.sum()

    print(f"\nForward pass result: {result.data}")

    # Backward pass
    result.backward()

    print(f"Gradients computed:")
    print(f"  x.grad = \n{x.grad}")
    print(f"  y.grad = \n{y.grad}")


def demo_neural_network():
    """Demonstrate neural network training on GPU."""
    print("\n" + "=" * 70)
    print("DEMO 3: Neural Network on GPU")
    print("=" * 70)

    try:
        cortex.set_backend('gpu')
        print(f"\nBackend: {cortex.get_backend_name()}")
    except ImportError:
        cortex.set_backend('cpu')
        print(f"\nBackend: {cortex.get_backend_name()} (GPU not available)")

    # Create simple dataset
    np.random.seed(42)
    X = cortex.Tensor(np.random.randn(100, 10).astype(np.float32))
    y = cortex.Tensor(np.random.randn(100, 1).astype(np.float32))

    # Create model using autograd tensors
    W1 = cortex.Tensor(np.random.randn(10, 5).astype(np.float32) * 0.01)
    b1 = cortex.Tensor(np.zeros((1, 5)).astype(np.float32))
    W2 = cortex.Tensor(np.random.randn(5, 1).astype(np.float32) * 0.01)
    b2 = cortex.Tensor(np.zeros((1, 1)).astype(np.float32))

    print(f"\nTraining simple network (100 samples, 10 features -> 1 output)")

    # Training loop
    learning_rate = 0.01
    for epoch in range(5):
        # Forward pass
        h = (X @ W1 + b1).relu()
        y_pred = h @ W2 + b2

        # Compute loss (MSE)
        loss = ((y_pred - y) ** 2).mean()

        # Backward pass
        W1.zero_grad()
        b1.zero_grad()
        W2.zero_grad()
        b2.zero_grad()
        loss.backward(retain_graph=True)

        # Update weights (manual gradient descent)
        W1.data = W1.data - learning_rate * W1.grad
        b1.data = b1.data - learning_rate * b1.grad
        W2.data = W2.data - learning_rate * W2.grad
        b2.data = b2.data - learning_rate * b2.grad

        print(f"  Epoch {epoch + 1}, Loss: {float(loss.data):.6f}")


def demo_data_movement():
    """Demonstrate moving data between CPU and GPU."""
    print("\n" + "=" * 70)
    print("DEMO 4: Data Movement (CPU ↔ GPU)")
    print("=" * 70)

    try:
        # Create data on GPU
        cortex.set_backend('gpu')
        print(f"\n1. Created tensor on GPU")
        x_gpu = cortex.Tensor([1, 2, 3, 4, 5])
        print(f"   Type: {type(x_gpu.data)}")

        # Move to CPU
        x_cpu = cortex.to_numpy(x_gpu.data)
        print(f"\n2. Moved to CPU using to_numpy()")
        print(f"   Type: {type(x_cpu)}")
        print(f"   Data: {x_cpu}")

        # Move back to GPU
        x_back_to_gpu = cortex.to_backend(x_cpu)
        print(f"\n3. Moved back to GPU using to_backend()")
        print(f"   Type: {type(x_back_to_gpu)}")

    except ImportError:
        print("\nGPU not available. Using CPU only.")
        cortex.set_backend('cpu')
        x = cortex.Tensor([1, 2, 3, 4, 5])
        print(f"Tensor type on CPU: {type(x.data)}")


def demo_performance_comparison():
    """Compare CPU vs GPU performance (if GPU available)."""
    print("\n" + "=" * 70)
    print("DEMO 5: Performance Comparison")
    print("=" * 70)

    import time

    size = 1000
    iterations = 100

    # Test on CPU
    cortex.set_backend('cpu')
    print(f"\nTesting on CPU ({iterations} matrix multiplications of {size}x{size})...")

    x_cpu = cortex.Tensor(np.random.randn(size, size).astype(np.float32))
    y_cpu = cortex.Tensor(np.random.randn(size, size).astype(np.float32))

    start = time.time()
    for _ in range(iterations):
        z_cpu = x_cpu @ y_cpu
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.4f} seconds")

    # Test on GPU
    try:
        cortex.set_backend('gpu')
        print(f"\nTesting on GPU ({iterations} matrix multiplications of {size}x{size})...")

        x_gpu = cortex.Tensor(np.random.randn(size, size).astype(np.float32))
        y_gpu = cortex.Tensor(np.random.randn(size, size).astype(np.float32))

        start = time.time()
        for _ in range(iterations):
            z_gpu = x_gpu @ y_gpu
        gpu_time = time.time() - start
        print(f"GPU time: {gpu_time:.4f} seconds")

        speedup = cpu_time / gpu_time
        print(f"\nSpeedup: {speedup:.2f}x faster on GPU!")

    except ImportError:
        print("\nGPU not available for comparison.")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("CORTEX GPU BACKEND DEMONSTRATION")
    print("=" * 70)

    demo_backend_switching()
    demo_autograd_on_gpu()
    demo_neural_network()
    demo_data_movement()
    demo_performance_comparison()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
    print("\nTo enable GPU support, install CuPy:")
    print("  pip install cupy-cuda12x  # For CUDA 12.x")
    print("  pip install cupy-cuda11x  # For CUDA 11.x")
    print("\nSee GPU_SUPPORT.md for more information.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
