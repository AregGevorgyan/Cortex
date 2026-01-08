# Cortex
Deep learning framework from scratch in Python. It includes

- Tensor library with autograd
- Neural network module with 

Let me know if you have any suggestions or if you found this useful.

This project won an award at the Startup Shell Hackathon in fall 2025!

# Warning: This is a work in progress, first relese coming soon

# Todo

Backend support:
- [ ] Multi-GPU support
- [ ] Automatic device selection based on data size
- [ ] Memory profiling utilities
- [ ] Mixed precision (FP16/BF16)
- [ ] `Tensor.to(device)` method

# Some future directions
- Forward mode autodiff
- Higher order derivatives
- Lower to an internal IR for optimizations and lower that to device IR (LLVM, CUDA, etc)
- Stop relying on NumPy/CuPy and build my own backends

# Some other cool projects
- https://github.com/MarioSieg/magnetron
- https://github.com/karpathy/micrograd
- https://github.com/tinygrad/tinygrad
- https://github.com/tanishqkumar/beyond-nanogpt
