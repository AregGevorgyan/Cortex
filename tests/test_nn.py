from cortex.nn import Linear, Sequential
from cortex import Tensor

# Build a 2-layer network
model = Sequential(
    Linear(784, 128),
    lambda x: x.relu(),  # Activation
    Linear(128, 10)
)

# Forward pass
x = Tensor(Tensor.random.randn(32, 784))
y = model(x)

# Backward pass
loss = y.mean()
loss.backward()

# Check gradients
for param in model.parameters():
    print(param.grad.shape)