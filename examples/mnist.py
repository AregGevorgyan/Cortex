"""
Complete MNIST training example using modern Cortex API
"""
import numpy as np
from cortex import Tensor, set_backend
from cortex.nn import Linear, Sequential, ReLU, Dropout
from cortex.nn.loss import CrossEntropyLoss
from cortex.nn.training import DataLoader, fit
from cortex.core import Optimizer

# Try to use GPU if available
try:
    set_backend('gpu')
    print("Using GPU backend")
except:
    set_backend('cpu')
    print("Using CPU backend")

# Load MNIST data (you'll need to download this)
# For offline work, include data loading code
def load_mnist():
    """Load MNIST from numpy files"""
    # Assumes you have X_train.npy, y_train.npy, etc.
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    # Normalize
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    return X_train, y_train, X_test, y_test

# Build model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 10)
)

# Loss and optimizer
loss_fn = CrossEntropyLoss()
optimizer = Optimizer(lr=0.001, algorithm='adam')

# Data
X_train, y_train, X_test, y_test = load_mnist()
train_loader = DataLoader(X_train, y_train, batch_size=32)
test_loader = DataLoader(X_test, y_test, batch_size=32, shuffle=False)

# Train
history = fit(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    n_epochs=10,
    verbose=True
)

# Final evaluation
test_loss = eval(model, test_loader, loss_fn)
print(f"\nFinal test loss: {test_loss:.4f}")
