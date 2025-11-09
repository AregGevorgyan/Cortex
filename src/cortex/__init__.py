__version__ = "0.1.0"
__author__ = "Areg Gevorgyan"

# Backend abstraction for CPU/GPU support
from .backend import (
    set_backend,
    get_backend,
    get_backend_name,
    to_numpy,
    to_backend
)

# Import Tensor and autograd functionality
from .tensor import (
    Tensor,
    no_grad
)

from .core import (
    NeuralNetwork,
    Optimizer
)

from .layers import (
    Layer,
    Dense,
    Convolutional,
    Pooling,
    Recurrent,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax
)

from .loss import (
    Loss,
    CrossEntropyLoss,
    MSELoss
)

from .utils import Utils

# Make backend module accessible
from . import backend

__all__ = [
    # Backend
    "set_backend",
    "get_backend",
    "get_backend_name",
    "to_numpy",
    "to_backend",
    "backend",
    # Tensor and autograd
    "Tensor",
    "no_grad",
    # Core
    "NeuralNetwork",
    "Optimizer",
    "Utils",
    # Layers
    "Layer",
    "Dense",
    "Convolutional",
    "Pooling",
    "Recurrent",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    # Loss
    "Loss",
    "CrossEntropyLoss",
    "MSELoss",
]