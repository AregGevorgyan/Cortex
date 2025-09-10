# Numpy like but jit compatible and compiles to vectorized GPU and TPU code
import jax.numpy as jnp

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.params 

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

class Activation:
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn

class Autograd:
    def __init__(self):
        pass

class Loss:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn