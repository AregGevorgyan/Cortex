import numpy as np
from tqdm import tqdm

from .layers import *
from .loss import *

__all__ = [
    "NeuralNetwork", "Optimizer", "Utils"
]

class NeuralNetwork:
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, learning_rate):
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        for epoch in pbar:
            y_pred = self.predict(X)
            loss = self.loss.forward(y_pred, y)
            grad = self.loss.backward()
            grads = []
            params = []
            for layer in reversed(self.layers):
                if isinstance(layer, Dense):
                    grad, layer_grads = layer.backward(grad)
                    grads.extend(layer_grads)
                    params.extend(layer.params())

            self.optimizer.step(params, grads)
            pbar.set_postfix({"loss": float(loss)})

class Optimizer:
    def __init__(self, learning_rate=0.01, optimizer="SGD", beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Adam only
        self.m = []
        self.v = []
        self.t = 0

    def step(self, params, grads):
        if self.optimizer == "SGD":
            self.SGD(params, grads)
        elif self.optimizer == "Adam":
            self.Adam(params, grads)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def SGD(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

    def Adam(self, params, grads):
        if not self.m:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]
        
        self.t += 1 
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # update first momemnt estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # update second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # update parameter
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    