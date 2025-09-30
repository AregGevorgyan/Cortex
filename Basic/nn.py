import numpy as np
from tqdm import tqdm

class Layer: 
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Loss(Layer):
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.W, self.inputs) + self.b

    def backward(self, grad_output):
        grad_W = np.dot(grad_output, self.inputs.T)
        grad_b = np.sum(grad_output, axis=1, keepdims=True)
        grad_inputs = np.dot(self.W.T, grad_output)
        return grad_inputs, [grad_W, grad_b]

    def params(self):
        return [self.W, self.b]

# TODO: other types of layers
class Convolutional(Layer):
    def __init__(self, num_filters, filter_size, input_depth):
        pass

    def forward(self, inputs):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class Pooling(Layer):
    def __init__(self, pool_size, input_depth):
        pass

    def forward(self, inputs):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class Recurrent(Layer):
    def __init__(self, input_size, hidden_size):
        pass

    def forward(self, inputs):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient):
        return output_gradient * (self.inputs > 0)
    
class Sigmoid(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return 1 / (1 + np.exp(-inputs))

    def backward(self, output_gradient, learning_rate):
        sigmoid = self.forward(self.inputs)
        return output_gradient * sigmoid * (1 - sigmoid)

class Tanh(Layer):
    def forward(self, inputs):
        return np.tanh(inputs)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (1 - np.tanh(self.inputs) ** 2)
    
class Softmax(Layer):
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        self.outputs = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        return self.outputs

    def backward(self, output_gradient, learning_rate):
        jacobian = np.diagflat(self.outputs) - np.dot(self.outputs, self.outputs.T)
        return np.dot(jacobian, output_gradient)
    
class CrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=0))

    def backward(self):
        return - (self.y_true / (self.y_pred + 1e-9)) / self.y_true.shape[1]
    
class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]

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

    def SGD(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

    def Adam(self, params, grads):
        if self.m is None:
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
    
class Utils:
    @staticmethod
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    @staticmethod
    def shuffle(data, labels):
        p = np.random.permutation(len(data))
        return data[p], labels[p]
    
    @staticmethod
    def train_test_split(data, labels, test_size=0.2):
        split_idx = int(len(data) * (1 - test_size))
        return data[:split_idx], data[split_idx:], labels[:split_idx], labels[split_idx:]