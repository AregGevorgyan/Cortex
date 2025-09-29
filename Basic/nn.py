import numpy as np

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
        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.W, self.inputs) + self.b

    def backward(self, output_gradient, learning_rate):
        grad_inputs = np.dot(output_gradient, self.W.T)
        grad_W = np.dot(output_gradient.T, self.inputs.T)
        grad_b = np.sum(output_gradient, axis=1, keepdims=True)

        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

        return grad_inputs

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
        return np.maximum(0, inputs)

    def backward(self, output_gradient, learning_rate):
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

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.predict(X)
            loss = self.layers[-1].forward(output, y)
            grad = self.layers[-1].backward(output, y)
            for layer in reversed(self.layers[:-1]):
                grad = layer.backward(grad, learning_rate)

class Optimizer:
    def __init__(self, learning_rate, optimizer, loss):
        self.learning_rate = learning_rate
    
    def SGD(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

    def Adam(self, params, grads):
        pass

    def MSE(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def CrossEntropy(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=-1))
    
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