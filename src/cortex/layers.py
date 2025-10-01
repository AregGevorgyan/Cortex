import numpy as np

__all__ = [
    "Layer", "Dense", "Convolutional", "Pooling", "Recurrent",
    "ReLU", "Sigmoid", "Tanh", "Softmax"
]

class Layer: 
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.W) + self.b

    def backward(self, grad_output):
        grad_W = np.dot(self.inputs.T, grad_output)
        grad_b = np.sum(grad_output, axis=0, keepdims=True)
        grad_inputs = np.dot(grad_output, self.W.T)
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
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.outputs

    def backward(self, grad_output):
        return grad_output