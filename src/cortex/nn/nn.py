from tensor import Tensor

# Abstract base class to structure all the nn modules
class Module:
    """Base class for all neural network modules"""
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def parameters(self):
        """Return all parameters for optimization"""
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        self.training = False
        for module in self._modules.values():
            module.eval()

###########################################################################
# Layers
###########################################################################

class Linear(Module):
    """Fully connected layer"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # Xavier initialization
        limit = Tensor.sqrt(6 / (in_features + out_features))
        self._parameters['weight'] = Tensor(
            Tensor.random.uniform(-limit, limit, (in_features, out_features)),
            requires_grad=True
        )

        if bias:
            self._parameters['bias'] = Tensor(
                Tensor.zeros((1, out_features)),
                requires_grad=True
            )
        else:
            self._parameters['bias'] = None

    def forward(self, x):
        out = x @ self._parameters['weight']
        if self._parameters['bias'] is not None:
            out = out + self._parameters['bias']
        return out


class Sequential(Module):
    """Container for sequential modules"""
    def __init__(self, *layers):
        super().__init__()
        for idx, layer in enumerate(layers):
            self._modules[str(idx)] = layer

    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x
    
###########################################################################
# Activation Functions
###########################################################################

class ReLU(Module):
    """ReLU activation function"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return Tensor.maximum(x, Tensor.zeros_like(x))