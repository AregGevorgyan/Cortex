from contextlib import contextmanager
from .backend import np

# Global flag for no_grad context
_grad_enabled = True

@contextmanager
def no_grad():
    """Context manager to disable gradient tracking.

    Usage:
        with no_grad():
            output = model(inputs)  # No gradients computed
    """
    global _grad_enabled
    prev_state = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev_state

class Tensor:
    def __init__(self, data, requires_grad=True, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        # If this tensor is the result of an operation, infer requires_grad from parents
        # Only infer if requires_grad is True (default); respect explicit False
        if _children and requires_grad:
            requires_grad = any(getattr(c, 'requires_grad', False) for c in _children)
        # Respect no_grad context
        if not _grad_enabled:
            requires_grad = False
        self.requires_grad = requires_grad
        self.grad = None
        
        # For computational graph
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    def backward(self, grad=None, retain_graph=False):
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size == 1:
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("grad must be specified for non-scalar tensors")

        # Initialize gradient (don't add here, let _backward functions handle it)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

        # Topological sort and backprop
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in reversed(topo):
            node._backward()

        # Clear graph to prevent memory leaks (unless retain_graph=True)
        if not retain_graph:
            for node in topo:
                node._prev = set()
                node._backward = lambda: None

    def zero_grad(self):
        self.grad = None

    def _unbroadcast(self, grad, shape):
        """Helper to handle gradient broadcasting"""
        # Sum out added dims
        ndims_added = len(grad.shape) - len(shape)
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        
        # Sum over broadcasted dims
        for i, (dim, orig_dim) in enumerate(zip(grad.shape, shape)):
            if orig_dim == 1 and dim != 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                grad = self._unbroadcast(out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
            if other.requires_grad:
                grad = self._unbroadcast(out.grad, other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad = other.grad + grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data - other.data, _children=(self, other), _op='-')
        
        def _backward():
            if self.requires_grad:
                grad = self._unbroadcast(out.grad, self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
            if other.requires_grad:
                grad = self._unbroadcast(-out.grad, other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad = other.grad + grad
        
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        return Tensor(other, requires_grad=False).__sub__(self)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                grad = self._unbroadcast(out.grad * other.data, self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
            if other.requires_grad:
                grad = self._unbroadcast(out.grad * self.data, other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad = other.grad + grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, _children=(self, other), _op='/')
        
        def _backward():
            if self.requires_grad:
                grad = self._unbroadcast(out.grad / other.data, self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
            if other.requires_grad:
                grad = self._unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad = other.grad + grad
        
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                # Handle batched matmul: transpose only the last two dimensions
                if other.data.ndim >= 2:
                    other_T = np.swapaxes(other.data, -2, -1)
                    grad = out.grad @ other_T
                else:
                    grad = out.grad @ other.data.T
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
            if other.requires_grad:
                # Handle batched matmul: transpose only the last two dimensions
                if self.data.ndim >= 2:
                    self_T = np.swapaxes(self.data, -2, -1)
                    grad = self_T @ out.grad
                else:
                    grad = self.data.T @ out.grad
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad = other.grad + grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power only supports int/float exponents"
        out = Tensor(self.data ** other, _children=(self,), _op=f'**{other}')

        def _backward():
            if self.requires_grad:
                grad = out.grad * (other * self.data ** (other - 1))
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                grad = out.grad * (1 / self.data)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def exp(self):
        out_data = np.exp(self.data)
        out = Tensor(out_data, _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                grad = out.grad * out_data
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                if axis is None:
                    grad = out.grad * np.ones_like(self.data)
                else:
                    # Expand gradient back to original shape
                    grad_shape = list(self.data.shape)
                    if not keepdims:
                        if isinstance(axis, int):
                            grad_expanded = np.expand_dims(out.grad, axis)
                        else:
                            grad_expanded = out.grad
                            for ax in sorted(axis):
                                grad_expanded = np.expand_dims(grad_expanded, ax)
                    else:
                        grad_expanded = out.grad
                    grad = np.broadcast_to(grad_expanded, self.data.shape).copy()
                
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), _children=(self,), _op='max')

        def _backward():
            if self.requires_grad:
                if axis is None:
                    # For scalar max, gradient goes to all max elements equally
                    mask = (self.data == np.max(self.data)).astype(np.float32)
                    grad = out.grad * mask / np.sum(mask)
                else:
                    # For axis max, expand to match original shape
                    if not keepdims:
                        if isinstance(axis, int):
                            out_grad_expanded = np.expand_dims(out.grad, axis)
                            out_data_expanded = np.expand_dims(out.data, axis)
                        else:
                            out_grad_expanded = out.grad
                            out_data_expanded = out.data
                            for ax in sorted(axis):
                                out_grad_expanded = np.expand_dims(out_grad_expanded, ax)
                                out_data_expanded = np.expand_dims(out_data_expanded, ax)
                    else:
                        out_grad_expanded = out.grad
                        out_data_expanded = out.data
                    
                    mask = (self.data == out_data_expanded).astype(np.float32)
                    grad = out_grad_expanded * mask / (np.sum(mask, axis=axis, keepdims=True) + 1e-10)
                
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.data.shape)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out = Tensor(np.transpose(self.data, axes), _children=(self,), _op='transpose')

        def _backward():
            if self.requires_grad:
                inv_axes = None if axes is None else np.argsort(axes)
                grad = np.transpose(out.grad, inv_axes)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, _children=(self,), _op='neg')

        def _backward():
            if self.requires_grad:
                grad = -out.grad
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def __getitem__(self, slc):
        out = Tensor(self.data[slc], _children=(self,), _op=f'getitem {slc}')

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                # Accumulate in case of repeated/advanced indices
                np.add.at(grad, slc, out.grad)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), _children=(self,), _op='mean')

        def _backward():
            if self.requires_grad:
                if axis is None:
                    # Global mean: distribute gradient equally to all elements
                    grad = out.grad * np.ones_like(self.data) / self.data.size
                else:
                    # Mean along specific axes: expand gradient back to original shape
                    if not keepdims:
                        if isinstance(axis, int):
                            grad_expanded = np.expand_dims(out.grad, axis)
                        else:
                            grad_expanded = out.grad
                            for ax in sorted(axis):
                                grad_expanded = np.expand_dims(grad_expanded, ax)
                    else:
                        grad_expanded = out.grad

                    # Divide by the number of elements averaged over
                    n = np.prod([self.data.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
                    grad = np.broadcast_to(grad_expanded, self.data.shape).copy() / n

                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out
    
    def unsqueeze(self, axis):
        out = Tensor(np.expand_dims(self.data, axis), _children=(self,), _op='unsqueeze')

        def _backward():
            if self.requires_grad:
                grad = np.squeeze(out.grad, axis)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out
    
    def squeeze(self, axis):
        out = Tensor(np.squeeze(self.data, axis), _children=(self,), _op='squeeze')

        def _backward():
            if self.requires_grad:
                grad = np.expand_dims(out.grad, axis)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out
    
    def concat(self, other, axis=0):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(np.concatenate((self.data, other.data), axis=axis), _children=(self, other), _op='concat')

        def _backward():
            split_idx = self.data.shape[axis]
            if self.requires_grad:
                grad = np.split(out.grad, [split_idx], axis=axis)[0]
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
            if other.requires_grad:
                grad = np.split(out.grad, [split_idx], axis=axis)[1]
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad = other.grad + grad

        out._backward = _backward
        return out
    
    def stack(self, other, axis=0):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(np.stack((self.data, other.data), axis=axis), _children=(self, other), _op='stack')

        def _backward():
            if self.requires_grad:
                grad = np.take(out.grad, 0, axis=axis)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
            if other.requires_grad:
                grad = np.take(out.grad, 1, axis=axis)
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad = other.grad + grad

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        exps = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        out = Tensor(exps / np.sum(exps, axis=axis, keepdims=True), _children=(self,), _op='softmax')

        def _backward():
            if self.requires_grad:
                s = out.data
                grad = s * (out.grad - (out.grad * s).sum(axis=axis, keepdims=True))
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def log_softmax(self, axis=-1):
        # Numerically stable log-softmax
        max_x = np.max(self.data, axis=axis, keepdims=True)
        shifted = self.data - max_x
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        lsm = shifted - log_sum_exp
        out = Tensor(lsm, _children=(self,), _op='log_softmax')

        def _backward():
            if self.requires_grad:
                # d/dx log_softmax(x) = I - softmax(x)
                s = np.exp(out.data)  # softmax derived from log-softmax
                grad = out.grad - s * np.sum(out.grad, axis=axis, keepdims=True)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def cross_entropy(self, y_true):
        """Compute cross-entropy loss with proper autograd support"""
        # Ensure y_true is a Tensor
        if not isinstance(y_true, Tensor):
            y_true = Tensor(y_true, requires_grad=False)
        
        # Use numerically stable log_softmax directly
        log_probs = self.log_softmax(axis=-1)
        
        # Multiply with one-hot labels and sum
        prod = y_true * log_probs
        sum_val = prod.sum()
        
        # Negative mean
        batch_size = float(self.data.shape[0])
        loss = -sum_val / batch_size
        
        return loss

    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op='ReLU')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, _children=(self,), _op='Sigmoid')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * sig * (1 - sig)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
        
        out._backward = _backward
        return out
    
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, _children=(self,), _op='Tanh')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (1 - t ** 2)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def __str__(self):
        return f"Tensor({self.data})"
    
    def clip(self, min_val, max_val):
        """Clip tensor values to range [min_val, max_val]"""
        out = Tensor(np.clip(self.data, min_val, max_val), _children=(self,), _op='clip')
        
        def _backward():
            if self.requires_grad:
                mask = ((self.data >= min_val) & (self.data <= max_val)).astype(np.float32)
                grad = out.grad * mask
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad
        
        out._backward = _backward
        return out
    
    def abs(self):
        """Absolute value"""
        out = Tensor(np.abs(self.data), _children=(self,), _op='abs')

        def _backward():
            if self.requires_grad:
                grad = out.grad * np.sign(self.data)
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad = self.grad + grad

        out._backward = _backward
        return out

    def detach(self):
        """Return a new tensor detached from the computation graph.
        The returned tensor will not track gradients."""
        return Tensor(self.data.copy(), requires_grad=False)