from cortex import Tensor

class Optimizer:
    def __init__(self, learning_rate=0.01, optimizer="SGD", beta1=0.9, beta2=0.999, epsilon=1e-8, rho=0.9):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.rho = rho
        # Adam only
        self.m = []
        self.v = []
        self.t = 0
        # RMSprop only
        self.s = []

    def step(self, params, grads): # TODO: check this
        if self.optimizer == "SGD":
            self.SGD(params, grads)
        elif self.optimizer == "Adam":
            self.Adam(params, grads)
        elif self.optimizer == "RMSprop":
            self.RMSprop(params, grads)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
    def zero_grad(self, params):
        for param in params:
            param.fill(0) # TODO: check this

    def SGD(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

    def Adam(self, params, grads):
        if not self.m:
            self.m = [Tensor.zeros_like(param) for param in params]
            self.v = [Tensor.zeros_like(param) for param in params]
        
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
            param -= self.learning_rate * m_hat / (Tensor.sqrt(v_hat) + self.epsilon)
            
    def RMSprop(self, params, grads):
        if not self.s:
            self.s = [Tensor.zeros_like(param) for param in params]
        for i, (param, grad) in enumerate(zip(params, grads)):
            # update squared gradients
            self.s[i] = self.rho * self.s[i] + (1 - self.rho) * (grad ** 2)
            # update parameter
            param -= self.learning_rate * grad / (Tensor.sqrt(self.s[i]) + self.epsilon)
    