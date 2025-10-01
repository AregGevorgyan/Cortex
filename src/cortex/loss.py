import numpy as np

__all__ = ["Loss", "CrossEntropyLoss", "MSELoss"]

class Loss(object):
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self, y_pred, y_true):
        raise NotImplementedError
    
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