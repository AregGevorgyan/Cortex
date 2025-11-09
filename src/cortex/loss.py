from .backend import np

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
        # clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-9, 1 - 1e-9)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    def backward(self):
        grad = self.y_pred - self.y_true
        grad = grad / self.y_true.shape[0]
        return grad
    
class MSELoss(Loss):
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]