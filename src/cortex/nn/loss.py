from cortex import Tensor


class Loss:
    """Base class for loss functions"""
    def __call__(self, pred, target):
        return self.forward(pred, target)

    def forward(self, pred, target):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """Cross entropy loss using Tensor operations"""
    def forward(self, pred, target):
        """
        Args:
            pred: Tensor of shape (batch, num_classes) - logits
            target: Tensor of shape (batch,) - class indices

        Returns:
            Scalar loss tensor
        """
        # Use built-in cross_entropy
        return pred.cross_entropy(target)


class MSELoss(Loss):
    """Mean squared error loss"""
    def forward(self, pred, target):
        """
        Args:
            pred: Tensor of predictions
            target: Tensor of targets (same shape as pred)

        Returns:
            Scalar loss tensor
        """
        diff = pred - target
        return (diff * diff).mean()


class BCELoss(Loss):
    """Binary cross entropy loss"""
    def forward(self, pred, target):
        """
        Args:
            pred: Tensor of predictions (probabilities)
            target: Tensor of binary targets (0 or 1)

        Returns:
            Scalar loss tensor
        """
        # BCE = -[y*log(p) + (1-y)*log(1-p)]
        eps = 1e-7  # For numerical stability
        pred_clipped = pred.clip(eps, 1 - eps)
        loss = -(target * pred_clipped.log() + (1 - target) * (1 - pred_clipped).log())
        return loss.mean()