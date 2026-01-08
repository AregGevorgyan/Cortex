from cortex import Tensor
import numpy as np
from tqdm import tqdm

class DataLoader:
    """Simple batched data loader"""
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]

            yield Tensor(X_batch), Tensor(y_batch)

    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def train_epoch(model, dataloader, loss_fn, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        # Forward pass
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        # Backward pass
        optimizer.zero_grad(model.parameters())
        loss.backward()

        # Update weights
        optimizer.step(model.parameters())

        total_loss += loss.data
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, loss_fn):
    """Evaluate model on validation/test data"""
    model.eval()
    total_loss = 0
    n_batches = 0

    from cortex import no_grad
    with no_grad():
        for X_batch, y_batch in dataloader:
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            total_loss += loss.data
            n_batches += 1

    return total_loss / n_batches


def fit(model, train_loader, val_loader, loss_fn, optimizer,
        n_epochs=10, verbose=True):
    """Complete training loop"""
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss = evaluate(model, val_loader, loss_fn)
        history['val_loss'].append(val_loss)

        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"train_loss: {train_loss:.4f} - "
                  f"val_loss: {val_loss:.4f}")

    return history
