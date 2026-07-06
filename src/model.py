import numpy as np


def initialize_params(n_features: int):
    """
    Initialize weights (zeros) and bias (zero).
    """
    w = np.zeros(n_features)
    b = 0.0
    return w, b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """
    Linear model prediction: y = Xw + b
    """
    return np.dot(X, w) + b


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray = None, lambda_: float = 0.0) -> float:
    """
    Mean Squared Error, optionally with L2 (ridge) penalty.
    Bias is never penalized, by convention.
    """
    mse = np.mean((y_pred - y_true) ** 2)

    if w is not None and lambda_ > 0:
        return mse + lambda_ * np.sum(w ** 2)

    return mse


def compute_gradients(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, w: np.ndarray = None, lambda_: float = 0.0):
    """
    Compute gradients of the (optionally ridge-regularized) MSE loss.
    Ridge adds +2*lambda*w to dw only (bias is not regularized).
    """
    n = len(y)
    error = y_pred - y

    dw = (2 / n) * np.dot(X.T, error)
    db = (2 / n) * np.sum(error)

    if w is not None and lambda_ > 0:
        dw += 2 * lambda_ * w

    return dw, db


def gradient_descent(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 500, lambda_: float = 0.0):
    """
    Train linear regression using batch gradient descent.
    Set lambda_ > 0 to apply L2 (ridge) regularization.

    Returns:
        w, b, loss_history
    """
    n_samples, n_features = X.shape
    w, b = initialize_params(n_features)

    loss_history = []

    for epoch in range(epochs):
        y_pred = predict(X, w, b)

        loss = compute_loss(y, y_pred, w, lambda_)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, y_pred, w, lambda_)

        w -= lr * dw
        b -= lr * db

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    return w, b, loss_history


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.features import prepare_dataset

    raw = load_data()
    X, y, df = prepare_dataset(raw)

    # Normalize features before GD — required for stable convergence
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    w, b, losses = gradient_descent(X_norm, y, lr=0.1, epochs=1000, lambda_=0.01)

    print("Final weights:", w)
    print("Final bias:", b)
    print("Final loss:", losses[-1])