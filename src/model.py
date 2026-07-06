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


def compute_loss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (MSE)
    """
    return np.mean((y - y_pred) ** 2)


def compute_gradients(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
    """
    Compute gradients of MSE loss.

    dw = -2/n * X^T (y - y_pred)
    db = -2/n * sum(y - y_pred)
    """
    n = len(y)
    error = y - y_pred

    dw = (-2 / n) * np.dot(X.T, error)
    db = (-2 / n) * np.sum(error)

    return dw, db


def gradient_descent(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000):
    """
    Train linear regression using batch gradient descent.

    Returns:
        w, b, loss_history
    """
    n_features = X.shape[1]
    w, b = initialize_params(n_features)

    loss_history = []

    for i in range(epochs):
        y_pred = predict(X, w, b)

        loss = compute_loss(y, y_pred)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, y_pred)

        w -= lr * dw
        b -= lr * db

        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss:.6f}")

    return w, b, loss_history


if __name__ == "__main__":
    from src.data_loader import load_data
    from src.features import prepare_dataset

    raw = load_data()
    X, y, df = prepare_dataset(raw)

    # IMPORTANT: features are on very different scales — normalize before GD
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    w, b, losses = gradient_descent(X_norm, y, lr=0.1, epochs=1000)

    print("Final weights:", w)
    print("Final bias:", b)
    print("Final loss:", losses[-1])