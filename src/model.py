import numpy as np


def initialize_params(n_features: int):
    w = np.zeros(n_features)
    b = 0.0
    return w, b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.dot(X, w) + b


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray = None,
                  lambda_: float = 0.0, penalty: str = "l2") -> float:
    """
    MSE, optionally with L2 (ridge) or L1 (lasso) penalty.
    penalty: "l2" (ridge, default) or "l1" (lasso). Bias is never penalized.
    """
    mse = np.mean((y_pred - y_true) ** 2)

    if w is not None and lambda_ > 0:
        if penalty == "l2":
            return mse + lambda_ * np.sum(w ** 2)
        elif penalty == "l1":
            return mse + lambda_ * np.sum(np.abs(w))
        else:
            raise ValueError(f"Unknown penalty type: {penalty}")

    return mse


def compute_gradients(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, w: np.ndarray = None,
                       lambda_: float = 0.0, penalty: str = "l2"):
    """
    Gradients of the (optionally regularized) MSE loss.
    Ridge (l2): dw += 2*lambda*w
    Lasso (l1): dw += lambda*sign(w)
    """
    n = len(y)
    error = y_pred - y

    dw = (2 / n) * np.dot(X.T, error)
    db = (2 / n) * np.sum(error)

    if w is not None and lambda_ > 0:
        if penalty == "l2":
            dw += 2 * lambda_ * w
        elif penalty == "l1":
            dw += lambda_ * np.sign(w)
        else:
            raise ValueError(f"Unknown penalty type: {penalty}")

    return dw, db


def gradient_descent(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 500,
                      lambda_: float = 0.0, penalty: str = "l2"):
    """
    Train linear regression using batch gradient descent.
    penalty: "l2" for ridge, "l1" for lasso. Set lambda_=0.0 for plain OLS.

    Returns:
        w, b, loss_history
    """
    n_samples, n_features = X.shape
    w, b = initialize_params(n_features)

    loss_history = []

    for epoch in range(epochs):
        y_pred = predict(X, w, b)

        loss = compute_loss(y, y_pred, w, lambda_, penalty)
        loss_history.append(loss)

        dw, db = compute_gradients(X, y, y_pred, w, lambda_, penalty)

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

    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    w, b, losses = gradient_descent(X_norm, y, lr=0.1, epochs=1000, lambda_=0.01, penalty="l2")

    print("Final weights:", w)
    print("Final bias:", b)
    print("Final loss:", losses[-1])