import numpy as np
from src.model import gradient_descent, predict, compute_loss


def walk_forward_validation(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                             lr: float = 0.1, epochs: int = 2000,
                             lambda_: float = 0.0, penalty: str = "l2"):
    """
    Rolling-origin walk-forward validation for time series.

    Splits the data into n_splits sequential folds. For each fold i,
    trains on everything before it and tests on fold i itself —
    never trains on future data, unlike a random k-fold split.

    Returns:
        fold_mses: list of test MSE per fold
        avg_mse: mean test MSE across folds
    """
    n = len(X)
    fold_size = n // (n_splits + 1)  # reserve first chunk purely for initial training

    fold_mses = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)

        X_train_fold, y_train_fold = X[:train_end], y[:train_end]
        X_test_fold, y_test_fold = X[train_end:test_end], y[train_end:test_end]

        if len(X_test_fold) == 0:
            continue

        # Normalize using only this fold's training data
        X_mean, X_std = X_train_fold.mean(axis=0), X_train_fold.std(axis=0)
        X_train_norm = (X_train_fold - X_mean) / X_std
        X_test_norm = (X_test_fold - X_mean) / X_std

        w, b, _ = gradient_descent(X_train_norm, y_train_fold, lr=lr, epochs=epochs,
                                    lambda_=lambda_, penalty=penalty)

        y_pred_fold = predict(X_test_norm, w, b)
        fold_mse = compute_loss(y_test_fold, y_pred_fold)
        fold_mses.append(fold_mse)

        print(f"Fold {i}: train_size={train_end}, test_size={len(X_test_fold)}, Test MSE={fold_mse:.6f}")

    avg_mse = np.mean(fold_mses)
    return fold_mses, avg_mse