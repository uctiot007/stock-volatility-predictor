import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from src.model import predict, compute_loss


def evaluate_model(X_train, y_train, X_test, y_test, w, b):
    """
    Compare custom gradient descent model with sklearn.
    Evaluates both on train and test sets.
    """

    # ----- Custom model -----
    y_train_pred = predict(X_train, w, b)
    y_test_pred = predict(X_test, w, b)

    train_mse_custom = compute_loss(y_train, y_train_pred)
    test_mse_custom = compute_loss(y_test, y_test_pred)

    # ----- Sklearn model -----
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred_sklearn = model.predict(X_train)
    y_test_pred_sklearn = model.predict(X_test)

    train_mse_sklearn = compute_loss(y_train, y_train_pred_sklearn)
    test_mse_sklearn = compute_loss(y_test, y_test_pred_sklearn)

    # ----- Print results -----
    print("\n=== MODEL EVALUATION ===")

    print("\nCustom Model:")
    print(f"Train MSE: {train_mse_custom:.6f}")
    print(f"Test MSE:  {test_mse_custom:.6f}")

    print("\nSklearn Model:")
    print(f"Train MSE: {train_mse_sklearn:.6f}")
    print(f"Test MSE:  {test_mse_sklearn:.6f}")

    print("\nWeights comparison:")
    print("Custom :", w)
    print("Sklearn:", model.coef_)

    print("\nBias comparison:")
    print("Custom :", b)
    print("Sklearn:", model.intercept_)

    return y_test_pred, y_test_pred_sklearn


def plot_predictions(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    """
    Plot actual vs predicted values. Optionally save to disk.
    """
    plt.figure(figsize=(10, 5))

    plt.plot(y_true[:100], label="Actual")
    plt.plot(y_pred[:100], label="Predicted")

    plt.title(title)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Volatility")

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()