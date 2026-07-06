import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

from src.model import predict, compute_loss


def evaluate_model(X_train, y_train, X_test, y_test, w, b, lambda_=0.0):
    """
    Compare custom gradient descent model with sklearn.
    Evaluates against both plain LinearRegression and Ridge
    (Ridge's alpha is scaled by n_train to match our averaged-MSE
    loss convention, for a fair comparison).
    """
    n_train = X_train.shape[0]

    # ----- Custom model -----
    y_train_pred = predict(X_train, w, b)
    y_test_pred = predict(X_test, w, b)

    train_mse_custom = compute_loss(y_train, y_train_pred)
    test_mse_custom = compute_loss(y_test, y_test_pred)

    # ----- Sklearn plain LinearRegression (no regularization) -----
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)

    y_train_pred_sklearn = lin_model.predict(X_train)
    y_test_pred_sklearn = lin_model.predict(X_test)

    train_mse_sklearn = compute_loss(y_train, y_train_pred_sklearn)
    test_mse_sklearn = compute_loss(y_test, y_test_pred_sklearn)

    # ----- Sklearn Ridge (alpha scaled by n_train for a fair comparison) -----
    # Our loss:    (1/n) * sum(error^2) + lambda_ * sum(w^2)
    # Sklearn loss:       sum(error^2) + alpha    * sum(w^2)
    # So alpha = lambda_ * n_train makes the penalties equivalent.
    ridge_model = Ridge(alpha=lambda_ * n_train)
    ridge_model.fit(X_train, y_train)

    y_test_pred_ridge = ridge_model.predict(X_test)
    test_mse_ridge_sklearn = compute_loss(y_test, y_test_pred_ridge)

    # ----- Print results -----
    print("\n=== MODEL EVALUATION ===")

    print("\nCustom Model:")
    print(f"Train MSE: {train_mse_custom:.6f}")
    print(f"Test MSE:  {test_mse_custom:.6f}")

    print("\nSklearn LinearRegression (no regularization):")
    print(f"Train MSE: {train_mse_sklearn:.6f}")
    print(f"Test MSE:  {test_mse_sklearn:.6f}")

    print(f"\nSklearn Ridge (alpha={lambda_ * n_train:.4f}, equivalent to our lambda_={lambda_}):")
    print(f"Test MSE:  {test_mse_ridge_sklearn:.6f}")

    print("\nWeights comparison:")
    print("Custom       :", w)
    print("Sklearn OLS  :", lin_model.coef_)
    print("Sklearn Ridge:", ridge_model.coef_)

    print("\nBias comparison:")
    print("Custom       :", b)
    print("Sklearn OLS  :", lin_model.intercept_)
    print("Sklearn Ridge:", ridge_model.intercept_)

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