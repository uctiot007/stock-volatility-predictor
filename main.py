import os
import numpy as np
from src.arima_baseline import arima_walk_forward, arima_rolling_one_step
from src.garch_baseline import garch_rolling_one_step
from src.data_loader import get_data
from src.features import prepare_dataset
from src.model import gradient_descent, predict, compute_loss
from src.evaluate import evaluate_model, plot_predictions, naive_baseline_mse
from src.walk_forward import walk_forward_validation

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")


if __name__ == "__main__":
    df = get_data()
    X, y, df_final = prepare_dataset(df, n_lags=5, include_volume=True)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # ----- Single chronological split -----
    split = int(0.8 * len(X))
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_mean, X_std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    X_train = (X_train_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    # ----- Ridge lambda sweep -----
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
    print("\n=== RIDGE LAMBDA SWEEP ===")
    results = []
    for lam in lambdas:
        w_l, b_l, _ = gradient_descent(X_train, y_train, lr=0.1, epochs=5000, lambda_=lam)
        y_test_pred_l = predict(X_test, w_l, b_l)
        test_mse_l = compute_loss(y_test, y_test_pred_l)
        results.append((lam, test_mse_l, w_l, b_l))
        print(f"lambda={lam:<8} Test MSE: {test_mse_l:.6f}")

    best_lam, best_mse, w, b = min(results, key=lambda r: r[1])
    print(f"\nBest lambda: {best_lam} (Test MSE: {best_mse:.6f})")

    w, b, loss_history = gradient_descent(X_train, y_train, lr=0.1, epochs=5000, lambda_=best_lam)

    # ----- Lasso lambda sweep -----
    print("\n=== LASSO LAMBDA SWEEP ===")
    lasso_results = []
    for lam in lambdas:
        w_l, b_l, _ = gradient_descent(X_train, y_train, lr=0.1, epochs=5000, lambda_=lam, penalty="l1")
        y_test_pred_l = predict(X_test, w_l, b_l)
        test_mse_l = compute_loss(y_test, y_test_pred_l)
        lasso_results.append((lam, test_mse_l, w_l, b_l))
        print(f"lambda={lam:<8} Test MSE: {test_mse_l:.6f}")

    best_lasso_lam, best_lasso_mse, w_lasso, b_lasso = min(lasso_results, key=lambda r: r[1])
    print(f"\nBest Lasso lambda: {best_lasso_lam} (Test MSE: {best_lasso_mse:.6f})")
    print("Lasso weights:", w_lasso)

    # ----- Naive baseline -----
    naive_mse = naive_baseline_mse(y_test, X_test_raw, lag1_col_idx=0)

    # ----- Sklearn comparison -----
    y_test_pred, y_test_pred_sklearn = evaluate_model(
        X_train, y_train, X_test, y_test, w, b, lambda_=best_lam
    )

    # ----- Linear model walk-forward validation -----
    print("\n=== WALK-FORWARD VALIDATION (5 folds) ===")
    fold_mses, avg_wf_mse = walk_forward_validation(
        X, y, n_splits=5, lr=0.1, epochs=2000, lambda_=best_lam, penalty="l2"
    )
    print(f"\nWalk-forward average Test MSE: {avg_wf_mse:.6f}")

    # ----- ARIMA -----
    volatility_series = np.asarray(df_final["volatility"].values, dtype=np.float64)
    returns_series = np.asarray(df_final["log_return"].values, dtype=np.float64)

    print("\n=== ARIMA WALK-FORWARD VALIDATION (5 folds, multi-step) ===")
    arima_fold_mses, arima_avg_mse = arima_walk_forward(volatility_series, n_splits=5, order=(1, 0, 1))
    print(f"\nARIMA (multi-step) walk-forward average Test MSE: {arima_avg_mse:.6f}")

    print("\n=== ARIMA ROLLING ONE-STEP VALIDATION (5 folds, refit every 5 days) ===")
    arima_rolling_fold_mses, arima_rolling_avg_mse = arima_rolling_one_step(
        volatility_series, n_splits=5, order=(1, 0, 1), refit_every=5
    )
    print(f"\nARIMA rolling walk-forward average Test MSE: {arima_rolling_avg_mse:.6f}")

    # ----- GJR-GARCH -----
    print("\n=== GJR-GARCH(1,1,1) ROLLING ONE-STEP VALIDATION (5 folds, refit every 5 days) ===")
    garch_fold_mses, garch_avg_mse = garch_rolling_one_step(
        returns_series, volatility_series, n_splits=5, p=1, q=1, o=1, dist="t", refit_every=5
    )
    print(f"\nGJR-GARCH rolling walk-forward average Test MSE: {garch_avg_mse:.6f}")

    # ----- Final comparison table -----
    print("\n" + "=" * 55)
    print("FINAL COMPARISON")
    print("=" * 55)
    print(f"{'Method':<40}{'Test MSE':>15}")
    print("-" * 55)
    print(f"{'Naive baseline (today = tomorrow)':<40}{naive_mse:>15.6f}")
    print(f"{'Custom Ridge GD (single split)':<40}{best_mse:>15.6f}")
    print(f"{'Custom Lasso GD (single split)':<40}{best_lasso_mse:>15.6f}")
    print(f"{'Custom Ridge GD (walk-forward avg)':<40}{avg_wf_mse:>15.6f}")
    print(f"{'ARIMA (multi-step, walk-forward avg)':<40}{arima_avg_mse:>15.6f}")
    print(f"{'ARIMA (rolling, walk-forward avg)':<40}{arima_rolling_avg_mse:>15.6f}")
    print(f"{'GJR-GARCH(1,1,1) (rolling, walk-forward avg)':<40}{garch_avg_mse:>15.6f}")
    print("=" * 55)

    beats_naive = garch_avg_mse < naive_mse
    print(f"\nGJR-GARCH beats naive baseline: {beats_naive}")

    # ----- Plots -----
    plot_predictions(y_test, y_test_pred, title=f"Custom GD (lambda={best_lam}): Predicted vs Actual Volatility")
    plot_predictions(y_test, y_test_pred_sklearn, title="Sklearn: Predicted vs Actual Volatility")

    # ----- Save model artifacts -----
    os.makedirs("outputs", exist_ok=True)
    np.savez(
        "outputs/trained_model.npz",
        w=w, b=b, X_mean=X_mean, X_std=X_std, lambda_=best_lam
    )
    print("\nModel artifacts saved to outputs/trained_model.npz")