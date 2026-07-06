import os
import numpy as np
from src.data_loader import get_data
from src.features import prepare_dataset
from src.model import gradient_descent, predict, compute_loss
from src.evaluate import evaluate_model, plot_predictions, naive_baseline_mse
from src.walk_forward import walk_forward_validation

if __name__ == "__main__":
    df = get_data()
    X, y, df_final = prepare_dataset(df, n_lags=5, include_volume=True)

    # ----- Single chronological split (existing approach) -----
    split = int(0.8 * len(X))
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_mean, X_std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    X_train = (X_train_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    # ----- Lambda sweep (unchanged) -----
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
    # vol_lag_1 is the first feature column (most recent day) — use RAW, unnormalized values
    naive_mse = naive_baseline_mse(y_test, X_test_raw, lag1_col_idx=0)

    # ----- Sklearn comparison (existing) -----
    y_test_pred, y_test_pred_sklearn = evaluate_model(
        X_train, y_train, X_test, y_test, w, b, lambda_=best_lam
    )

    # ----- Walk-forward validation -----
    print("\n=== WALK-FORWARD VALIDATION (5 folds) ===")
    fold_mses, avg_wf_mse = walk_forward_validation(
        X, y, n_splits=5, lr=0.1, epochs=2000, lambda_=best_lam, penalty="l2"
    )
    print(f"\nWalk-forward average Test MSE: {avg_wf_mse:.6f}")

    # ----- Proper comparison table -----
    print("\n" + "=" * 55)
    print("FINAL COMPARISON")
    print("=" * 55)
    print(f"{'Method':<40}{'Test MSE':>15}")
    print("-" * 55)
    print(f"{'Naive baseline (today = tomorrow)':<40}{naive_mse:>15.6f}")
    print(f"{'Custom Ridge GD (single split)':<40}{best_mse:>15.6f}")
    print(f"{'Custom Lasso GD (single split)':<40}{best_lasso_mse:>15.6f}")
    print(f"{'Custom Ridge GD (walk-forward avg)':<40}{avg_wf_mse:>15.6f}")
    print("=" * 55)

    beats_naive = best_mse < naive_mse
    print(f"\nModel beats naive baseline: {beats_naive}")

    # ----- Plots (unchanged) -----
    plot_predictions(y_test, y_test_pred, title=f"Custom GD (lambda={best_lam}): Predicted vs Actual Volatility")
    plot_predictions(y_test, y_test_pred_sklearn, title="Sklearn: Predicted vs Actual Volatility")

    # ----- Save model artifacts -----
    os.makedirs("outputs", exist_ok=True)
    np.savez(
        "outputs/trained_model.npz",
        w=w, b=b, X_mean=X_mean, X_std=X_std, lambda_=best_lam
    )
    print("\nModel artifacts saved to outputs/trained_model.npz")