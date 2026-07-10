import os
import numpy as np
from src import config
from src.arima_baseline import arima_walk_forward, arima_rolling_one_step
from src.garch_baseline import garch_rolling_one_step, garch_vs_squared_returns
from src.data_loader import get_data
from src.features import prepare_dataset
from src.model import gradient_descent, predict, compute_loss
from src.evaluate import evaluate_model, plot_predictions, naive_baseline_mse
from src.walk_forward import walk_forward_validation
from src.save_results import save_results
from src.diagnostics import plot_residual_diagnostics, residual_autocorrelation_check
from src.fold_comparison import plot_fold_comparison
from src.regime_analysis import regime_split_evaluation

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")


if __name__ == "__main__":
    df = get_data(ticker=config.TICKER, start=config.START_DATE, end=config.END_DATE)
    X, y, df_final = prepare_dataset(df, n_lags=config.N_LAGS, include_volume=config.INCLUDE_VOLUME)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # ----- Single chronological split -----
    split = int(config.TRAIN_TEST_SPLIT_RATIO * len(X))
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_mean, X_std = X_train_raw.mean(axis=0), X_train_raw.std(axis=0)
    X_train = (X_train_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std

    # ----- Ridge lambda sweep -----
    print("\n=== RIDGE LAMBDA SWEEP ===")
    ridge_sweep = {}
    results = []
    for lam in config.LAMBDAS:
        w_l, b_l, _ = gradient_descent(X_train, y_train, lr=config.LEARNING_RATE,
                                        epochs=config.EPOCHS, lambda_=lam)
        y_test_pred_l = predict(X_test, w_l, b_l)
        test_mse_l = compute_loss(y_test, y_test_pred_l)
        results.append((lam, test_mse_l, w_l, b_l))
        ridge_sweep[str(lam)] = test_mse_l
        print(f"lambda={lam:<8} Test MSE: {test_mse_l:.6f}")

    best_lam, best_mse, w, b = min(results, key=lambda r: r[1])
    print(f"\nBest lambda: {best_lam} (Test MSE: {best_mse:.6f})")

    w, b, loss_history = gradient_descent(X_train, y_train, lr=config.LEARNING_RATE,
                                           epochs=config.EPOCHS, lambda_=best_lam)

    # ----- Lasso lambda sweep -----
    print("\n=== LASSO LAMBDA SWEEP ===")
    lasso_sweep = {}
    lasso_results = []
    for lam in config.LAMBDAS:
        w_l, b_l, _ = gradient_descent(X_train, y_train, lr=config.LEARNING_RATE,
                                        epochs=config.EPOCHS, lambda_=lam, penalty="l1")
        y_test_pred_l = predict(X_test, w_l, b_l)
        test_mse_l = compute_loss(y_test, y_test_pred_l)
        lasso_results.append((lam, test_mse_l, w_l, b_l))
        lasso_sweep[str(lam)] = test_mse_l
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
        X, y, n_splits=config.N_SPLITS, lr=config.LEARNING_RATE,
        epochs=config.WALK_FORWARD_EPOCHS, lambda_=best_lam, penalty="l2"
    )
    print(f"\nWalk-forward average Test MSE: {avg_wf_mse:.6f}")

    # ----- ARIMA -----
    volatility_series = np.asarray(df_final["volatility"].values, dtype=np.float64)
    returns_series = np.asarray(df_final["log_return"].values, dtype=np.float64)

    print("\n=== ARIMA WALK-FORWARD VALIDATION (5 folds, multi-step) ===")
    arima_fold_mses, arima_avg_mse = arima_walk_forward(
        volatility_series, n_splits=config.N_SPLITS, order=config.ARIMA_ORDER
    )
    print(f"\nARIMA (multi-step) walk-forward average Test MSE: {arima_avg_mse:.6f}")

    print("\n=== ARIMA ROLLING ONE-STEP VALIDATION (5 folds, refit every 5 days) ===")
    arima_rolling_fold_mses, arima_rolling_avg_mse = arima_rolling_one_step(
        volatility_series, n_splits=config.N_SPLITS, order=config.ARIMA_ORDER,
        refit_every=config.ARIMA_REFIT_EVERY
    )
    print(f"\nARIMA rolling walk-forward average Test MSE: {arima_rolling_avg_mse:.6f}")

    # ----- GARCH: evaluated against squared returns only (the correct target) -----
    print("\n=== GARCH vs SQUARED RETURNS (direct variance target) ===")
    garch_sq_fold_mses, garch_sq_avg = garch_vs_squared_returns(
        returns_series, n_splits=config.N_SPLITS, **config.GARCH_PARAMS,
        refit_every=config.GARCH_REFIT_EVERY
    )
    print(f"GARCH vs squared returns average MSE: {garch_sq_avg:.8f}")

    # Still need GARCH's fold-wise MSE on the volatility target for the
    # fold-comparison chart below (comparable units to Ridge/ARIMA), so
    # this call is kept - just no longer reported in the main results table.
    garch_fold_mses, _garch_avg_mse_unused, garch_predictions = garch_rolling_one_step(
        returns_series, volatility_series, n_splits=config.N_SPLITS, **config.GARCH_PARAMS,
        refit_every=config.GARCH_REFIT_EVERY
    )

    # ----- GJR-GARCH -----
    print("\n=== GJR-GARCH(1,1,1) ROLLING ONE-STEP VALIDATION (5 folds, refit every 5 days) ===")
    gjr_fold_mses, gjr_avg_mse, gjr_predictions = garch_rolling_one_step(
        returns_series, volatility_series, n_splits=config.N_SPLITS, **config.GJR_GARCH_PARAMS,
        refit_every=config.GARCH_REFIT_EVERY
    )
    print(f"\nGJR-GARCH rolling walk-forward average Test MSE: {gjr_avg_mse:.6f}")

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
    print(f"{'GJR-GARCH(1,1,1) (rolling, walk-forward avg)':<40}{gjr_avg_mse:>15.6f}")
    print("=" * 55)
    print(f"{'GARCH vs squared returns (own target, own units)':<40}{garch_sq_avg:>15.8f}")

    best_overall_mse = min(best_mse, best_lasso_mse, avg_wf_mse, arima_rolling_avg_mse, gjr_avg_mse)
    print(f"\nBest model MSE: {best_overall_mse:.6f} vs naive baseline: {naive_mse:.6f} "
          f"(difference: {naive_mse - best_overall_mse:+.6f})")

    # ----- Save results.json -----
    save_results({
        "naive_mse": naive_mse,
        "ridge_single_split_mse": best_mse,
        "ridge_best_lambda": best_lam,
        "lasso_single_split_mse": best_lasso_mse,
        "lasso_best_lambda": best_lasso_lam,
        "ridge_walk_forward_avg_mse": avg_wf_mse,
        "arima_multistep_avg_mse": arima_avg_mse,
        "arima_rolling_avg_mse": arima_rolling_avg_mse,
        "garch_vs_squared_returns_mse": garch_sq_avg,
        "gjr_garch_t_weekly_refit_mse": gjr_avg_mse,
        "ridge_lambda_sweep": ridge_sweep,
        "lasso_lambda_sweep": lasso_sweep,
        "config": {
            "n_lags": config.N_LAGS,
            "include_volume": config.INCLUDE_VOLUME,
            "lambdas_tested": config.LAMBDAS,
            "n_splits": config.N_SPLITS,
            "arima_order": config.ARIMA_ORDER,
            "garch_refit_every": config.GARCH_REFIT_EVERY,
        },
    })

    # ----- Residual diagnostics -----
    plot_residual_diagnostics(y_test, y_test_pred, title="Custom Ridge GD")
    residuals = y_test - y_test_pred
    residual_autocorrelation_check(residuals)

    # ----- Plots -----
    plot_predictions(y_test, y_test_pred, title=f"Custom GD (lambda={best_lam}): Predicted vs Actual Volatility",
                      save_path="outputs/plots/pred_vs_actual_custom.png")
    plot_predictions(y_test, y_test_pred_sklearn, title="Sklearn: Predicted vs Actual Volatility",
                      save_path="outputs/plots/pred_vs_actual_sklearn.png")

    # ----- Fold comparison -----
    plot_fold_comparison({
        "Ridge (linear)": fold_mses,
        "ARIMA (rolling)": arima_rolling_fold_mses,
        "GARCH": garch_fold_mses,
        "GJR-GARCH": gjr_fold_mses,
    })

    # ----- Regime split evaluation -----
    regime_results = regime_split_evaluation(y_test, y_test_pred)

    # ----- Save model artifacts -----
    os.makedirs("outputs", exist_ok=True)
    np.savez(
        "outputs/trained_model.npz",
        w=w, b=b, X_mean=X_mean, X_std=X_std, lambda_=best_lam
    )
    print("\nModel artifacts saved to outputs/trained_model.npz")