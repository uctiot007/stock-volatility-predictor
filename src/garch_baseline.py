import numpy as np
import warnings
from arch import arch_model
from src.model import compute_loss

warnings.filterwarnings("ignore")


from typing import Literal

def fit_predict_garch(train_returns: np.ndarray, n_steps: int = 1, p: int = 1, q: int = 1,
                       o: int = 0, dist: Literal["normal", "t"] = "normal"):
    """
    Fit a GARCH(p,o,q) model on returns and forecast n_steps ahead variance.
    o=1 makes this GJR-GARCH (asymmetric: negative shocks raise volatility
    more than positive shocks of the same size — the "leverage effect").
    dist="t" uses Student's t errors instead of normal, for fatter tails.
    """
    scaled_returns = train_returns * 100
    
    model = arch_model(scaled_returns, vol="GARCH", p=p, o=o, q=q, dist=dist)
    fitted = model.fit(disp="off")

    forecast = fitted.forecast(horizon=n_steps, reindex=False)
    variance_forecast = forecast.variance.values[-1]

    daily_vol_forecast = np.sqrt(variance_forecast) / 100
    annualized_vol_forecast = daily_vol_forecast * np.sqrt(252)

    return annualized_vol_forecast


def garch_rolling_one_step(returns_series: np.ndarray, volatility_target: np.ndarray,
                            n_splits: int = 5, p: int = 1, q: int = 1, o: int = 0,
                            dist: Literal["normal", "t"] = "normal", refit_every: int = 5):
    """
    Rolling one-step-ahead GARCH forecasting, refitting periodically.
    Returns fold MSEs, their average, AND the full predictions array
    (aligned to volatility_target's index, NaN before the first fold starts)
    so it can be reused as a feature elsewhere.
    """
    n = len(returns_series)
    fold_size = n // (n_splits + 1)

    fold_mses = []
    all_predictions = np.full(len(volatility_target), np.nan)

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)

        if test_end > n:
            continue

        history_returns = list(returns_series[:train_end])
        test_targets = volatility_target[train_end:test_end]

        if len(test_targets) == 0:
            continue

        preds = []
        fitted_forecast_cache = None

        for step in range(len(test_targets)):
            if step % refit_every == 0:
                try:
                    scaled = np.array(history_returns) * 100
                    model = arch_model(scaled, vol="GARCH", p=p, o=o, q=q, dist=dist)
                    fitted = model.fit(disp="off")
                    forecast = fitted.forecast(horizon=refit_every, reindex=False)
                    variance_batch = forecast.variance.values[-1]
                    daily_vol_batch = np.sqrt(variance_batch) / 100
                    annualized_batch = daily_vol_batch * np.sqrt(252)
                    fitted_forecast_cache = list(annualized_batch)
                except Exception:
                    fitted_forecast_cache = None

            if fitted_forecast_cache:
                pred = fitted_forecast_cache.pop(0)
            else:
                pred = volatility_target[train_end + step - 1] if step > 0 else volatility_target[train_end - 1]

            preds.append(pred)
            all_predictions[train_end + step] = pred  # store aligned to original index

            next_idx = train_end + step
            if next_idx < len(returns_series):
                history_returns.append(returns_series[next_idx])

        fold_mse = compute_loss(np.array(test_targets), np.array(preds))
        fold_mses.append(fold_mse)
        model_name = f"GJR-GARCH({p},{o},{q})" if o > 0 else f"GARCH({p},{q})"
        print(f"Fold {i}: test_size={len(test_targets)}, {model_name} dist={dist} Test MSE={fold_mse:.6f}")

    avg_mse = np.mean(fold_mses) if fold_mses else float("nan")
    return fold_mses, avg_mse, all_predictions

from typing import Literal

def garch_vs_squared_returns(returns_series: np.ndarray, n_splits: int = 5,
                              p: int = 1, q: int = 1, o: int = 0,
                              dist: Literal["normal", "t"] = "normal", refit_every: int = 5):
    """
    Evaluate GARCH's one-step variance forecast against next-day squared
    returns directly, instead of the smoothed rolling volatility target.
    This isolates whether GARCH's weaker score elsewhere is a target-mismatch
    artifact rather than a real modeling failure.
    """
    n = len(returns_series)
    fold_size = n // (n_splits + 1)

    fold_mses = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        if test_end > n:
            continue

        history_returns = list(returns_series[:train_end])
        test_returns = returns_series[train_end:test_end]
        actual_squared = test_returns ** 2

        preds_variance = []
        fitted_forecast_cache = None

        for step in range(len(test_returns)):
            if step % refit_every == 0:
                try:
                    scaled = np.array(history_returns) * 100
                    model = arch_model(scaled, vol="GARCH", p=p, o=o, q=q, dist=dist)
                    fitted = model.fit(disp="off")
                    forecast = fitted.forecast(horizon=refit_every, reindex=False)
                    var_batch = forecast.variance.values[-1] / (100 ** 2)  # undo scaling
                    fitted_forecast_cache = list(var_batch)
                except Exception:
                    fitted_forecast_cache = None

            if fitted_forecast_cache:
                pred_var = fitted_forecast_cache.pop(0)
            else:
                pred_var = actual_squared[step - 1] if step > 0 else np.var(history_returns[-21:])

            preds_variance.append(pred_var)
            next_idx = train_end + step
            if next_idx < len(returns_series):
                history_returns.append(returns_series[next_idx])

        fold_mse = np.mean((np.array(preds_variance) - actual_squared) ** 2)
        fold_mses.append(fold_mse)
        print(f"Fold {i}: GARCH vs squared-returns MSE = {fold_mse:.8f}")

    avg_mse = np.mean(fold_mses) if fold_mses else float("nan")
    return fold_mses, avg_mse