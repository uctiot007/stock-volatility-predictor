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
    n = len(returns_series)
    fold_size = n // (n_splits + 1)

    fold_mses = []

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
                    forecast_batch = fit_predict_garch(
                        np.array(history_returns), n_steps=refit_every, p=p, q=q, o=o, dist=dist
                    )
                    fitted_forecast_cache = list(forecast_batch)
                except Exception:
                    fitted_forecast_cache = None

            if fitted_forecast_cache:
                pred = fitted_forecast_cache.pop(0)
            else:
                pred = volatility_target[train_end + step - 1] if step > 0 else volatility_target[train_end - 1]

            preds.append(pred)
            next_idx = train_end + step
            if next_idx < len(returns_series):
                history_returns.append(returns_series[next_idx])

        fold_mse = compute_loss(np.array(test_targets), np.array(preds))
        fold_mses.append(fold_mse)
        model_name = f"GJR-GARCH({p},{o},{q})" if o > 0 else f"GARCH({p},{q})"
        print(f"Fold {i}: test_size={len(test_targets)}, {model_name} dist={dist} Test MSE={fold_mse:.6f}")

    avg_mse = np.mean(fold_mses) if fold_mses else float("nan")
    return fold_mses, avg_mse