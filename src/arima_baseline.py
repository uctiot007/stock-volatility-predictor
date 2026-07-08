import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from src.model import compute_loss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="statsmodels")


def fit_predict_arima(train_series: np.ndarray, n_steps: int, order=(1, 0, 1)):
    """
    Fit ARIMA on train_series and forecast n_steps ahead in one shot.
    """
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=n_steps)
    return np.asarray(forecast)


def arima_walk_forward(volatility_series: np.ndarray, n_splits: int = 5, order=(1, 0, 1)):
    """
    Walk-forward validation for ARIMA using a single multi-step forecast per fold.
    NOTE: this is not directly comparable to the linear model, since it forecasts
    ~330 days ahead from one fit rather than getting fresh daily inputs.
    Kept here for reference/comparison against the rolling version below.
    """
    n = len(volatility_series)
    fold_size = n // (n_splits + 1)

    fold_mses = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)

        train_series = volatility_series[:train_end]
        test_series = volatility_series[train_end:test_end]

        if len(test_series) == 0:
            continue

        try:
            forecast = fit_predict_arima(train_series, n_steps=len(test_series), order=order)
            fold_mse = compute_loss(test_series, forecast)
        except Exception as e:
            print(f"Fold {i}: ARIMA failed to fit ({e}), skipping fold")
            continue

        fold_mses.append(fold_mse)
        print(f"Fold {i}: train_size={train_end}, test_size={len(test_series)}, ARIMA (multi-step) Test MSE={fold_mse:.6f}")

    avg_mse = np.mean(fold_mses) if fold_mses else float("nan")
    return fold_mses, avg_mse


def arima_rolling_one_step(volatility_series: np.ndarray, n_splits: int = 5, order=(1, 0, 1), refit_every: int = 1):
    """
    Rolling one-step-ahead ARIMA forecasting, refitting periodically within
    each fold. Directly comparable to the linear model, which effectively
    gets fresh lagged inputs every day.

    refit_every: refit the model every N days instead of daily, to keep
    runtime practical. refit_every=1 is the most accurate but slowest.
    """
    n = len(volatility_series)
    fold_size = n // (n_splits + 1)

    fold_mses = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)

        history = list(volatility_series[:train_end])
        test_series = volatility_series[train_end:test_end]

        if len(test_series) == 0:
            continue

        preds = []
        fitted = None

        for step, actual in enumerate(test_series):
            if fitted is None or step % refit_every == 0:
                try:
                    model = ARIMA(history, order=order)
                    fitted = model.fit()
                except Exception:
                    fitted = None

            if fitted is not None:
                try:
                    pred = fitted.forecast(steps=1)[0]
                except Exception:
                    pred = history[-1]
            else:
                pred = history[-1]  # fallback: naive

            preds.append(pred)
            history.append(actual)

        fold_mse = compute_loss(np.array(test_series), np.array(preds))
        fold_mses.append(fold_mse)
        print(f"Fold {i}: test_size={len(test_series)}, ARIMA (rolling, refit_every={refit_every}) Test MSE={fold_mse:.6f}")

    avg_mse = np.mean(fold_mses) if fold_mses else float("nan")
    return fold_mses, avg_mse