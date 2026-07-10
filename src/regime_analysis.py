from typing import Optional
import numpy as np

def regime_split_evaluation(y_true: np.ndarray, y_pred: np.ndarray, threshold: Optional[float] = None):
    """
    Split test set into high/low volatility regimes (by median of actual
    volatility unless threshold given) and report MSE separately for each.
    """
    if threshold is None:
        threshold = float(np.median(y_true))

    high_mask = y_true > threshold
    low_mask = ~high_mask

    high_mse = np.mean((y_true[high_mask] - y_pred[high_mask]) ** 2)
    low_mse = np.mean((y_true[low_mask] - y_pred[low_mask]) ** 2)

    print(f"\nRegime split (threshold={threshold:.4f}):")
    print(f"  High-vol regime ({high_mask.sum()} days): MSE = {high_mse:.6f}")
    print(f"  Low-vol regime ({low_mask.sum()} days):  MSE = {low_mse:.6f}")
    print(f"  Ratio (high/low): {high_mse/low_mse:.2f}x")

    return {"high_vol_mse": high_mse, "low_vol_mse": low_mse, "threshold": threshold}