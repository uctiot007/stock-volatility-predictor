import numpy as np
import matplotlib.pyplot as plt
import os


def plot_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, dates=None,
                               title: str = "Model", save_dir: str = "outputs/plots"):
    """
    Plot residuals (actual - predicted) over time and as a histogram.
    Patterns over time suggest missing structure; skew/fat tails in the
    histogram suggest the model handles extreme moves poorly.
    """
    residuals = y_true - y_pred
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    x_axis = dates if dates is not None else np.arange(len(residuals))
    axes[0].plot(x_axis, residuals, linewidth=0.8)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"{title}: Residuals Over Time")
    axes[0].set_ylabel("Actual - Predicted")

    axes[1].hist(residuals, bins=40, edgecolor="black", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title(f"{title}: Residual Distribution")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    fname = f"{save_dir}/residuals_{title.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    print(f"Residual plot saved to {fname}")
    plt.show()

    print(f"\n{title} residual stats:")
    print(f"  Mean: {residuals.mean():.6f} (should be near 0 if unbiased)")
    print(f"  Std:  {residuals.std():.6f}")
    print(f"  Skew: {((residuals - residuals.mean())**3).mean() / residuals.std()**3:.3f}")

    return residuals


from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def residual_autocorrelation_check(residuals: np.ndarray, lags: int = 20,
                                    save_path: str = "outputs/plots/residual_acf_pacf.png"):
    """
    ACF/PACF plots and Ljung-Box test on residuals. A significant Ljung-Box
    p-value (< 0.05) means residuals still have structure the model missed;
    a high p-value means residuals look like white noise (good — model has
    captured the linear structure available).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(residuals, lags=lags, ax=axes[0])
    plot_pacf(residuals, lags=lags, ax=axes[1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    lb_result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    p_value = lb_result["lb_pvalue"].values[0]
    print(f"\nLjung-Box test (lag={lags}): p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("→ Residuals show significant autocorrelation (model missed some structure)")
    else:
        print("→ Residuals look like white noise (no obvious remaining linear structure)")

    return p_value
