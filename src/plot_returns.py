import os
import matplotlib.pyplot as plt
from src.data_loader import get_data
from src.features import compute_log_returns


def plot_returns_and_squared_returns(n_splits: int = 5, save_path: str = "outputs/plots/returns_and_squared_returns.png"):
    df = get_data()
    df = compute_log_returns(df)
    df = df.dropna()

    returns = df["log_return"]
    squared_returns = returns ** 2

    n = len(df)
    fold_size = n // (n_splits + 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(df.index, returns, linewidth=0.7)
    axes[0].set_title("Daily Log Returns")
    axes[0].set_ylabel("Log Return")

    axes[1].plot(df.index, squared_returns, linewidth=0.7, color="darkred")
    axes[1].set_title("Squared Returns (proxy for realized variance/volatility shocks)")
    axes[1].set_ylabel("Squared Return")

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        if train_end < n:
            date_train_end = df.index[train_end]
            date_test_end = df.index[min(test_end, n - 1)]
            for ax in axes:
                ax.axvline(date_train_end, color="green", linestyle="--", alpha=0.6)
                ax.axvline(date_test_end, color="blue", linestyle=":", alpha=0.6)

    axes[1].set_xlabel("Date")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_returns_and_squared_returns()