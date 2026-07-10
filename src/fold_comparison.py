import matplotlib.pyplot as plt
import os


def plot_fold_comparison(fold_results: dict, save_path: str = "outputs/plots/fold_comparison.png"):
    """
    fold_results: dict mapping method name -> list of per-fold MSEs
        e.g. {"Ridge": [0.00016, 0.00024, ...], "ARIMA": [...], "GARCH": [...]}
    All lists must have the same length (same number of folds).
    """
    n_folds = len(next(iter(fold_results.values())))
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(n_folds)
    width = 0.8 / len(fold_results)

    for idx, (method, mses) in enumerate(fold_results.items()):
        offset = (idx - len(fold_results) / 2) * width + width / 2
        ax.bar([xi + offset for xi in x], mses, width=width, label=method)

    ax.set_xticks(list(x))
    ax.set_xticklabels(fold_labels)
    ax.set_ylabel("Test MSE")
    ax.set_title("Per-Fold MSE by Method")
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Fold comparison plot saved to {save_path}")
    plt.show()