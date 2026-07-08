import matplotlib.pyplot as plt
from src.data_loader import get_data
from src.features import compute_log_returns

df = get_data()
df = compute_log_returns(df)
df = df.dropna()

returns = df["log_return"]
squared_returns = returns ** 2

# Recreate your fold boundaries the same way walk_forward.py does
n = len(df)
n_splits = 5
fold_size = n // (n_splits + 1)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(df.index, returns, linewidth=0.7)
axes[0].set_title("Daily Log Returns")
axes[0].set_ylabel("Log Return")

axes[1].plot(df.index, squared_returns, linewidth=0.7, color="darkred")
axes[1].set_title("Squared Returns (proxy for realized variance/volatility shocks)")
axes[1].set_ylabel("Squared Return")

# Mark each fold's train/test boundary
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
plt.savefig("outputs/returns_and_squared_returns.png")
plt.show()